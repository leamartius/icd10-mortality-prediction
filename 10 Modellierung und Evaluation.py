#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELLIERUNG UND EVALUATION

In dem Skript werden die verschiedenen Feature-Sets des Haupt- und No-Leakage-Ansatzes 
sowie das CCSR-Modell erstellt und darauf XGBoost-Modelle trainiert. Die Hyperparameter 
werden mittels patientenbasierter Cross-Validation optimiert und der 
Threshold anhand von Out-of-Fold-Vorhersagen bestimmt. Anschließend werden die Modelle 
auf dem gemeinsamen Testdatensatz evaluiert und mittels Bootstrapping, Konfusionsmatrizen, 
Feature Importance, SHAP-Analysen, Interaktionsanalysen sowie weiteren Redundanz- und 
Feature-Gruppen-Analysen untersucht.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from scipy import sparse
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)
import shap
import matplotlib.pyplot as plt
import gc
from scipy.special import expit
from sklearn.metrics import ConfusionMatrixDisplay

# PFADE ANPASSEN!
path = "Pfad_zu_zuvor_gespeicherten_Daten/"
output_dir = "Pfad_zum_Ergebnisordner"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Dateien für Modelle der Hauptanalyse
# ---------------------------------------------------------------------------
baseline_path = os.path.join(path, "baseline_model")
full_path = os.path.join(path, "full_model")

baseline_no_icd10 = pd.read_csv(path + "baseline_no_icd10_reduced.csv")
proc_metrics = pd.read_csv(path + "procedure_metrics.csv")

# ---------------------------------------------------------------------------
# Dateien für Modelle ohne Temporal Leakage
# ---------------------------------------------------------------------------
baseline_no_leakage_path = os.path.join(path, "baseline_no_leakage_last_stay")
full_no_leakage_path = os.path.join(path, "full_model_no_leakage_last_stay")

# ---------------------------------------------------------------------------
# CCSR-Datei
# ---------------------------------------------------------------------------
ccsr_multihot = pd.read_pickle(
    os.path.join(path, "ccsr_multihot.pkl")
)

print("\n=== Geladener CCSR-Datensatz ===")
print("Shape:", ccsr_multihot.shape)
print("Erste Spalten:", ccsr_multihot.columns[:10].tolist())

# ---------------------------------------------------------------------------
# Speicherort für Ergebnisse
# ---------------------------------------------------------------------------

model_output_dir = os.path.join(
    output_dir,
    "xgb_model_pkl_files"
)
os.makedirs(model_output_dir, exist_ok=True)


# =========================================================
# Funktion zum Laden eines Modells
# =========================================================

def load_model_data(model_path):
    
    # -----------------------------
    # JSON-Dateien
    # -----------------------------
    with open(os.path.join(model_path, "base_cols.json"), "r") as f:
        base_cols = json.load(f)

    with open(os.path.join(model_path, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    with open(os.path.join(model_path, "list_columns.json"), "r") as f:
        list_columns = json.load(f)

    # -----------------------------
    # Pickle-Datei
    # -----------------------------
    with open(os.path.join(model_path, "mlb_dict.pkl"), "rb") as f:
        mlb_dict = pickle.load(f)

    # -----------------------------
    # NumPy-Dateien
    # -----------------------------
    subject_id = np.load(os.path.join(model_path, "subject_id.npy"))
    hadm_id = np.load(os.path.join(model_path, "hadm_id.npy"))

    y = np.load(os.path.join(model_path, "y.npy"))

    # -----------------------------
    # Sparse Matrix
    # -----------------------------
    X_sparse = sparse.load_npz(
        os.path.join(model_path, "X_sparse.npz")
    )

    # -----------------------------
    # Rückgabe
    # -----------------------------
    return {
        "X_sparse": X_sparse,
        "y": y,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "base_cols": base_cols,
        "feature_names": feature_names,
        "list_columns": list_columns,
        "mlb_dict": mlb_dict
    }

# =========================================================
# Hilfsfunktionen: Feature entfernen / Feature-Sets vorbereiten
# =========================================================

def remove_features_by_name_or_prefix(
    X,
    feature_names,
    exact_features=None,
    feature_prefixes=None
):
    """
    Entfernt mehrere Features aus einer Matrix anhand von:

    1. exakten Feature-Namen
    2. Feature-Präfixen

    Funktioniert für Sparse-Matrizen, DataFrames und NumPy-Arrays.
    Nicht vorhandene Features werden ignoriert.
    """

    if exact_features is None:
        exact_features = []

    if feature_prefixes is None:
        feature_prefixes = []

    feature_names = list(feature_names)
    exact_features = set(exact_features)
    feature_prefixes = tuple(feature_prefixes)

    features_to_remove = [
        feature
        for feature in feature_names
        if (
            feature in exact_features
            or (
                feature_prefixes
                and feature.startswith(feature_prefixes)
            )
        )
    ]

    keep_indices = [
        index
        for index, feature in enumerate(feature_names)
        if feature not in features_to_remove
    ]

    if sparse.issparse(X):
        X_new = X[:, keep_indices].tocsr()
    elif hasattr(X, "iloc"):
        X_new = X.iloc[:, keep_indices]
    else:
        X_new = X[:, keep_indices]

    feature_names_new = [
        feature_names[index]
        for index in keep_indices
    ]

    return X_new, feature_names_new, features_to_remove


def remove_gender_from_df(df):
    return df.drop(columns=["gender_M"], errors="ignore")


def prepare_no_icd10_feature_set(df):
    """
    Bereitet ein tabellarisches Feature-Set ohne ICD-10 Multi-Hot-Matrix vor.
    Entfernt hospital_expire_flag, subject_id und hadm_id aus X.
    gender_M wird vorher entfernt.
    """

    df = remove_gender_from_df(df.copy())

    y_current = df["hospital_expire_flag"].values
    subject_id_current = df["subject_id"].values
    hadm_id_current = df["hadm_id"].values

    X_df = df.drop(
        columns=[
            "hospital_expire_flag",
            "subject_id",
            "hadm_id"
        ],
        errors="ignore"
    )

    feature_names = X_df.columns.tolist()
    X = X_df.values

    return X, y_current, subject_id_current, hadm_id_current, feature_names


def prepare_basic_feature_set(df):

    # Erstellt das einfache Baseline-Set ohne historische Features.

    df = remove_gender_from_df(df.copy())

    historical_features_to_remove = [
        "n_prev_admissions", 
        "n_prev_emergency_admissions",
        "time_since_last_discharge_days", 
        "had_admission_last_30d",
        "max_past_los", 
        "avg_past_los",
        "has_previous_admission"
    ]

    historical_features_to_remove += [
        col
        for col in df.columns
        if col.startswith("last_")
    ]

    df = df.drop(
        columns=historical_features_to_remove,
        errors="ignore"
    )

    X_df = df.drop(
        columns=[
            "hospital_expire_flag",
            "subject_id",
            "hadm_id"
        ],
        errors="ignore"
    )

    feature_names = X_df.columns.tolist()
    X = X_df.values

    return X, feature_names


def prepare_proc_feature_blocks(proc_metrics_input, hadm_order, label=""):
    """
    Richtet proc_metrics an der hadm_id-Reihenfolge aus und trennt:
    1. pcs_body_system-Features
    2. übrige Prozedurmetriken
    """

    proc_metrics_aligned = (
        pd.DataFrame({"hadm_id": hadm_order})
        .merge(
            proc_metrics_input,
            on="hadm_id",
            how="left"
        )
    )

    print(f"\n=== Alignment Checks proc_metrics {label} ===")
    print("hadm_order Länge:", len(hadm_order))
    print("proc_metrics_aligned shape:", proc_metrics_aligned.shape)

    print(
        "Reihenfolge hadm_id korrekt:",
        np.array_equal(
            proc_metrics_aligned["hadm_id"].values,
            hadm_order
        )
    )

    print("\nFehlende Werte nach Alignment:")
    print(proc_metrics_aligned.isna().sum())

    proc_feature_df = (
        proc_metrics_aligned
        .drop(columns=["hadm_id"])
        .replace([np.inf, -np.inf], np.nan)
    )

    proc_feature_df = proc_feature_df.astype(float)

    pcs_body_system_cols = [
        col for col in proc_feature_df.columns
        if col.startswith("pcs_body_system")
    ]

    proc_metric_cols_without_body_system = [
        col for col in proc_feature_df.columns
        if not col.startswith("pcs_body_system")
    ]

    print(f"\n=== proc_metrics Aufteilung {label} ===")
    print("pcs_body_system Features:", len(pcs_body_system_cols))
    print("Übrige Behandlungsmetriken:", len(proc_metric_cols_without_body_system))

    pcs_body_system_df = proc_feature_df[pcs_body_system_cols].copy()
    pcs_body_system_feature_names = pcs_body_system_df.columns.tolist()

    X_pcs_body_system = sparse.csr_matrix(
        pcs_body_system_df.values
    )

    proc_metric_df_without_body_system = (
        proc_feature_df[proc_metric_cols_without_body_system]
        .copy()
    )

    proc_metric_feature_names = (
        proc_metric_df_without_body_system.columns.tolist()
    )

    X_proc_metrics_without_body_system = sparse.csr_matrix(
        proc_metric_df_without_body_system.values
    )

    return (
        X_pcs_body_system,
        pcs_body_system_feature_names,
        X_proc_metrics_without_body_system,
        proc_metric_feature_names
    )

def align_df_to_reference_hadm(df, reference_hadm_id, label=""):
    """
    Filtert und sortiert einen Dataframe exakt auf die hadm_id-Reihenfolge
    des ursprünglichen Full-Modells.
    """

    if df["hadm_id"].duplicated().any():
        duplicated = df.loc[df["hadm_id"].duplicated(), "hadm_id"].head()
        raise ValueError(
            f"{label}: Doppelte hadm_id gefunden, z.B. {duplicated.tolist()}"
        )

    available_hadm = set(df["hadm_id"].values)
    reference_hadm = set(reference_hadm_id)

    missing_in_df = sorted(reference_hadm - available_hadm)
    extra_in_df = sorted(available_hadm - reference_hadm)

    print(f"\n=== Alignment Dataframe: {label} ===")
    print("Zeilen vorher:", len(df))
    print("Referenz-Zeilen:", len(reference_hadm_id))
    print("Fehlen im Dataframe:", len(missing_in_df))
    print("Zusätzlich im Dataframe:", len(extra_in_df))

    if len(missing_in_df) > 0:
        raise ValueError(
            f"{label}: Es fehlen hadm_id aus der Referenz. "
            f"Beispiele: {missing_in_df[:10]}"
        )

    df_aligned = (
        df
        .set_index("hadm_id")
        .loc[reference_hadm_id]
        .reset_index()
    )

    assert np.array_equal(df_aligned["hadm_id"].values, reference_hadm_id)

    print("Zeilen nach Alignment:", len(df_aligned))

    return df_aligned


# =========================================================
# Baseline ohne ICD-10 vorbereiten
# =========================================================

X_baseline_no_icd10, y_baseline_no_icd10, subject_id_baseline_no_icd10, hadm_id_baseline_no_icd10, baseline_no_icd10_feature_names = prepare_no_icd10_feature_set(
    baseline_no_icd10
)

# =========================================================
# Baseline-Modell mit ICD-10 laden
# =========================================================

baseline_data = load_model_data(baseline_path)

X_baseline = baseline_data["X_sparse"]
y_baseline = baseline_data["y"]
subject_id_baseline = baseline_data["subject_id"]
hadm_id_baseline = baseline_data["hadm_id"]

baseline_base_cols = baseline_data["base_cols"]
baseline_feature_names = baseline_data["feature_names"]
baseline_list_columns = baseline_data["list_columns"]
baseline_mlb_dict = baseline_data["mlb_dict"]

# gender_M aus ursprünglichem Baseline-Modell entfernen
X_baseline, baseline_feature_names, _ = remove_features_by_name_or_prefix(
    X=X_baseline,
    feature_names=baseline_feature_names,
    exact_features=["gender_M"]
)

baseline_base_cols = [
    col for col in baseline_base_cols
    if col != "gender_M"
]


# =========================================================
# Baseline-Modell mit ICD-10 ohne Temporal Leakage laden
# =========================================================

baseline_no_leakage_data = load_model_data(baseline_no_leakage_path)

X_baseline_no_leakage = baseline_no_leakage_data["X_sparse"]
y_baseline_no_leakage = baseline_no_leakage_data["y"]
subject_id_baseline_no_leakage = baseline_no_leakage_data["subject_id"]
hadm_id_baseline_no_leakage = baseline_no_leakage_data["hadm_id"]

baseline_no_leakage_base_cols = baseline_no_leakage_data["base_cols"]
baseline_no_leakage_feature_names = baseline_no_leakage_data["feature_names"]
baseline_no_leakage_list_columns = baseline_no_leakage_data["list_columns"]
baseline_no_leakage_mlb_dict = baseline_no_leakage_data["mlb_dict"]

# Sicherheitscheck: falls gender_M doch noch enthalten ist
X_baseline_no_leakage, baseline_no_leakage_feature_names, _ = remove_features_by_name_or_prefix(
    X=X_baseline_no_leakage,
    feature_names=baseline_no_leakage_feature_names,
    exact_features=["gender_M"]
)

baseline_no_leakage_base_cols = [
    col
    for col in baseline_no_leakage_base_cols
    if col != "gender_M"
]

# =========================================================
# Vollständiges Modell laden
# =========================================================

full_data = load_model_data(full_path)

X_full = full_data["X_sparse"]
y_full = full_data["y"]
subject_id_full = full_data["subject_id"]
hadm_id_full = full_data["hadm_id"]

full_base_cols = full_data["base_cols"]
full_feature_names = full_data["feature_names"]
full_list_columns = full_data["list_columns"]
full_mlb_dict = full_data["mlb_dict"]

# Redundante Variablen aus ursprünglichem Modell entfernen:
# - gender_M  entfernen, weil redundant zu gender_F
# - leaf_count entfernen, weil redundant zur Codeanzahl, weil fast alle 
#   Codes im Datnsatz Blattknoten sind
# - subtree_rel_log_std entfernen, weil es nicht genug eigenständige Information
#   ggü. subtree_rel_log_mean besitzt und schwerer zu interpretieren ist
# - die historischen Verteilungsmerkmale werden entfernt, weil sie in diesem Ansatz inkonsistent sind

features_to_remove_from_full = [
    "gender_M",
    "leaf_count",
    "subtree_rel_log_std",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters"
]

feature_prefixes_to_remove_from_full = [
    "prev_icd10_chapter_"
]

(
    X_full,
    full_feature_names,
    removed_features_from_full
) = remove_features_by_name_or_prefix(
    X=X_full,
    feature_names=full_feature_names,
    exact_features=features_to_remove_from_full,
    feature_prefixes=feature_prefixes_to_remove_from_full
)

# base_cols ebenfalls konsistent bereinigen
full_base_cols = [
    col
    for col in full_base_cols
    if (
        col not in features_to_remove_from_full
        and not col.startswith(
            tuple(feature_prefixes_to_remove_from_full)
        )
    )
]

print("\n=== Entfernte Features aus full_model ===")
print("Anzahl entfernt:", len(removed_features_from_full))
print("Entfernte Features:")
print(removed_features_from_full)

remaining_unwanted_features_full = [
    feature
    for feature in full_feature_names
    if (
        feature in features_to_remove_from_full
        or feature.startswith(
            tuple(feature_prefixes_to_remove_from_full)
        )
    )
]

print(
    "Unerwünschte Features noch enthalten:",
    remaining_unwanted_features_full
)

assert not remaining_unwanted_features_full, (
    "Im full_model sind weiterhin auszuschließende Features enthalten."
)

assert X_full.shape[1] == len(full_feature_names), (
    "Spaltenanzahl von X_full stimmt nach der Entfernung nicht "
    "mit full_feature_names überein."
)

# =========================================================
# Vollständiges Modell ohne Temporal Leakage laden
# =========================================================

full_no_leakage_data = load_model_data(full_no_leakage_path)

X_full_no_leakage = full_no_leakage_data["X_sparse"]
y_full_no_leakage = full_no_leakage_data["y"]
subject_id_full_no_leakage = full_no_leakage_data["subject_id"]
hadm_id_full_no_leakage = full_no_leakage_data["hadm_id"]

full_no_leakage_base_cols = full_no_leakage_data["base_cols"]
full_no_leakage_feature_names = full_no_leakage_data["feature_names"]
full_no_leakage_list_columns = full_no_leakage_data["list_columns"]
full_no_leakage_mlb_dict = full_no_leakage_data["mlb_dict"]

# Nicht mehr zu verwendende Features entfernen
features_to_remove_from_full_no_leakage = [
    "gender_M",
    "last_leaf_count",
    "last_subtree_rel_log_std",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters"
]

feature_prefixes_to_remove_from_full_no_leakage = [
    "prev_icd10_chapter_"
]

(
    X_full_no_leakage,
    full_no_leakage_feature_names,
    removed_features_from_full_no_leakage
) = remove_features_by_name_or_prefix(
    X=X_full_no_leakage,
    feature_names=full_no_leakage_feature_names,
    exact_features=features_to_remove_from_full_no_leakage,
    feature_prefixes=feature_prefixes_to_remove_from_full_no_leakage
)

# base_cols ebenfalls konsistent bereinigen
full_no_leakage_base_cols = [
    col
    for col in full_no_leakage_base_cols
    if (
        col not in features_to_remove_from_full_no_leakage
        and not col.startswith(
            tuple(feature_prefixes_to_remove_from_full_no_leakage)
        )
    )
]

print("\n=== Entfernte Features aus full_model_no_leakage ===")
print(
    "Anzahl entfernt:",
    len(removed_features_from_full_no_leakage)
)
print("Entfernte Features:")
print(removed_features_from_full_no_leakage)

remaining_unwanted_features_full_no_leakage = [
    feature
    for feature in full_no_leakage_feature_names
    if (
        feature in features_to_remove_from_full_no_leakage
        or feature.startswith(
            tuple(feature_prefixes_to_remove_from_full_no_leakage)
        )
    )
]

print(
    "Unerwünschte Features noch enthalten:",
    remaining_unwanted_features_full_no_leakage
)

assert not remaining_unwanted_features_full_no_leakage, (
    "Im full_model_no_leakage sind weiterhin "
    "auszuschließende Features enthalten."
)

assert (
    X_full_no_leakage.shape[1]
    == len(full_no_leakage_feature_names)
), (
    "Spaltenanzahl von X_full_no_leakage stimmt nach der "
    "Entfernung nicht mit full_no_leakage_feature_names überein."
)


# =========================================================
# No-Leakage-Daten auf Reihenfolge des Hauptansatzes ausrichten
# =========================================================

reference_hadm_id = hadm_id_full.copy()


def reorder_matrix_to_reference_hadm(
    X,
    y_current,
    subject_current,
    hadm_current,
    reference_hadm_id,
    label=""
):
    """
    Prüft, dass exakt dieselben Aufenthalte enthalten sind,
    und bringt sie anschließend in dieselbe Reihenfolge.
    """

    assert len(np.unique(hadm_current)) == len(hadm_current), (
        f"{label}: hadm_id sind nicht eindeutig."
    )

    # Es dürfen weder Aufenthalte fehlen noch zusätzliche vorhanden sein
    assert set(hadm_current) == set(reference_hadm_id), (
        f"{label}: Aufenthaltsmenge stimmt nicht mit der "
        "finalen Analysepopulation überein."
    )

    # Falls bereits gleiche Reihenfolge: nichts tun
    if np.array_equal(hadm_current, reference_hadm_id):
        print(f"{label}: Aufenthalte bereits korrekt ausgerichtet.")
        return X, y_current, subject_current, hadm_current

    # Sonst nur Reihenfolge korrigieren
    hadm_to_position = {
        hadm: i
        for i, hadm in enumerate(hadm_current)
    }

    row_idx = np.array([
        hadm_to_position[hadm]
        for hadm in reference_hadm_id
    ])

    X = X[row_idx]
    y_current = y_current[row_idx]
    subject_current = subject_current[row_idx]
    hadm_current = hadm_current[row_idx]

    assert np.array_equal(
        hadm_current,
        reference_hadm_id
    )

    print(f"{label}: Reihenfolge an Hauptansatz angepasst.")

    return (
        X,
        y_current,
        subject_current,
        hadm_current
    )        

# ---------------------------------------------------------
# No-Leakage-Baseline mit ICD-10 - Reihenfolge anpassen
# ---------------------------------------------------------
(
    X_baseline_no_leakage,
    y_baseline_no_leakage,
    subject_id_baseline_no_leakage,
    hadm_id_baseline_no_leakage
) = reorder_matrix_to_reference_hadm(
    X=X_baseline_no_leakage,
    y_current=y_baseline_no_leakage,
    subject_current=subject_id_baseline_no_leakage,
    hadm_current=hadm_id_baseline_no_leakage,
    reference_hadm_id=reference_hadm_id,
    label="baseline_with_icd10_no_leakage"
)

# ---------------------------------------------------------
# No-Leakage-Full-Modell - Reihenfolge anpassen
# ---------------------------------------------------------
(
    X_full_no_leakage,
    y_full_no_leakage,
    subject_id_full_no_leakage,
    hadm_id_full_no_leakage
) = reorder_matrix_to_reference_hadm(
    X=X_full_no_leakage,
    y_current=y_full_no_leakage,
    subject_current=subject_id_full_no_leakage,
    hadm_current=hadm_id_full_no_leakage,
    reference_hadm_id=reference_hadm_id,
    label="full_no_leakage"
)    
    
    
# =========================================================
# CCSR-Feature-Set vorbereiten
# =========================================================

def prepare_ccsr_multihot_feature_set(
    ccsr_df,
    basic_df,
    reference_hadm_id,
    expected_y=None,
    expected_subject_id=None
):
    """
    Erstellt das CCSR-Benchmark-Feature-Set aus:
    - denselben Basisvariablen wie baseline_no_icd10
    - den bereits vorbereiteten CCSR-Multi-Hot-Features

    Sowohl die Basisdaten als auch die CCSR-Daten werden exakt auf die
    hadm_id-Reihenfolge von reference_hadm_id ausgerichtet.
    """

    ccsr_df = ccsr_df.copy()
    basic_df = basic_df.copy()

    # -----------------------------------------------------
    # hadm_id sicherstellen
    # -----------------------------------------------------

    if "hadm_id" not in ccsr_df.columns:
        if ccsr_df.index.name == "hadm_id":
            ccsr_df = ccsr_df.reset_index()
        else:
            raise ValueError(
                "CCSR-Datensatz enthält keine hadm_id-Spalte "
                "und keinen hadm_id-Index."
            )

    # -----------------------------------------------------
    # Doppelte Aufenthalte ausschließen
    # -----------------------------------------------------

    if ccsr_df["hadm_id"].duplicated().any():
        duplicated_hadm = (
            ccsr_df.loc[
                ccsr_df["hadm_id"].duplicated(),
                "hadm_id"
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Der CCSR-Datensatz sollte genau eine Zeile pro Aufenthalt "
            f"enthalten. Doppelte hadm_id, z. B.: {duplicated_hadm}"
        )

    # -----------------------------------------------------
    # Datentyp der hadm_id angleichen
    # -----------------------------------------------------

    reference_hadm_id = np.asarray(reference_hadm_id)

    ccsr_df["hadm_id"] = pd.to_numeric(
        ccsr_df["hadm_id"],
        errors="raise"
    ).astype(reference_hadm_id.dtype)

    basic_df["hadm_id"] = pd.to_numeric(
        basic_df["hadm_id"],
        errors="raise"
    ).astype(reference_hadm_id.dtype)

    # -----------------------------------------------------
    # CCSR-Daten auf Referenz ausrichten
    # -----------------------------------------------------

    ccsr_aligned = align_df_to_reference_hadm(
        df=ccsr_df,
        reference_hadm_id=reference_hadm_id,
        label="CCSR Multi-Hot"
    )

    # -----------------------------------------------------
    # Basisdaten auf dieselbe Referenz ausrichten
    # -----------------------------------------------------

    basic_aligned = align_df_to_reference_hadm(
        df=basic_df,
        reference_hadm_id=reference_hadm_id,
        label="CCSR Basisvariablen"
    )

    # -----------------------------------------------------
    # Dieselben Basisvariablen wie baseline_no_icd10 erzeugen
    # -----------------------------------------------------
    
    (
        X_base,
        y_base,
        subject_id_base,
        hadm_id_base,
        base_feature_names
    ) = prepare_no_icd10_feature_set(
        basic_aligned
    )
    
    X_base = sparse.csr_matrix(X_base)

    assert np.array_equal(
        hadm_id_base,
        reference_hadm_id
    ), "hadm_id-Reihenfolge der Basisdaten stimmt nicht."
    
    if expected_y is not None:
        assert np.array_equal(
            y_base,
            expected_y
        ), "Outcome-Reihenfolge der Basisdaten stimmt nicht."
    
    if expected_subject_id is not None:
        assert np.array_equal(
            subject_id_base,
            expected_subject_id
        ), "subject_id-Reihenfolge der Basisdaten stimmt nicht."
    
    # -----------------------------------------------------
    # ID-, Outcome- und andere Nicht-Feature-Spalten entfernen
    # -----------------------------------------------------

    ccsr_non_feature_cols = [
        "hadm_id",
        "subject_id",
        "hospital_expire_flag"
    ]

    X_ccsr_df = ccsr_aligned.drop(
        columns=ccsr_non_feature_cols,
        errors="ignore"
    )

    # -----------------------------------------------------
    # Nur numerische Multi-Hot-Features zulassen
    # -----------------------------------------------------

    non_numeric_cols = X_ccsr_df.select_dtypes(
        exclude=[np.number, "bool"]
    ).columns.tolist()

    if non_numeric_cols:
        raise ValueError(
            "Im CCSR-Feature-Set sind nichtnumerische Spalten enthalten: "
            f"{non_numeric_cols[:20]}"
        )

    # Bool in 0/1 umwandeln
    bool_cols = X_ccsr_df.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_cols) > 0:
        X_ccsr_df[bool_cols] = X_ccsr_df[bool_cols].astype(np.uint8)

    # -----------------------------------------------------
    # Unendliche und fehlende Werte prüfen
    # -----------------------------------------------------

    X_ccsr_df = X_ccsr_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    n_missing = int(X_ccsr_df.isna().sum().sum())

    if n_missing > 0:
        raise ValueError(
            f"Die CCSR-Matrix enthält {n_missing} fehlende Werte. "
            "Prüfen, ob diese tatsächlich als 0 interpretiert werden dürfen."
        )

    # -----------------------------------------------------
    # Prüfen, ob es wirklich Multi-Hot-Features sind
    # -----------------------------------------------------

    unique_values = np.unique(X_ccsr_df.values)

    if not np.isin(unique_values, [0, 1]).all():
        print(
            "\nWARNUNG: Die CCSR-Spalten enthalten nicht ausschließlich 0/1."
        )
        print(
            "Beispiele vorhandener Werte:",
            unique_values[:20]
        )

    ccsr_feature_names = X_ccsr_df.columns.tolist()

    X_ccsr_codes = sparse.csr_matrix(
        X_ccsr_df.astype(np.float32).values
    )

    # -----------------------------------------------------
    # Basisvariablen + CCSR-Features
    # -----------------------------------------------------

    X_ccsr_model = sparse.hstack(
        [
            X_base,
            X_ccsr_codes
        ],
        format="csr"
    )
    
    ccsr_model_feature_names = (
        base_feature_names
        + ccsr_feature_names
    )
    
    # -----------------------------------------------------
    # Alignment-Checks
    # -----------------------------------------------------

    assert np.array_equal(
        ccsr_aligned["hadm_id"].values,
        reference_hadm_id
    )

    assert np.array_equal(
        basic_aligned["hadm_id"].values,
        reference_hadm_id
    )

    assert X_ccsr_model.shape[0] == len(reference_hadm_id)

    assert (
        X_ccsr_model.shape[1]
        == len(ccsr_model_feature_names)
    )

    # -----------------------------------------------------
    # Übersicht
    # -----------------------------------------------------

    n_stays_without_ccsr = int(
        (X_ccsr_df.sum(axis=1) == 0).sum()
    )

    print("\n=== CCSR-Modell vorbereitet ===")
    print("Aufenthalte:", X_ccsr_model.shape[0])
    print("Basisfeatures:", len(base_feature_names))
    print("CCSR-Features:", len(ccsr_feature_names))
    print("Gesamtfeatures:", X_ccsr_model.shape[1])
    print(
        "Aufenthalte ohne gesetzte CCSR-Kategorie:",
        n_stays_without_ccsr
    )
    
    return (
        X_ccsr_model,
        ccsr_model_feature_names,
        base_feature_names,
        ccsr_feature_names,
        ccsr_aligned
    )

    
# =========================================================
# Checks
# =========================================================

print("=== Baseline-Modell ohne ICD-10 ===")
print("X shape:", X_baseline_no_icd10.shape)
print("y shape:", y_baseline_no_icd10.shape)
print("subject_id shape:", subject_id_baseline_no_icd10.shape)

print("\n=== Baseline-Modell mit ICD-10 ===")
print("X shape:", X_baseline.shape)
print("y shape:", y_baseline.shape)
print("subject_id shape:", subject_id_baseline.shape)
print("hadm_id shape:", hadm_id_baseline.shape)

print("\n=== Baseline-Modell mit ICD-10, no leakage ===")
print("X shape:", X_baseline_no_leakage.shape)
print("y shape:", y_baseline_no_leakage.shape)
print("subject_id shape:", subject_id_baseline_no_leakage.shape)
print("hadm_id shape:", hadm_id_baseline_no_leakage.shape)

print("\n=== Vollständiges Modell ===")
print("X shape:", X_full.shape)
print("y shape:", y_full.shape)
print("subject_id shape:", subject_id_full.shape)
print("hadm_id shape:", hadm_id_full.shape)

print("\n=== Vollständiges Modell, no leakage ===")
print("X shape:", X_full_no_leakage.shape)
print("y shape:", y_full_no_leakage.shape)
print("subject_id shape:", subject_id_full_no_leakage.shape)
print("hadm_id shape:", hadm_id_full_no_leakage.shape)

assert len(hadm_id_full_no_leakage) == len(hadm_id_full)

assert np.array_equal(
    hadm_id_full_no_leakage,
    hadm_id_full
)

print(
    "No-Leakage-Full-Modell enthält dieselben", len(hadm_id_full),
    "Aufenthalte wie der Hauptansatz und ist identisch ausgerichtet."
)


# =========================================================
# Testen, ob y, subject_id und hadm_id bei allen Feature-Sets identisch sind
# =========================================================

objects_to_check = {
    "baseline_no_icd10": (
        y_baseline_no_icd10,
        subject_id_baseline_no_icd10,
        hadm_id_baseline_no_icd10
    ),
    "baseline_with_icd10": (
        y_baseline,
        subject_id_baseline,
        hadm_id_baseline
    ),
    "baseline_with_icd10_no_leakage": (
        y_baseline_no_leakage,
        subject_id_baseline_no_leakage,
        hadm_id_baseline_no_leakage
    ),
    "full": (
        y_full,
        subject_id_full,
        hadm_id_full
    ),
    "full_no_leakage": (
        y_full_no_leakage,
        subject_id_full_no_leakage,
        hadm_id_full_no_leakage
    )
}

for name, (y_current, subject_current, hadm_current) in objects_to_check.items():
    print(f"\nCheck gegen full: {name}")
    print("y identisch:", np.array_equal(y_current, y_full))
    print("subject_id identisch:", np.array_equal(subject_current, subject_id_full))
    print("hadm_id identisch:", np.array_equal(hadm_current, hadm_id_full))

    assert np.array_equal(y_current, y_full)
    assert np.array_equal(subject_current, subject_id_full)
    assert np.array_equal(hadm_current, hadm_id_full)


# gleiches y, subject_id und hadm_id für alle Feature-Sets verwenden
y = y_full
subject_id = subject_id_full
hadm_id = hadm_id_full


# =========================================================
# Baseline ohne historische Features
# nur simple Features aus MIMIC
# =========================================================

X_baseline_basic, baseline_basic_feature_names = prepare_basic_feature_set(
    baseline_no_icd10
)


# =========================================================
# Semantik-Modelle vorbereiten
# =========================================================

# -------------------------------------------
# Diagnose-Strukturmetriken: gemeinsame Basisnamen
# -------------------------------------------

diagnosis_structure_metric_base_names = [
    "depth_mean",
    "depth_std",
    "leaf_share",
    #"leaf_count",
    "siblings_max",
    "siblings_q25",
    "siblings_q75",
    "siblings_median",
    "siblings_mean",
    "siblings_std",
    "subtree_rel_log_max",
    "subtree_rel_log_mean",
    #"subtree_rel_log_std",
    "subtree_rel_log_internal_mean",
    "siblings_3cat_share_high",
    "siblings_3cat_share_low",
    "siblings_3cat_share_medium",
    "depth_ord_share_1.0",
    "depth_ord_share_2.0",
    "depth_ord_share_3.0",
    "depth_ord_share_4.0",
    "depth_ord_share_5.0"
]

# Diagnose-Strukturmetriken im ursprünglichen Ansatz
diagnosis_structure_metrics_original = diagnosis_structure_metric_base_names.copy()

# Diagnose-Strukturmetriken im No-Leakage-Ansatz
diagnosis_structure_metrics_no_leakage = [
    "last_" + metric
    for metric in diagnosis_structure_metric_base_names
]


# -------------------------------------------
# Nur tatsächlich vorhandene No-Leakage-Diagnosemetriken behalten
# -------------------------------------------
# Dadurch bricht der Code nicht ab, falls einzelne Features im Modell fehlen.

diagnosis_structure_metrics_no_leakage = [
    feature
    for feature in diagnosis_structure_metrics_no_leakage
    if feature in full_no_leakage_feature_names
]

print("\n=== Diagnose-Strukturmetriken ===")
print("Original:", len([
    f for f in diagnosis_structure_metrics_original
    if f in full_feature_names
]))
print("No Leakage:", len(diagnosis_structure_metrics_no_leakage))


# -------------------------------------------
# Prozedurmetriken im No-Leakage-Ansatz
# -------------------------------------------
# Diese sind bereits in full_model_no_leakage enthalten.
# Deshalb werden sie NICHT aus einer separaten Datei geladen.

procedure_metric_base_names_no_leakage = [
    "proc_body_system_diversity",
    "proc_intervention_diversity",
    "proc_invasiveness_share_high",
    "proc_invasiveness_share_medium",
    "proc_invasiveness_share_low",
    "proc_invasiveness_share_non_invasive",
    "proc_n_high_invasive",
    "proc_has_section_0_procedure"
]

procedure_metric_features_no_leakage = [
    "last_" + metric
    for metric in procedure_metric_base_names_no_leakage
]

missing_procedure_metrics_no_leakage = [
    feature
    for feature in procedure_metric_features_no_leakage
    if feature not in full_no_leakage_feature_names
]

assert not missing_procedure_metrics_no_leakage, (
    "Folgende erwartete No-Leakage-Behandlungsmetriken fehlen: "
    f"{missing_procedure_metrics_no_leakage}"
)

print("\n=== No-Leakage-Prozedurmetriken im full_model_no_leakage ===")
print("Anzahl:", len(procedure_metric_features_no_leakage))
print(procedure_metric_features_no_leakage)

# -------------------------------------------
# Hilfsfunktionen
# -------------------------------------------

def get_feature_indices(feature_names, features_to_keep):
    """
    Gibt die Spaltenindizes für eine gegebene Feature-Namen-Liste zurück.
    """

    feature_to_idx = {
        feature: i
        for i, feature in enumerate(feature_names)
    }

    return [
        feature_to_idx[feature]
        for feature in features_to_keep
    ]


def subset_features(X, feature_names, features_to_keep):
    """
    Subsettet X anhand einer Liste zu behaltender Feature-Namen.
    """

    indices = get_feature_indices(
        feature_names=feature_names,
        features_to_keep=features_to_keep
    )

    X_subset = X[:, indices]

    return X_subset, features_to_keep


def remove_features_from_sparse(X, feature_names, features_to_remove):
    """
    Entfernt Features anhand ihrer Namen aus einer Sparse Matrix.
    Nicht vorhandene Features werden ignoriert.
    """

    features_to_remove = set(features_to_remove)

    features_to_keep = [
        feature
        for feature in feature_names
        if feature not in features_to_remove
    ]

    return subset_features(
        X=X,
        feature_names=feature_names,
        features_to_keep=features_to_keep
    )


def select_features_from_sparse(X, feature_names, features_to_select):
    """
    Selektiert Features anhand ihrer Namen aus einer Sparse Matrix.
    Nicht vorhandene Features werden ignoriert.
    """

    features_to_select_existing = [
        feature
        for feature in features_to_select
        if feature in feature_names
    ]

    return subset_features(
        X=X,
        feature_names=feature_names,
        features_to_keep=features_to_select_existing
    )


def get_code_features_to_remove(feature_names):
    """
    Entfernt Diagnose- und Prozedurcode-Multi-Hot-Features.
    Funktioniert für ursprüngliche Feature-Namen und No-Leakage-Feature-Namen.
    """

    return [
        feature
        for feature in feature_names
        if (
            feature.startswith("diag_code")
            or feature.startswith("proc_code")
            or feature.startswith("last_diag_code")
            or feature.startswith("last_proc_code")
        )
    ]

# =========================================================
# Prozedurmetriken für ursprünglichen Ansatz vorbereiten
# =========================================================
(
    X_pcs_body_system,
    pcs_body_system_feature_names,
    X_proc_metrics_without_body_system,
    proc_metric_feature_names
) = prepare_proc_feature_blocks(
    proc_metrics_input=proc_metrics,
    hadm_order=hadm_id_full,
    label="original"
)


# =========================================================
# CCSR-Benchmark-Modell vorbereiten
# =========================================================

(
    X_baseline_with_ccsr,
    baseline_with_ccsr_feature_names,
    ccsr_base_feature_names,
    ccsr_category_feature_names,
    ccsr_aligned
) = prepare_ccsr_multihot_feature_set(
    ccsr_df=ccsr_multihot,
    basic_df=baseline_no_icd10,
    reference_hadm_id=hadm_id,
    expected_y=y,
    expected_subject_id=subject_id
) 
    

assert (
    ccsr_base_feature_names
    == baseline_no_icd10_feature_names
), (
    "CCSR-Modell und baseline_no_icd10 enthalten nicht "
    "dieselben Basisvariablen."
)

print(
    "CCSR und baseline_no_icd10 verwenden identische Basisfeatures:",
    ccsr_base_feature_names == baseline_no_icd10_feature_names
)   


    
# =========================================================
# Hauptansatz - Semantik-Modelle bauen
# =========================================================

# -------------------------------------------
# Feature-Blöcke aus dem vollständigen Modell extrahieren
# -------------------------------------------

statistical_code_features_original = [
    feature
    for feature in full_feature_names
    if (
        feature in [
            "n_diag_codes",
            "n_proc_codes",
            "n_diag_chapters"
        ]
        or feature.startswith("ic_")
    )
]

chapter_multihot_features_original = [
    feature
    for feature in full_feature_names
    if feature.startswith("icd10_chapter")
]

X_statistical_code_features_original, statistical_code_feature_names_original = (
    select_features_from_sparse(
        X=X_full,
        feature_names=full_feature_names,
        features_to_select=statistical_code_features_original
    )
)

X_chapter_multihot_original, chapter_multihot_feature_names_original = (
    select_features_from_sparse(
        X=X_full,
        feature_names=full_feature_names,
        features_to_select=chapter_multihot_features_original
    )
)

X_diagnosis_metrics_original, diagnosis_metric_feature_names_original = (
    select_features_from_sparse(
        X=X_full,
        feature_names=full_feature_names,
        features_to_select=diagnosis_structure_metrics_original
    )
)

print("\n=== Feature-Blöcke des Hauptansatzes ===")
print("Statistische Code-Merkmale:", len(statistical_code_feature_names_original))
print("Diagnosekapitel-Multi-Hot:", len(chapter_multihot_feature_names_original))
print("PCS-Body-System-Multi-Hot:", len(pcs_body_system_feature_names))
print("Diagnosemetriken:", len(diagnosis_metric_feature_names_original))
print("Behandlungsmetriken:", len(proc_metric_feature_names))

# -------------------------------------------
# Semantik-Modell 1:
# Baseline mit ICD-10 + statistische Code-Merkmale
# -------------------------------------------

X_semantic_model_1 = sparse.hstack(
    [
        X_baseline,
        X_statistical_code_features_original
    ],
    format="csr"
)

semantic_model_1_feature_names = (
    baseline_feature_names
    + statistical_code_feature_names_original
)

# -------------------------------------------
# Semantik-Modell 2:
# Baseline mit ICD-10
# + Multi-Hot-Features der Diagnosekapitel und Organsysteme
# -------------------------------------------

X_semantic_model_2 = sparse.hstack(
    [
        X_baseline,
        X_chapter_multihot_original,
        X_pcs_body_system
    ],
    format="csr"
)

semantic_model_2_feature_names = (
    baseline_feature_names
    + chapter_multihot_feature_names_original
    + pcs_body_system_feature_names
)

# -------------------------------------------
# Semantik-Modell 3:
# Baseline mit ICD-10 + Diagnosemetriken
# -------------------------------------------

X_semantic_model_3 = sparse.hstack(
    [
        X_baseline,
        X_diagnosis_metrics_original
    ],
    format="csr"
)

semantic_model_3_feature_names = (
    baseline_feature_names
    + diagnosis_metric_feature_names_original
)

# -------------------------------------------
# Semantik-Modell 4:
# Baseline mit ICD-10 + Behandlungsmetriken
# -------------------------------------------

X_semantic_model_4 = sparse.hstack(
    [
        X_baseline,
        X_proc_metrics_without_body_system
    ],
    format="csr"
)

semantic_model_4_feature_names = (
    baseline_feature_names
    + proc_metric_feature_names
)

# -------------------------------------------
# Semantik-Modell 5:
# Baseline mit ICD-10 + alle vier Feature-Kategorien
# -------------------------------------------

X_semantic_model_5 = sparse.hstack(
    [
        X_full,
        X_pcs_body_system,
        X_proc_metrics_without_body_system
    ],
    format="csr"
)

semantic_model_5_feature_names = (
    full_feature_names
    + pcs_body_system_feature_names
    + proc_metric_feature_names
)

# Semantik-Modell 5 exportieren
semantic_5_path = os.path.join(path, "semantic_model_5")
os.makedirs(semantic_5_path, exist_ok=True)

sparse.save_npz(os.path.join(semantic_5_path, "X_sparse.npz"), X_semantic_model_5)

with open(
    os.path.join(semantic_5_path, "feature_names.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        semantic_model_5_feature_names,
        f,
        ensure_ascii=False,
        indent=2
    )

np.save(os.path.join(semantic_5_path, "hadm_id.npy"), hadm_id_full)
np.save(os.path.join(semantic_5_path, "subject_id.npy"), subject_id_full)
np.save(os.path.join(semantic_5_path, "y.npy"), y_full)

print("Semantik-Modell 5 gespeichert:", X_semantic_model_5.shape)

# -------------------------------------------
# Semantik-Modell 6:
# Semantik-Modell 5 ohne direkte ICD-10-Codes
# -------------------------------------------

codes_to_remove_original = get_code_features_to_remove(
    semantic_model_5_feature_names
)

X_semantic_model_6, semantic_model_6_feature_names = remove_features_from_sparse(
    X=X_semantic_model_5,
    feature_names=semantic_model_5_feature_names,
    features_to_remove=codes_to_remove_original
)

# -------------------------------------------
# Checks
# -------------------------------------------

print("\n=== Semantik-Modelle der Hauptanalyse ===")

for name, X_current, feature_names_current in [
    ("semantic_model_1", X_semantic_model_1, semantic_model_1_feature_names),
    ("semantic_model_2", X_semantic_model_2, semantic_model_2_feature_names),
    ("semantic_model_3", X_semantic_model_3, semantic_model_3_feature_names),
    ("semantic_model_4", X_semantic_model_4, semantic_model_4_feature_names),
    ("semantic_model_5", X_semantic_model_5, semantic_model_5_feature_names),
    ("semantic_model_6", X_semantic_model_6, semantic_model_6_feature_names),
]:
    print(name)
    print("X shape:", X_current.shape)
    print("Feature-Namen:", len(feature_names_current))
    print("Feature-Anzahl passt:", X_current.shape[1] == len(feature_names_current))

    assert X_current.shape[1] == len(feature_names_current)



# =========================================================
# No-Leakage-Semantik-Modelle bauen
# =========================================================

# -------------------------------------------
# Feature-Blöcke aus full_model_no_leakage extrahieren
# -------------------------------------------

statistical_code_features_no_leakage = [
    feature
    for feature in full_no_leakage_feature_names
    if (
        feature in [
            "last_n_diag_codes",
            "last_n_proc_codes",
            "last_n_diag_chapters"
        ]
        or feature.startswith("last_ic_")
    )
]

chapter_body_system_multihot_features_no_leakage = [
    feature
    for feature in full_no_leakage_feature_names
    if (
        feature.startswith("last_icd10_chapter")
        or feature.startswith("last_pcs_body_system")
    )
]

# Diagnosemetriken
X_diagnosis_metrics_no_leakage, diagnosis_metric_no_leakage_feature_names = (
    select_features_from_sparse(
        X=X_full_no_leakage,
        feature_names=full_no_leakage_feature_names,
        features_to_select=diagnosis_structure_metrics_no_leakage
    )
)

# Prozedurmetriken
X_proc_metrics_no_leakage, proc_metric_no_leakage_feature_names = (
    select_features_from_sparse(
        X=X_full_no_leakage,
        feature_names=full_no_leakage_feature_names,
        features_to_select=procedure_metric_features_no_leakage
    )
)

# Statistische Code-Merkmale
X_statistical_code_features_no_leakage, statistical_code_feature_names_no_leakage = (
    select_features_from_sparse(
        X=X_full_no_leakage,
        feature_names=full_no_leakage_feature_names,
        features_to_select=statistical_code_features_no_leakage
    )
)

# Diagnosekapitel- und Body-System-Multi-Hot
X_chapter_body_system_multihot_no_leakage, chapter_body_system_multihot_feature_names_no_leakage = (
    select_features_from_sparse(
        X=X_full_no_leakage,
        feature_names=full_no_leakage_feature_names,
        features_to_select=chapter_body_system_multihot_features_no_leakage
    )
)

print("\n=== Feature-Blöcke des No-Leakage-Ansatzes ===")
print("Statistische Code-Merkmale:", len(statistical_code_feature_names_no_leakage))
print(
    "Diagnosekapitel-/Body-System-Multi-Hot:",
    len(chapter_body_system_multihot_feature_names_no_leakage)
)
print("Diagnosemetriken:", len(diagnosis_metric_no_leakage_feature_names))
print("Behandlungsmetriken:", len(proc_metric_no_leakage_feature_names))

# -------------------------------------------
# Semantik-Modell 1 ohne Leakage:
# Baseline mit ICD-10 + statistische Code-Merkmale
# des letzten Aufenthalts
# -------------------------------------------

X_semantic_model_1_no_leakage = sparse.hstack(
    [
        X_baseline_no_leakage,
        X_statistical_code_features_no_leakage
    ],
    format="csr"
)

semantic_model_1_no_leakage_feature_names = (
    baseline_no_leakage_feature_names
    + statistical_code_feature_names_no_leakage
)

# -------------------------------------------
# Semantik-Modell 2 ohne Leakage:
# Baseline mit ICD-10 + Multi-Hot-Features der Diagnosekapitel
# und Organsysteme des letzten Aufenthalts
# -------------------------------------------

X_semantic_model_2_no_leakage = sparse.hstack(
    [
        X_baseline_no_leakage,
        X_chapter_body_system_multihot_no_leakage
    ],
    format="csr"
)

semantic_model_2_no_leakage_feature_names = (
    baseline_no_leakage_feature_names
    + chapter_body_system_multihot_feature_names_no_leakage
)

# -------------------------------------------
# Semantik-Modell 3 ohne Leakage:
# Baseline mit ICD-10 + Diagnosemetriken
# -------------------------------------------

X_semantic_model_3_no_leakage = sparse.hstack(
    [
        X_baseline_no_leakage,
        X_diagnosis_metrics_no_leakage
    ],
    format="csr"
)

semantic_model_3_no_leakage_feature_names = (
    baseline_no_leakage_feature_names
    + diagnosis_metric_no_leakage_feature_names
)

# -------------------------------------------
# Semantik-Modell 4 ohne Leakage:
# Baseline mit ICD-10 + Behandlungsmetriken
# -------------------------------------------

X_semantic_model_4_no_leakage = sparse.hstack(
    [
        X_baseline_no_leakage,
        X_proc_metrics_no_leakage
    ],
    format="csr"
)

semantic_model_4_no_leakage_feature_names = (
    baseline_no_leakage_feature_names
    + proc_metric_no_leakage_feature_names
)

# -------------------------------------------
# Semantik-Modell 5 ohne Leakage:
# vollständiges Modell mit allen vier Merkmalskategorien
# -------------------------------------------

X_semantic_model_5_no_leakage = X_full_no_leakage
semantic_model_5_no_leakage_feature_names = (
    full_no_leakage_feature_names.copy()
)

# -------------------------------------------
# Semantik-Modell 6 ohne Leakage:
# Modell 5 ohne direkte ICD-10-Codes
# -------------------------------------------

codes_to_remove_no_leakage = get_code_features_to_remove(
    semantic_model_5_no_leakage_feature_names
)

X_semantic_model_6_no_leakage, semantic_model_6_no_leakage_feature_names = remove_features_from_sparse(
    X=X_semantic_model_5_no_leakage,
    feature_names=semantic_model_5_no_leakage_feature_names,
    features_to_remove=codes_to_remove_no_leakage
)

# -------------------------------------------
# Checks
# -------------------------------------------

print("\n=== No-Leakage-Semantik-Modelle ===")

for name, X_current, feature_names_current in [
    ("semantic_model_1_no_leakage", X_semantic_model_1_no_leakage, semantic_model_1_no_leakage_feature_names),
    ("semantic_model_2_no_leakage", X_semantic_model_2_no_leakage, semantic_model_2_no_leakage_feature_names),
    ("semantic_model_3_no_leakage", X_semantic_model_3_no_leakage, semantic_model_3_no_leakage_feature_names),
    ("semantic_model_4_no_leakage", X_semantic_model_4_no_leakage, semantic_model_4_no_leakage_feature_names),
    ("semantic_model_5_no_leakage", X_semantic_model_5_no_leakage, semantic_model_5_no_leakage_feature_names),
    ("semantic_model_6_no_leakage", X_semantic_model_6_no_leakage, semantic_model_6_no_leakage_feature_names),
]:
    print(name)
    print("X shape:", X_current.shape)
    print("Feature-Namen:", len(feature_names_current))
    print("Feature-Anzahl passt:", X_current.shape[1] == len(feature_names_current))

    assert X_current.shape[1] == len(feature_names_current)







# =========================================================
# Modellierung - XGBoost
# =========================================================

# -------------------------------------------
# Gespeicherten patientenbasierten Train/Test-Split laden
# -------------------------------------------

groups = subject_id

train_idx = np.load(os.path.join(path, "train_test_split/train_idx.npy"))
test_idx = np.load(os.path.join(path, "train_test_split/test_idx.npy"))

# Gespeicherte hadm_id des ursprünglichen Splits laden
train_hadm_id_saved = np.load(os.path.join(path, "train_test_split/train_hadm_id.npy"))
test_hadm_id_saved = np.load(os.path.join(path, "train_test_split/test_hadm_id.npy"))

# Prüfen, ob die Positionsindizes weiterhin genau dieselben
# Aufenthalte auswählen wie beim Erstellen des Splits
assert np.array_equal(
    hadm_id[train_idx],
    train_hadm_id_saved
), ("Train-Split stimmt nicht mehr mit dem gespeicherten Train-Split überein.")

assert np.array_equal(
    hadm_id[test_idx],
    test_hadm_id_saved
), ("Test-Split stimmt nicht mehr mit dem gespeicherten Test-Split überein.")

print("Gespeicherte Train-/Test-hadm_id stimmen exakt überein.")

# ---------------------------------------------------------
# Sicherheitschecks
# ---------------------------------------------------------

assert train_idx.ndim == 1
assert test_idx.ndim == 1

assert train_idx.max() < len(y)
assert test_idx.max() < len(y)

assert len(
    np.intersect1d(train_idx, test_idx)
) == 0, (
    "Train- und Test-Indizes überschneiden sich."
)

assert len(train_idx) + len(test_idx) == len(y), (
    "Train- und Test-Indizes decken nicht alle Aufenthalte ab."
)

assert len(
    set(subject_id[train_idx]).intersection(
        set(subject_id[test_idx])
    )
) == 0, (
    "Patienten kommen sowohl im Train- als auch im Testset vor."
)

print("Gespeicherter Train/Test-Split geladen.")
print("Train-Größe:", len(train_idx))
print("Test-Größe:", len(test_idx))
print("Train-Mortalitätsrate:", y[train_idx].mean())
print("Test-Mortalitätsrate:", y[test_idx].mean())
print(
    "Subject-Overlap:",
    len(
        set(subject_id[train_idx]).intersection(
            set(subject_id[test_idx])
        )
    )
)


def get_subset(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    else:
        return X[indices]
    
  
# -------------------------------------------
# Threshold-Optimierung
# -------------------------------------------

def find_best_threshold(y_true, y_proba, metric="youden"):
    thresholds = np.arange(0.01, 1.00, 0.01)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        youden = sensitivity + specificity - 1

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": balanced_acc,
            "youden": youden
        })

    threshold_df = pd.DataFrame(rows)

    if metric == "f1":
        best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    elif metric == "youden":
        best_row = threshold_df.loc[threshold_df["youden"].idxmax()]
    elif metric == "balanced_accuracy":
        best_row = threshold_df.loc[threshold_df["balanced_accuracy"].idxmax()]
    else:
        raise ValueError("metric muss 'f1', 'youden' oder 'balanced_accuracy' sein.")

    return best_row, threshold_df


# -------------------------------------------
# Berechnung der Evaluationsmetriken
# -------------------------------------------

def calculate_pr_auc(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_proba
    )

    return auc(recall, precision)

def evaluate_predictions(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": calculate_pr_auc(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
# -------------------------------------------
# Tuning: Führt RandomizedSearchCV für ein XGBoost-Modell mit StratifiedGroupKFold durch
# -------------------------------------------

def tune_evaluate_xgboost(
    X,
    y,
    groups,
    train_idx,
    test_idx,
    feature_set_name,
    n_iter=50,
    cv_splits=5,
    scoring="average_precision",
    threshold_metric="youden",
    random_state=42
):
    print("\n==============================")
    print(f"Feature-Set: {feature_set_name}")
    print("==============================")

    X_train = get_subset(X, train_idx)
    X_test = get_subset(X, test_idx)

    y_train = y[train_idx]
    y_test = y[test_idx]

    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    print("Train-Fälle:", len(y_train))
    print("Test-Fälle:", len(y_test))
    print("Train-Mortalitätsrate:", y_train.mean())
    print("Test-Mortalitätsrate:", y_test.mean())

    overlap = set(groups_train).intersection(set(groups_test))
    print("Subject-Overlap zwischen Train und Test:", len(overlap))

    if len(overlap) > 0:
        raise ValueError("Data Leakage: Einige subject_id kommen sowohl in Train als auch Test vor.")

    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)

    scale_pos_weight = n_negative / n_positive

    print("scale_pos_weight:", scale_pos_weight)

    inner_cv = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight
    )

    param_distributions = {
        "n_estimators": randint(100, 600),
        "max_depth": randint(2, 7),
        "learning_rate": loguniform(0.005, 0.2),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 16),
        "gamma": uniform(0, 5),
        "reg_alpha": loguniform(1e-3, 10),
        "reg_lambda": loguniform(1e-2, 100)
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        refit=True
    )

    # Parameter-Tuning nur auf Trainingsdaten
    search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    print("\nBeste Parameter:")
    print(search.best_params_)

    print(f"\nBester CV-Score ({scoring}): {search.best_score_:.4f}")

    best_model = search.best_estimator_

    # Out-of-Fold-Wahrscheinlichkeiten auf Trainingsdaten
    # Diese dienen für Threshold-Tuning, nicht das Test-Set.
    oof_train_proba = cross_val_predict(
        best_model,
        X_train,
        y_train,
        groups=groups_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    best_threshold_row, threshold_results = find_best_threshold(
        y_true=y_train,
        y_proba=oof_train_proba,
        metric=threshold_metric
    )

    best_threshold = float(best_threshold_row["threshold"])

    print("\nBester Threshold auf Trainings-/OOF-Predictions:")
    print(best_threshold_row)

    # Finales Modell auf vollständigem Trainingsset trainieren
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **search.best_params_
    )

    final_model.fit(X_train, y_train)

    # Finale Test-Predictions
    test_proba = final_model.predict_proba(X_test)[:, 1]

    test_metrics_default_threshold = evaluate_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=0.5
    )

    test_metrics_optimized_threshold = evaluate_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold
    )

    test_metrics_default_threshold["feature_set"] = feature_set_name
    test_metrics_default_threshold["threshold_type"] = "default_0_5"

    test_metrics_optimized_threshold["feature_set"] = feature_set_name
    test_metrics_optimized_threshold["threshold_type"] = f"optimized_{threshold_metric}"

    print("\nTest-Metriken mit Threshold 0.5:")
    print(pd.Series(test_metrics_default_threshold))

    print("\nTest-Metriken mit optimiertem Threshold:")
    print(pd.Series(test_metrics_optimized_threshold))

    return {
        "feature_set": feature_set_name,
        "search": search,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "best_threshold": best_threshold,
        "threshold_results": threshold_results,
        "oof_train_proba": oof_train_proba,
        "final_model": final_model,
        "test_proba": test_proba,
        "test_metrics_default_threshold": test_metrics_default_threshold,
        "test_metrics_optimized_threshold": test_metrics_optimized_threshold,
        "train_idx": train_idx,
        "test_idx": test_idx
    }


# -------------------------------------------
# Bootstrapping (für ein einzelnes Modell)
# -------------------------------------------

def bootstrap_test_metrics_by_group(
    y_test,
    test_proba,
    groups_test,
    threshold,
    n_bootstraps=500,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups_test)

    bootstrap_rows = []

    for i in range(n_bootstraps):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True
        )

        sampled_indices = []

        for group in sampled_groups:
            group_indices = np.where(groups_test == group)[0]
            sampled_indices.extend(group_indices)

        sampled_indices = np.array(sampled_indices)

        y_sample = y_test[sampled_indices]
        proba_sample = test_proba[sampled_indices]

        if len(np.unique(y_sample)) < 2:
            continue

        metrics = evaluate_predictions(
            y_true=y_sample,
            y_proba=proba_sample,
            threshold=threshold
        )

        bootstrap_rows.append(metrics)

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    if bootstrap_df.empty:
        raise ValueError(
            "Alle Bootstrap-Samples enthielten nur eine Klasse. "
            "CIs konnten nicht berechnet werden."
        )

    ci_rows = []

    for metric in [
        "roc_auc",
        "pr_auc",
        "average_precision",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy"
    ]:        
        ci_rows.append({
            "metric": metric,
            "mean": bootstrap_df[metric].mean(),
            "ci_lower": bootstrap_df[metric].quantile(0.025),
            "ci_upper": bootstrap_df[metric].quantile(0.975)
        })

    ci_df = pd.DataFrame(ci_rows)

    return bootstrap_df, ci_df


# -------------------------------------------
# Patientenbasiertes paarweises Bootstrapping 
# zum Vergleich zweier Modelle.
# -------------------------------------------

groups_test = subject_id[test_idx]
y_test = y[test_idx]

def paired_bootstrap_metric_differences_by_group(
    y_test,
    proba_model_a,
    proba_model_b,
    groups_test,
    threshold_a,
    threshold_b,
    model_a_name,
    model_b_name,
    n_bootstraps=1000,
    random_state=42
):
    """
    In jeder Bootstrap-Replikation werden dieselben Patienten für beide Modelle gezogen.
    Danach wird berechnet:
        Differenz = Metrik Modell B - Metrik Modell A

    Positive Werte bedeuten also: Modell B ist besser als Modell A.
    """

    rng = np.random.default_rng(random_state)

    y_test = np.asarray(y_test)
    proba_model_a = np.asarray(proba_model_a)
    proba_model_b = np.asarray(proba_model_b)
    groups_test = np.asarray(groups_test)

    unique_groups = np.unique(groups_test)

    bootstrap_rows = []

    metrics_to_compare = [
        "roc_auc",
        "pr_auc",
        "average_precision",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy"
    ]

    for i in range(n_bootstraps):

        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True
        )

        sampled_indices = []

        for group in sampled_groups:
            group_indices = np.where(groups_test == group)[0]
            sampled_indices.extend(group_indices)

        sampled_indices = np.array(sampled_indices)

        y_sample = y_test[sampled_indices]
        proba_a_sample = proba_model_a[sampled_indices]
        proba_b_sample = proba_model_b[sampled_indices]

        # AUROC und PR-AUC sind nicht definiert, wenn nur eine Klasse vorkommt
        if len(np.unique(y_sample)) < 2:
            continue

        metrics_a = evaluate_predictions(
            y_true=y_sample,
            y_proba=proba_a_sample,
            threshold=threshold_a
        )

        metrics_b = evaluate_predictions(
            y_true=y_sample,
            y_proba=proba_b_sample,
            threshold=threshold_b
        )

        row = {
            "bootstrap_iteration": i,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "comparison": f"{model_b_name} - {model_a_name}"
        }

        for metric in metrics_to_compare:
            row[f"{metric}_model_a"] = metrics_a[metric]
            row[f"{metric}_model_b"] = metrics_b[metric]
            row[f"{metric}_diff"] = metrics_b[metric] - metrics_a[metric]

        bootstrap_rows.append(row)

    bootstrap_diff_df = pd.DataFrame(bootstrap_rows)

    if bootstrap_diff_df.empty:
        raise ValueError(
            "Alle Bootstrap-Samples enthielten nur eine Klasse. "
            "CIs der Differenzen konnten nicht berechnet werden."
        )

    ci_rows = []

    for metric in metrics_to_compare:
        diff_col = f"{metric}_diff"

        ci_rows.append({
            "model_a": model_a_name,
            "model_b": model_b_name,
            "comparison": f"{model_b_name} - {model_a_name}",
            "metric": metric,
            "mean_difference": bootstrap_diff_df[diff_col].mean(),
            "ci_lower": bootstrap_diff_df[diff_col].quantile(0.025),
            "ci_upper": bootstrap_diff_df[diff_col].quantile(0.975),
            "n_successful_bootstraps": len(bootstrap_diff_df)
        })

    ci_diff_df = pd.DataFrame(ci_rows)

    return bootstrap_diff_df, ci_diff_df

# -------------------------------------------
# Vorbereitung
# -------------------------------------------

# Feature-Sets definieren
feature_sets = {
    # Feature-Sets des Hauptansatzes
    "baseline_basic": X_baseline_basic,
    "baseline_no_icd10": X_baseline_no_icd10,
    "baseline_with_icd10": X_baseline,
    "semantic_model_1": X_semantic_model_1,
    "semantic_model_2": X_semantic_model_2,
    "semantic_model_3": X_semantic_model_3,
    "semantic_model_4": X_semantic_model_4,
    "semantic_model_5": X_semantic_model_5,
    "semantic_model_6": X_semantic_model_6,

    # Feature-Sets ohne Temporal Leakage
    "baseline_with_icd10_no_leakage": X_baseline_no_leakage,
    "semantic_model_1_no_leakage": X_semantic_model_1_no_leakage,
    "semantic_model_2_no_leakage": X_semantic_model_2_no_leakage,
    "semantic_model_3_no_leakage": X_semantic_model_3_no_leakage,
    "semantic_model_4_no_leakage": X_semantic_model_4_no_leakage,
    "semantic_model_5_no_leakage": X_semantic_model_5_no_leakage,
    "semantic_model_6_no_leakage": X_semantic_model_6_no_leakage,
    
    # CCSR-Benchmark
    "baseline_with_ccsr": X_baseline_with_ccsr
}

# Zentrale Konfigurationen
XGB_TUNING_CONFIG = {
    "n_iter": 50,
    "cv_splits": 5,
    "scoring": "average_precision",
    "threshold_metric": "youden",
    "random_state": 42
}

SINGLE_BOOTSTRAP_CONFIG = {
    "n_bootstraps": 500,
    "random_state": 42
}

PAIRED_BOOTSTRAP_CONFIG = {
    "n_bootstraps": 1000,
    "random_state": 42
}

# -------------------------------------------
# Einheitliche Funktion für die Durchführung
# -------------------------------------------

def run_single_xgb_feature_set(feature_set_name):
    """
    Führt Tuning, Test-Evaluation und Bootstrapping für genau ein Feature-Set aus.
    Voraussetzung: feature_sets, y, subject_id, train_idx, test_idx, y_test und groups_test existieren.
    """

    if feature_set_name not in feature_sets:
        raise ValueError(
            f"Unbekanntes Feature-Set: {feature_set_name}. "
            f"Verfügbare Feature-Sets: {list(feature_sets.keys())}"
        )

    X_current = feature_sets[feature_set_name]

    print("\n##################################################")
    print(f"Starte Modell für Feature-Set: {feature_set_name}")
    print("##################################################")

    result = tune_evaluate_xgboost(
        X=X_current,
        y=y,
        groups=subject_id,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_set_name=feature_set_name,
        **XGB_TUNING_CONFIG
    )

    metrics_df = pd.DataFrame([
        result["test_metrics_default_threshold"],
        result["test_metrics_optimized_threshold"]
    ])
    
    # Einzelmodell-Bootstrapping
    bootstrap_df, ci_df = bootstrap_test_metrics_by_group(
        y_test=y_test,
        test_proba=result["test_proba"],
        groups_test=groups_test,
        threshold=result["best_threshold"],
        **SINGLE_BOOTSTRAP_CONFIG
    )
    
    print("\nBeste Parameter:")
    print(result["best_params"])

    print("\nTest-Metriken:")
    print(metrics_df)
    
    print("\nBootstrap-Konfidenzintervalle:")
    print(ci_df)
    
    # Ergebnisse speichern
    pkl_path = os.path.join(
        model_output_dir,
        f"xgb_result_{feature_set_name}.pkl"
    )
    
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)

    return result, metrics_df, bootstrap_df, ci_df


# -------------------------------------------
# Modelle aussführen
# -------------------------------------------

xgb_results = {}
xgb_metrics_dfs = {}
xgb_bootstrap_dfs = {}
xgb_ci_dfs = {}

def run_and_store_xgb_feature_set(feature_set_name):
    """
    Führt genau ein Feature-Set aus und speichert die Ergebnisse
    zentral in Dictionaries.

    Dadurch kann jedes Modell einzeln gestartet werden, ohne die anderen
    Modelle erneut zu trainieren.
    """

    result, metrics_df, bootstrap_df, ci_df = run_single_xgb_feature_set(
        feature_set_name
    )

    xgb_results[feature_set_name] = result
    xgb_metrics_dfs[feature_set_name] = metrics_df
    xgb_bootstrap_dfs[feature_set_name] = bootstrap_df
    xgb_ci_dfs[feature_set_name] = ci_df

    return result, metrics_df, bootstrap_df, ci_df


# Modelle des Hauptansatzes
result_baseline_basic, results_baseline_basic_df, bootstrap_baseline_basic, ci_baseline_basic = run_and_store_xgb_feature_set(
    "baseline_basic"
)

result_baseline_no_icd10, results_baseline_no_icd10_df, bootstrap_baseline_no_icd10, ci_baseline_no_icd10 = run_and_store_xgb_feature_set(
    "baseline_no_icd10"
)

result_baseline, results_baseline_df, bootstrap_baseline, ci_baseline = run_and_store_xgb_feature_set(
    "baseline_with_icd10"
)

result_semantic_model_1, results_semantic_model_1_df, bootstrap_semantic_model_1, ci_semantic_model_1 = run_and_store_xgb_feature_set(
    "semantic_model_1"
)

result_semantic_model_2, results_semantic_model_2_df, bootstrap_semantic_model_2, ci_semantic_model_2 = run_and_store_xgb_feature_set(
    "semantic_model_2"
)

result_semantic_model_3, results_semantic_model_3_df, bootstrap_semantic_model_3, ci_semantic_model_3 = run_and_store_xgb_feature_set(
    "semantic_model_3"
)

result_semantic_model_4, results_semantic_model_4_df, bootstrap_semantic_model_4, ci_semantic_model_4 = run_and_store_xgb_feature_set(
    "semantic_model_4"
)

result_semantic_model_5, results_semantic_model_5_df, bootstrap_semantic_model_5, ci_semantic_model_5 = run_and_store_xgb_feature_set(
    "semantic_model_5"
)

result_semantic_model_6, results_semantic_model_6_df, bootstrap_semantic_model_6, ci_semantic_model_6 = run_and_store_xgb_feature_set(
    "semantic_model_6"
)


# Modelle ohne Data Leakage
result_baseline_with_icd10_no_leakage, results_baseline_with_icd10_no_leakage_df, bootstrap_baseline_with_icd10_no_leakage, ci_baseline_with_icd10_no_leakage = run_and_store_xgb_feature_set(
    "baseline_with_icd10_no_leakage"
)

result_semantic_model_1_no_leakage, results_semantic_model_1_no_leakage_df, bootstrap_semantic_model_1_no_leakage, ci_semantic_model_1_no_leakage = run_and_store_xgb_feature_set(
    "semantic_model_1_no_leakage"
)

result_semantic_model_2_no_leakage, results_semantic_model_2_no_leakage_df, bootstrap_semantic_model_2_no_leakage, ci_semantic_model_2_no_leakage = run_and_store_xgb_feature_set(
    "semantic_model_2_no_leakage"
)

result_semantic_model_3_no_leakage, results_semantic_model_3_no_leakage_df, bootstrap_semantic_model_3_no_leakage, ci_semantic_model_3_no_leakage = run_and_store_xgb_feature_set(
    "semantic_model_3_no_leakage"
)

result_semantic_model_4_no_leakage, results_semantic_model_4_no_leakage_df, bootstrap_semantic_model_4_no_leakage, ci_semantic_model_4_no_leakage = run_and_store_xgb_feature_set(
    "semantic_model_4_no_leakage"
)

result_semantic_model_5_no_leakage, results_semantic_model_5_no_leakage_df, bootstrap_semantic_model_5_no_leakage, ci_semantic_model_5_no_leakage = run_and_store_xgb_feature_set(
    "semantic_model_5_no_leakage"
)

result_semantic_model_6_no_leakage, results_semantic_model_6_no_leakage_df, bootstrap_semantic_model_6_no_leakage, ci_semantic_model_6_no_leakage = run_and_store_xgb_feature_set(
    "semantic_model_6_no_leakage"
)


# CCSR-Benchmark
result_baseline_with_ccsr, results_baseline_with_ccsr_df, bootstrap_baseline_with_ccsr, ci_baseline_with_ccsr = run_and_store_xgb_feature_set(
    "baseline_with_ccsr"
)


# -------------------------------------------
# Ergebnisse der ausgeführten Modelle sammeln
# -------------------------------------------

if xgb_metrics_dfs:
    all_xgb_test_results = pd.concat(
        xgb_metrics_dfs.values(),
        ignore_index=True
    )
else:
    all_xgb_test_results = pd.DataFrame()

all_xgb_test_results.to_csv(
    os.path.join(output_dir, "all_xgb_test_results.csv"),
    index=False
)

all_xgb_test_results

# -------------------------------------------
# Ergebnisse exportieren
# -------------------------------------------

if xgb_ci_dfs:
    all_xgb_ci_results = pd.concat(
        [
            ci_df.assign(feature_set=feature_set_name)
            for feature_set_name, ci_df in xgb_ci_dfs.items()
        ],
        ignore_index=True
    )
else:
    all_xgb_ci_results = pd.DataFrame()

all_xgb_ci_results.to_csv(
    os.path.join(output_dir, "all_xgb_ci_results.csv"),
    index=False
)

model_overview_rows = []

for feature_set_name, result in xgb_results.items():
    row = {
        "feature_set": feature_set_name,
        "best_cv_score": result["best_cv_score"],
        "best_threshold": result["best_threshold"]
    }

    for param_name, param_value in result["best_params"].items():
        row[f"param_{param_name}"] = param_value

    model_overview_rows.append(row)

xgb_model_overview = pd.DataFrame(model_overview_rows)

xgb_model_overview.to_csv(
    os.path.join(output_dir, "xgb_model_overview.csv"),
    index=False
)

# -------------------------------------------
# Paarweise Bootstrap-Vergleiche definieren
# -------------------------------------------

pairwise_comparisons = [
    # Mit Data Leakage
    ("baseline_with_icd10", "semantic_model_1"),
    ("baseline_with_icd10", "semantic_model_2"),
    ("baseline_with_icd10", "semantic_model_3"),
    ("baseline_with_icd10", "semantic_model_4"),
    ("baseline_with_icd10", "semantic_model_5"),
    ("baseline_with_icd10", "semantic_model_6"),
    ("baseline_no_icd10", "semantic_model_6"),
    ("semantic_model_5", "semantic_model_6"),

    # Ohne Temporal Leakage
    ("baseline_with_icd10_no_leakage", "semantic_model_1_no_leakage"),
    ("baseline_with_icd10_no_leakage", "semantic_model_2_no_leakage"),
    ("baseline_with_icd10_no_leakage", "semantic_model_3_no_leakage"),
    ("baseline_with_icd10_no_leakage", "semantic_model_4_no_leakage"),
    ("baseline_with_icd10_no_leakage", "semantic_model_5_no_leakage"),
    ("baseline_with_icd10_no_leakage", "semantic_model_6_no_leakage"),
    ("baseline_no_icd10", "semantic_model_6_no_leakage"),
    ("semantic_model_5_no_leakage", "semantic_model_6_no_leakage"),
    
    # Mit vs. ohne Leakage
    ("baseline_with_icd10_no_leakage", "baseline_with_icd10"),
    ("semantic_model_5_no_leakage", "semantic_model_5"),
    ("semantic_model_6_no_leakage", "semantic_model_6"),
    
    # CCSR-Benchmark
    ("baseline_no_icd10", "baseline_with_ccsr"),
    ("baseline_with_icd10", "baseline_with_ccsr"),
    ("baseline_with_ccsr", "semantic_model_5"),
    ("baseline_with_ccsr", "semantic_model_6")
]

# -------------------------------------------
# Paarweise Bootstrap-Vergleiche ausführen
# -------------------------------------------

xgb_pairwise_bootstrap_dfs = {}
xgb_pairwise_ci_dfs = {}


def run_paired_bootstrap_comparison(model_a_name, model_b_name):
    """
    Führt einen paarweisen patient-level Bootstrap-Vergleich zweier bereits
    trainierter Modelle durch.

    Voraussetzung:
    - Beide Modelle wurden bereits mit run_and_store_xgb_feature_set ausgeführt.
    - Die Ergebnisse liegen in xgb_results.
    """

    if model_a_name not in xgb_results:
        raise ValueError(
            f"Modell A wurde noch nicht ausgeführt: {model_a_name}"
        )

    if model_b_name not in xgb_results:
        raise ValueError(
            f"Modell B wurde noch nicht ausgeführt: {model_b_name}"
        )

    print("\n##################################################")
    print(f"Paarweises Bootstrapping: {model_b_name} - {model_a_name}")
    print("##################################################")

    result_a = xgb_results[model_a_name]
    result_b = xgb_results[model_b_name]

    bootstrap_diff_df, ci_diff_df = paired_bootstrap_metric_differences_by_group(
        y_test=y_test,
        proba_model_a=result_a["test_proba"],
        proba_model_b=result_b["test_proba"],
        groups_test=groups_test,
        threshold_a=result_a["best_threshold"],
        threshold_b=result_b["best_threshold"],
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        **PAIRED_BOOTSTRAP_CONFIG
    )

    comparison_name = f"{model_b_name}_minus_{model_a_name}"

    xgb_pairwise_bootstrap_dfs[comparison_name] = bootstrap_diff_df
    xgb_pairwise_ci_dfs[comparison_name] = ci_diff_df

    print("\nKonfidenzintervalle der Metrik-Differenzen:")
    print(ci_diff_df)

    return bootstrap_diff_df, ci_diff_df

for model_a_name, model_b_name in pairwise_comparisons:
    run_paired_bootstrap_comparison(
        model_a_name=model_a_name,
        model_b_name=model_b_name
    )
    
# -------------------------------------------
# Ergebnisse der paarweisen Bootstrap-Vergleiche sammeln
# -------------------------------------------

if xgb_pairwise_ci_dfs:
    all_xgb_pairwise_ci_results = pd.concat(
        xgb_pairwise_ci_dfs.values(),
        ignore_index=True
    )
else:
    all_xgb_pairwise_ci_results = pd.DataFrame()

if xgb_pairwise_bootstrap_dfs:
    all_xgb_pairwise_bootstrap_results = pd.concat(
        xgb_pairwise_bootstrap_dfs.values(),
        ignore_index=True
    )
else:
    all_xgb_pairwise_bootstrap_results = pd.DataFrame()

all_xgb_pairwise_ci_results.to_csv(
    os.path.join(output_dir, "all_xgb_pairwise_ci_results.csv"),
    index=False
)

all_xgb_pairwise_bootstrap_results.to_csv(
    os.path.join(output_dir, "all_xgb_pairwise_bootstrap_results.csv"),
    index=False
)

all_xgb_pairwise_ci_results



# =========================================================
# Konfusionsmatrizen 
# =========================================================

confusion_matrix_output_dir = os.path.join(
    output_dir,
    "confusion_matrices"
)

os.makedirs(
    confusion_matrix_output_dir,
    exist_ok=True
)

# ---------------------------------------------------------
# Modelle auswählen
# ---------------------------------------------------------

results_dict = {
    # Modelle mit Data Leakage
    "baseline_with_icd10": result_baseline,
    "semantic_model_1": result_semantic_model_1,
    "semantic_model_2": result_semantic_model_2,
    "semantic_model_3": result_semantic_model_3,
    "semantic_model_4": result_semantic_model_4,
    "semantic_model_5": result_semantic_model_5,
    "semantic_model_6": result_semantic_model_6,

    # Modelle ohne Data Leakage
    "baseline_with_icd10_no_leakage": result_baseline_with_icd10_no_leakage,
    "semantic_model_1_no_leakage": result_semantic_model_1_no_leakage,
    "semantic_model_2_no_leakage": result_semantic_model_2_no_leakage,
    "semantic_model_3_no_leakage": result_semantic_model_3_no_leakage,
    "semantic_model_4_no_leakage": result_semantic_model_4_no_leakage,
    "semantic_model_5_no_leakage": result_semantic_model_5_no_leakage,
    "semantic_model_6_no_leakage": result_semantic_model_6_no_leakage,

    # CCSR-Benchmark
    "baseline_with_ccsr": result_baseline_with_ccsr
}

confusion_matrix_results_dict = {
    **results_dict,
    "baseline_no_icd10": result_baseline_no_icd10
}

confusion_matrix_model_names = list(
    confusion_matrix_results_dict.keys()
)

# ---------------------------------------------------------
# Konfusionsmatrix für ein Modell erzeugen
# ---------------------------------------------------------

def create_and_save_confusion_matrix_plots(
    model_name,
    result,
    y_test,
    output_directory
):
    """
    Erstellt für ein bereits trainiertes Modell:

    1. Konfusionsmatrix mit absoluten Fallzahlen
    2. zeilenweise normalisierte Konfusionsmatrix
    3. CSV-Tabelle mit TN, FP, FN und TP

    Der verwendete Threshold ist der auf den OOF-Trainingsdaten
    optimierte Threshold des jeweiligen Modells.
    """

    y_test = np.asarray(y_test)
    test_proba = np.asarray(
        result["test_proba"]
    )

    threshold = float(
        result["best_threshold"]
    )

    # -----------------------------------------------------
    # Sicherheitschecks
    # -----------------------------------------------------

    if len(test_proba) != len(y_test):
        raise ValueError(
            f"{model_name}: Anzahl Test-Predictions "
            f"({len(test_proba)}) stimmt nicht mit y_test "
            f"({len(y_test)}) überein."
        )

    if not np.isfinite(test_proba).all():
        raise ValueError(
            f"{model_name}: Die Test-Predictions enthalten "
            "NaN oder unendliche Werte."
        )

    # -----------------------------------------------------
    # Klassen anhand des modellspezifischen Thresholds
    # -----------------------------------------------------

    y_pred = (
        test_proba >= threshold
    ).astype(int)

    cm_absolute = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm_absolute.ravel()

    # Zeilenweise Normalisierung:
    # Zeile 0 = tatsächliche Überlebende
    # Zeile 1 = tatsächliche Verstorbene
    cm_normalized = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
        normalize="true"
    )

    # -----------------------------------------------------
    # Kennzahlen speichern
    # -----------------------------------------------------

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    summary_df = pd.DataFrame([
        {
            "model": model_name,
            "threshold": threshold,
            "n_test_cases": len(y_test),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "precision": precision
        }
    ])

    # -----------------------------------------------------
    # Gemeinsame Grafik:
    # links absolute Werte, rechts normalisierte Werte
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5)
    )

    # Absolute Fallzahlen
    display_absolute = ConfusionMatrixDisplay(
        confusion_matrix=cm_absolute,
        display_labels=[
            "Survived (0)",
            "Died (1)"
        ]
    )

    display_absolute.plot(
        ax=axes[0],
        colorbar=False,
        values_format="d"
    )

    axes[0].set_title(
        "Absolute counts"
    )

    axes[0].set_xlabel(
        "Predicted outcome"
    )

    axes[0].set_ylabel(
        "Actual outcome"
    )

    # Zeilenweise normalisierte Anteile
    display_normalized = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=[
            "Survived (0)",
            "Died (1)"
        ]
    )

    display_normalized.plot(
        ax=axes[1],
        colorbar=False,
        values_format=".2f"
    )

    axes[1].set_title(
        "Row-normalized proportions"
    )

    axes[1].set_xlabel(
        "Predicted outcome"
    )

    axes[1].set_ylabel(
        "Actual outcome"
    )

    fig.suptitle(
        f"Confusion matrices – {model_name}\n"
        f"Model-specific threshold = {threshold:.3f}",
        fontsize=14
    )

    plt.tight_layout()

    plot_path = os.path.join(
        output_directory,
        f"confusion_matrix_{model_name}.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n----------------------------------------")
    print(f"Konfusionsmatrix: {model_name}")
    print(f"Threshold: {threshold:.4f}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"Gespeichert: {plot_path}")

    return summary_df

# ---------------------------------------------------------
# Für alle ausgewählten Modelle ausführen
# ---------------------------------------------------------

confusion_matrix_summaries = {}

for model_name in confusion_matrix_model_names:

    if model_name not in confusion_matrix_results_dict:
        print(
            f"Übersprungen: Kein Ergebnis für "
            f"{model_name} vorhanden."
        )
        continue

    summary_df = create_and_save_confusion_matrix_plots(
        model_name=model_name,
        result=confusion_matrix_results_dict[model_name],
        y_test=y_test,
        output_directory=confusion_matrix_output_dir
    )

    confusion_matrix_summaries[model_name] = summary_df
    
# ---------------------------------------------------------
# Übersicht aller Modelle speichern
# ---------------------------------------------------------

if confusion_matrix_summaries:

    all_confusion_matrix_results = pd.concat(
        confusion_matrix_summaries.values(),
        ignore_index=True
    )

else:

    all_confusion_matrix_results = pd.DataFrame()


all_confusion_matrix_results.to_csv(
    os.path.join(
        confusion_matrix_output_dir,
        "all_models_confusion_matrix_results.csv"
    ),
    index=False
)

print("\nKonfusionsmatrizen abgeschlossen.")

all_confusion_matrix_results








# =========================================================
# Feature Importance
# =========================================================
# Nur ausgewählte Modelle:
# - Baseline mit ICD-10
# - Semantik-Modell 5
# - Semantik-Modell 6
# jeweils mit und ohne Data Leakage
# CCSR-Benchmark

selected_model_names = [
    "baseline_with_icd10",
    "semantic_model_5",
    "semantic_model_6",
    "baseline_with_icd10_no_leakage",
    "semantic_model_5_no_leakage",
    "semantic_model_6_no_leakage",
    "baseline_with_ccsr"
]


feature_names_dict = {
    # Modelle mit Data Leakage
    "baseline_with_icd10": baseline_feature_names,
    "semantic_model_1": semantic_model_1_feature_names,
    "semantic_model_2": semantic_model_2_feature_names,
    "semantic_model_3": semantic_model_3_feature_names,
    "semantic_model_4": semantic_model_4_feature_names,
    "semantic_model_5": semantic_model_5_feature_names,
    "semantic_model_6": semantic_model_6_feature_names,

    # Modelle ohne Data Leakage
    "baseline_with_icd10_no_leakage": baseline_no_leakage_feature_names,
    "semantic_model_1_no_leakage": semantic_model_1_no_leakage_feature_names,
    "semantic_model_2_no_leakage": semantic_model_2_no_leakage_feature_names,
    "semantic_model_3_no_leakage": semantic_model_3_no_leakage_feature_names,
    "semantic_model_4_no_leakage": semantic_model_4_no_leakage_feature_names,
    "semantic_model_5_no_leakage": semantic_model_5_no_leakage_feature_names,
    "semantic_model_6_no_leakage": semantic_model_6_no_leakage_feature_names,

    # CCSR-Benchmark
    "baseline_with_ccsr": baseline_with_ccsr_feature_names
}

selected_feature_sets = {
    name: feature_sets[name]
    for name in selected_model_names
}

# -------------------------------------------
# Sicherheitschecks
# -------------------------------------------

for feature_set_name, X_current in selected_feature_sets.items():
    assert X_current.shape[1] == len(feature_names_dict[feature_set_name]), (
        f"Feature-Anzahl passt nicht bei {feature_set_name}: "
        f"{X_current.shape[1]} Spalten, "
        f"{len(feature_names_dict[feature_set_name])} Feature-Namen"
    )

    assert X_current.shape[0] == len(y), (
        f"Zeilenanzahl passt nicht bei {feature_set_name}"
    )

print("\nAlle ausgewählten Feature-Sets haben passende Feature-Namen.")

# -------------------------------------------
# Hilfsfunktionen
# -------------------------------------------

def get_xgb_feature_importance(
    result,
    feature_names,
    feature_set_name,
    exclude_code_features=False
):
    """
    Extrahiert Feature Importances aus einem trainierten XGBoost-Modell.

    exclude_code_features=False:
        alle Features werden angezeigt.

    exclude_code_features=True:
        Features, die mit diag_code, proc_code, last_diag_code oder
        last_proc_code beginnen, werden entfernt.

    """

    model = result["final_model"]

    if len(feature_names) != model.n_features_in_:
        raise ValueError(
            f"Feature-Anzahl passt nicht für {feature_set_name}: "
            f"{len(feature_names)} Feature-Namen, aber Modell erwartet "
            f"{model.n_features_in_} Features."
        )

    importance_df = pd.DataFrame({
        "feature_set": feature_set_name,
        "feature": feature_names,
        "sklearn_importance": model.feature_importances_
    })

    booster = model.get_booster()

    importance_types = [
        "weight",
        "gain",
        "cover",
        "total_gain",
        "total_cover"
    ]

    for importance_type in importance_types:
        score_dict = booster.get_score(importance_type=importance_type)

        importance_df[importance_type] = [
            score_dict.get(f"f{i}", 0)
            for i in range(len(feature_names))
        ]

    # Multi-Hot ICD-10-Codes ausblenden (für bessere Übersichtlichkeit)
    if exclude_code_features:
        code_feature_prefixes = (
            "diag_code_",
            "proc_code_",
            "last_diag_code_",
            "last_proc_code_"
        )

        importance_df = importance_df[
            ~importance_df["feature"].str.startswith(code_feature_prefixes)
        ].copy()

    # Sortieren vor kumulativer Berechnung
    importance_df = importance_df.sort_values(
        by="total_gain",
        ascending=False
    ).reset_index(drop=True)

    # Relative total_gain berechnen
    total_gain_sum = importance_df["total_gain"].sum()

    if total_gain_sum > 0:
        importance_df["relative_total_gain"] = (
            importance_df["total_gain"] / total_gain_sum
        )
    else:
        importance_df["relative_total_gain"] = 0

    # Kumulative relative Feature Importance
    importance_df["cumulative_total_gain"] = (
        importance_df["relative_total_gain"].cumsum()
    )

    return importance_df


def get_both_feature_importances(
    result,
    feature_names,
    feature_set_name
):
    """
    Erstellt für ein Feature-Set zwei Feature-Importance-DataFrames:

    1. fi_all:
       Feature Importance für alle Features.

    2. fi_no_codes:
       Feature Importance ohne Multi-Hot ICD-10-Codes.
    """

    fi_all = get_xgb_feature_importance(
        result=result,
        feature_names=feature_names,
        feature_set_name=feature_set_name,
        exclude_code_features=False
    )

    fi_no_codes = get_xgb_feature_importance(
        result=result,
        feature_names=feature_names,
        feature_set_name=feature_set_name,
        exclude_code_features=True
    )

    return fi_all, fi_no_codes


# -------------------------------------------
# Feature Importances berechnen
# -------------------------------------------

feature_importance_all = {}
feature_importance_no_codes = {}

for feature_set_name in selected_model_names:
    fi_all, fi_no_codes = get_both_feature_importances(
        result=results_dict[feature_set_name],
        feature_names=feature_names_dict[feature_set_name],
        feature_set_name=feature_set_name
    )

    feature_importance_all[feature_set_name] = fi_all
    feature_importance_no_codes[feature_set_name] = fi_no_codes


# Jeweils eine Ausgabe mit allen Features und eine Ausgabe, wo die ICD-10-Codes ausgeschlossen 
# werden, um die Reihenfolge der semantischen Merkmale übersichtlicher einsehen zu können
fi_baseline_with_icd10_all = feature_importance_all["baseline_with_icd10"]
fi_baseline_with_icd10_no_codes = feature_importance_no_codes["baseline_with_icd10"]

fi_semantic_model_5_all = feature_importance_all["semantic_model_5"]
fi_semantic_model_5_no_codes = feature_importance_no_codes["semantic_model_5"]

fi_semantic_model_6_all = feature_importance_all["semantic_model_6"]
fi_semantic_model_6_no_codes = feature_importance_no_codes["semantic_model_6"]

fi_baseline_with_icd10_no_leakage_all = feature_importance_all["baseline_with_icd10_no_leakage"]
fi_baseline_with_icd10_no_leakage_no_codes = feature_importance_no_codes["baseline_with_icd10_no_leakage"]

fi_semantic_model_5_no_leakage_all = feature_importance_all["semantic_model_5_no_leakage"]
fi_semantic_model_5_no_leakage_no_codes = feature_importance_no_codes["semantic_model_5_no_leakage"]

fi_semantic_model_6_no_leakage_all = feature_importance_all["semantic_model_6_no_leakage"]
fi_semantic_model_6_no_leakage_no_codes = feature_importance_no_codes["semantic_model_6_no_leakage"]

# -------------------------------------------
# Feature Importance aller ausgewählten Modelle sammeln
# -------------------------------------------

fi_all_models = pd.concat(
    feature_importance_all.values(),
    ignore_index=True
)

fi_no_codes_models = pd.concat(
    feature_importance_no_codes.values(),
    ignore_index=True
)



# ============================================
# Feature-Gruppen-Analyse für ausgewählte Modelle
# ============================================

def assign_feature_group(feature, proc_metric_feature_names=None):
    if proc_metric_feature_names is None:
        proc_metric_feature_names = []

    # -------------------------------------------------
    # Basisvariablen
    # -------------------------------------------------
    if (
        feature.startswith("admission_type")
        or feature.startswith("admission_location")
        or feature in [
            "age_at_admission",
            "gender_F"
        ]
    ):
        return "basic_features"
    
    # -------------------------------------------------
    # Historische Aufenthaltsmerkmale
    # -------------------------------------------------
    elif (
        feature in [
            "n_prev_admissions",
            "n_prev_emergency_admissions",
            "time_since_last_discharge_days",
            "had_admission_last_30d",
            "max_past_los",
            "avg_past_los",
            "has_previous_admission"
        ]
    ):
        return "patient_history"

    # -------------------------------------------------
    # Direkte ICD-10-Codes
    # -------------------------------------------------
    elif (
        feature.startswith("diag_code")
        or feature.startswith("last_diag_code")
    ):
        return "diagnosis_code"

    elif (
        feature.startswith("proc_code")
        or feature.startswith("last_proc_code")
    ):
        return "procedure_code"

    # -------------------------------------------------
    # Diagnosemetriken 
    # -------------------------------------------------
    elif any(
        keyword in feature
        for keyword in [
            "depth",
            "leaf",
            "siblings",
            "subtree"
        ]
    ):
        return "diagnosis_metrics"

    # -------------------------------------------------
    # Behandlungsmetriken
    # -------------------------------------------------
    elif (
        feature in proc_metric_feature_names
        or any(
            keyword in feature
            for keyword in [
                "diversity",
                "invasiveness",
                "invasive",
                "section"
            ]
        )
    ):
        return "procedure_metrics"

    # -------------------------------------------------
    # Multi-Hot-Features der Diagnosekapitel und Organsysteme
    # -------------------------------------------------
    elif (
        feature.startswith("icd10_chapter")
        or feature.startswith("pcs_body_system")
        or feature.startswith("last_icd10_chapter")
        or feature.startswith("last_pcs_body_system")
    ):
        return "chapter_body_system_multihot"

    # -------------------------------------------------
    # Statistische Code-Merkmale
    # -------------------------------------------------
    elif (
        feature in [
            "n_diag_codes",
            "n_proc_codes",
            "n_diag_chapters",
            "last_n_diag_codes",
            "last_n_proc_codes",
            "last_n_diag_chapters"
        ]
        or feature.startswith("ic_")
        or feature.startswith("last_ic_")
    ):
        return "statistical_code_features"

    # -------------------------------------------------
    # CCSR-Gruppierungen
    # -------------------------------------------------
    elif feature.startswith("diag_ccsr"):
        return "diagnosis_ccsr_category"
    
    elif feature.startswith("proc_ccsr"):
        return "procedure_ccsr_category"
    
    # -------------------------------------------------
    # Fallback
    # -------------------------------------------------
    else:
        return "other"


def summarize_importance_by_group(fi_df, proc_metric_feature_names=None):
    temp = fi_df.copy()

    temp["feature_group"] = temp["feature"].apply(
        lambda x: assign_feature_group(
            x,
            proc_metric_feature_names=proc_metric_feature_names
        )
    )

    group_summary = (
        temp
        .groupby("feature_group")
        .agg(
            n_features=("feature", "count"),
            n_used=("total_gain", lambda x: (x > 0).sum()),
            total_gain=("total_gain", "sum"),
            mean_gain=("gain", "mean"),
            max_gain=("gain", "max"),
            mean_relative_total_gain=("relative_total_gain", "mean")
        )
        .reset_index()
    )

    total_gain_sum = group_summary["total_gain"].sum()

    if total_gain_sum > 0:
        group_summary["relative_group_total_gain"] = (
            group_summary["total_gain"] / total_gain_sum
        )
    else:
        group_summary["relative_group_total_gain"] = 0

    group_summary = group_summary.sort_values(
        "relative_group_total_gain",
        ascending=False
    )

    return group_summary

    
def get_proc_metric_names_for_feature_set(feature_set_name):
    if feature_set_name.endswith("_no_leakage"):
        return globals().get(
            "proc_metric_no_leakage_feature_names",
            []
        )
    else:
        return globals().get("proc_metric_feature_names", [])


group_summaries = {}

for feature_set_name in selected_model_names:
    group_summaries[feature_set_name] = summarize_importance_by_group(
        feature_importance_all[feature_set_name],
        proc_metric_feature_names=get_proc_metric_names_for_feature_set(feature_set_name)
    )


# Allgemeine Analyse
group_summary_baseline_with_icd10 = group_summaries["baseline_with_icd10"]
group_summary_semantic_model_5 = group_summaries["semantic_model_5"]
group_summary_semantic_model_6 = group_summaries["semantic_model_6"]

group_summary_baseline_with_icd10_no_leakage = group_summaries["baseline_with_icd10_no_leakage"]
group_summary_semantic_model_5_no_leakage = group_summaries["semantic_model_5_no_leakage"]
group_summary_semantic_model_6_no_leakage = group_summaries["semantic_model_6_no_leakage"]

# Beispiel anzeigen
group_summary_semantic_model_5


# ======================================================
# Export der Feature Importance- und Feature-Gruppen-Analyse in Excel
# ======================================================

output_file = os.path.join(output_dir, "feature_importance.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    # -----------------------------------------------------
    # 1. Feature Importance: alle Features
    # -----------------------------------------------------
    fi_all_models.to_excel(
        writer,
        sheet_name="FI_all_features",
        index=False
    )

    # -----------------------------------------------------
    # 2. Feature Importance: ohne Code-Dummies
    #    Gefilterte Ansicht (Feature Importance wird nicht neu berechnet)
    # -----------------------------------------------------
    fi_no_codes_models.to_excel(
        writer,
        sheet_name="FI_without_codes",
        index=False
    )

    # -----------------------------------------------------
    # 3. Gruppenanalyse je Modell
    # -----------------------------------------------------
    group_summary_all_models = pd.concat(
        [
            summary.assign(feature_set=model_name)
            for model_name, summary in group_summaries.items()
        ],
        ignore_index=True
    )

    group_summary_all_models.to_excel(
        writer,
        sheet_name="Group_summary",
        index=False
    )

    # -----------------------------------------------------
    # 5. Top 20 Features je Modell
    # -----------------------------------------------------
    top20_all_models = pd.concat(
        [
            fi_df.head(20).assign(feature_set=model_name)
            for model_name, fi_df in feature_importance_all.items()
        ],
        ignore_index=True
    )

    top20_all_models.to_excel(
        writer,
        sheet_name="Top20_all_features",
        index=False
    )

    # -----------------------------------------------------
    # 6. Top 20 ohne Multi-Hot Codes je Modell
    # -----------------------------------------------------
    top20_no_codes_all_models = pd.concat(
        [
            fi_df.head(20).assign(feature_set=model_name)
            for model_name, fi_df in feature_importance_no_codes.items()
        ],
        ignore_index=True
    )

    top20_no_codes_all_models.to_excel(
        writer,
        sheet_name="Top20_without_codes",
        index=False
    )

print("Export abgeschlossen.")





# ============================================
# Redundanz über Korrelation prüfen
# ============================================

def find_highly_correlated_features_df(X_df, threshold=0.90):
    corr = X_df.corr(method="spearman").abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    pairs = (
        upper
        .stack()
        .reset_index()
    )

    pairs.columns = ["feature_1", "feature_2", "abs_spearman_corr"]

    pairs = pairs[pairs["abs_spearman_corr"] >= threshold]

    pairs = pairs.sort_values(
        "abs_spearman_corr",
        ascending=False
    ).reset_index(drop=True)

    return pairs

# Anwendung auf Baseline ohne ICD-10
X_baseline_no_icd10_df = pd.DataFrame(
    X_baseline_no_icd10,
    columns=baseline_no_icd10_feature_names
)

high_corr_baseline_no_icd10 = find_highly_correlated_features_df(
    X_baseline_no_icd10_df,
    threshold=0.90
)

print(high_corr_baseline_no_icd10.head(30))

# Anwendung auf Semantik-Modell 6 des Hauptansatzes
X_semantic_model_6_df = pd.DataFrame(
    X_semantic_model_6.toarray() if sparse.issparse(X_semantic_model_6) else X_semantic_model_6,
    columns=semantic_model_6_feature_names
)

high_corr_semantic_model_6 = find_highly_correlated_features_df(
    X_semantic_model_6_df,
    threshold=0.90
)

print(high_corr_semantic_model_6.head(30))

# Anwendung auf Semantik-Modell 6 ohne Leakage
X_semantic_model_6_no_leakage_df = pd.DataFrame(
    X_semantic_model_6_no_leakage.toarray() if sparse.issparse(X_semantic_model_6_no_leakage) else X_semantic_model_6_no_leakage,
    columns=semantic_model_6_no_leakage_feature_names
)

high_corr_semantic_model_6_no_leakage = find_highly_correlated_features_df(
    X_semantic_model_6_no_leakage_df,
    threshold=0.90
)

print(high_corr_semantic_model_6_no_leakage.head(30))


# Vergleich von subtree_rel_log_mean/ _max / _std
subtree_features = [
    "subtree_rel_log_mean",
    "subtree_rel_log_std",
    "subtree_rel_log_max"
]

pattern = "|".join(subtree_features)

subtree_importance_semantic_model_6 = fi_semantic_model_6_all[
    fi_semantic_model_6_all["feature"].str.contains(pattern, regex=True, na=False)
][["feature", "total_gain", "relative_total_gain", "gain", "weight"]]

subtree_importance_semantic_model_6_no_leakage = fi_semantic_model_6_no_leakage_all[
    fi_semantic_model_6_no_leakage_all["feature"].str.contains(pattern, regex=True, na=False)
][["feature", "total_gain", "relative_total_gain", "gain", "weight"]]

print(subtree_importance_semantic_model_6)
print(subtree_importance_semantic_model_6_no_leakage)

"""
Entfernen:
    - leaf_count, weil es zu ähnlich zur Anzahl der Codes ist, weil der Großteil der
      Knoten Blätter sind
    - subtree_rel_log_std, weil es ggü. subtree_rel_log_mean nicht viel eigenständige
      Information liefert und schwieriger zu interpretieren ist
"""


# =========================================================
# Redundanz bei Multi-Hot-Features: zu seltene Features prüfen
# =========================================================

def classify_code_feature(feature):
    if (
        feature.startswith("diag_code")
        or feature.startswith("last_diag_code")
    ):
        return "diag_code"

    elif (
        feature.startswith("proc_code")
        or feature.startswith("last_proc_code")
    ):
        return "proc_code"

    else:
        return "other"


def summarize_code_feature_frequency_and_importance(
    X,
    feature_names,
    fi_df,
    min_count_thresholds=[1, 3, 5, 10, 20, 50]
):
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-Anzahl passt nicht: "
            f"{X.shape[1]} Spalten in X, aber "
            f"{len(feature_names)} Feature-Namen."
        )

    # Häufigkeit je Feature berechnen
    if sparse.issparse(X):
        counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        counts = np.asarray((X != 0).sum(axis=0)).ravel()

    freq_df = pd.DataFrame({
        "feature": feature_names,
        "n_nonzero": counts,
        "share_nonzero": counts / X.shape[0]
    })

    merged = freq_df.merge(
        fi_df[
            [
                "feature",
                "total_gain",
                "relative_total_gain",
                "gain",
                "weight"
            ]
        ],
        on="feature",
        how="left"
    )

    merged[
        [
            "total_gain",
            "relative_total_gain",
            "gain",
            "weight"
        ]
    ] = merged[
        [
            "total_gain",
            "relative_total_gain",
            "gain",
            "weight"
        ]
    ].fillna(0)

    merged["feature_type"] = merged["feature"].apply(classify_code_feature)

    summary = (
        merged
        .groupby("feature_type")
        .agg(
            n_features=("feature", "count"),
            n_used_gain_gt_0=("total_gain", lambda x: (x > 0).sum()),
            n_gain_eq_0=("total_gain", lambda x: (x == 0).sum()),
            median_n_nonzero=("n_nonzero", "median"),
            min_n_nonzero=("n_nonzero", "min"),
            max_n_nonzero=("n_nonzero", "max"),
            median_share_nonzero=("share_nonzero", "median"),
            total_gain_sum=("total_gain", "sum"),
            median_total_gain=("total_gain", "median"),
            max_total_gain=("total_gain", "max"),
            total_weight=("weight", "sum")
        )
        .reset_index()
    )

    summary["share_gain_eq_0"] = (
        summary["n_gain_eq_0"] / summary["n_features"]
    )

    summary["share_used_gain_gt_0"] = (
        summary["n_used_gain_gt_0"] / summary["n_features"]
    )

    threshold_rows = []

    for feature_type, group_df in merged.groupby("feature_type"):
        row = {
            "feature_type": feature_type,
            "n_features": len(group_df)
        }

        for threshold in min_count_thresholds:
            row[f"n_nonzero_lt_{threshold}"] = (
                group_df["n_nonzero"] < threshold
            ).sum()

            row[f"share_nonzero_lt_{threshold}"] = (
                group_df["n_nonzero"] < threshold
            ).mean()

        threshold_rows.append(row)

    frequency_threshold_summary = pd.DataFrame(threshold_rows)

    return summary, frequency_threshold_summary, merged


def summarize_importance_by_frequency_bins(merged_freq_fi_df):
    temp = merged_freq_fi_df.copy()

    temp["frequency_bin"] = pd.cut(
        temp["n_nonzero"],
        bins=[-1, 0, 2, 5, 10, 20, 50, 100, 500, 1000, np.inf],
        labels=[
            "0",
            "1-2",
            "3-5",
            "6-10",
            "11-20",
            "21-50",
            "51-100",
            "101-500",
            "501-1000",
            ">1000"
        ]
    )

    summary = (
        temp
        .groupby(["feature_type", "frequency_bin"], observed=True)
        .agg(
            n_features=("feature", "count"),
            n_zero_total_gain=("total_gain", lambda x: (x == 0).sum()),
            n_positive_total_gain=("total_gain", lambda x: (x > 0).sum()),
            total_gain_sum=("total_gain", "sum"),
            median_total_gain=("total_gain", "median"),
            max_total_gain=("total_gain", "max")
        )
        .reset_index()
    )

    summary["share_zero_total_gain"] = (
        summary["n_zero_total_gain"] / summary["n_features"]
    )

    return summary


# ---------------------------------------------------------
# Ausgewählte Modelle analysieren
# ---------------------------------------------------------

multi_hot_redundancy_inputs = {
    "baseline_with_icd10": {
        "X": X_baseline,
        "feature_names": baseline_feature_names,
        "fi": fi_baseline_with_icd10_all
    },
    "semantic_model_5": {
        "X": X_semantic_model_5,
        "feature_names": semantic_model_5_feature_names,
        "fi": fi_semantic_model_5_all
    },
    "baseline_with_icd10_no_leakage": {
        "X": X_baseline_no_leakage,
        "feature_names": baseline_no_leakage_feature_names,
        "fi": fi_baseline_with_icd10_no_leakage_all
    },
    "semantic_model_5_no_leakage": {
        "X": X_semantic_model_5_no_leakage,
        "feature_names": semantic_model_5_no_leakage_feature_names,
        "fi": fi_semantic_model_5_no_leakage_all
    }
}


multi_hot_summary = {}
multi_hot_frequency_thresholds = {}
multi_hot_merged_freq_fi = {}
multi_hot_frequency_bins = {}

for model_name, inputs in multi_hot_redundancy_inputs.items():

    print("\n==================================================")
    print(f"Multi-Hot-Redundanzanalyse: {model_name}")
    print("==================================================")

    summary, freq_thresholds, merged_freq_fi = summarize_code_feature_frequency_and_importance(
        X=inputs["X"],
        feature_names=inputs["feature_names"],
        fi_df=inputs["fi"]
    )

    freq_bins = summarize_importance_by_frequency_bins(
        merged_freq_fi
    )

    multi_hot_summary[model_name] = summary
    multi_hot_frequency_thresholds[model_name] = freq_thresholds
    multi_hot_merged_freq_fi[model_name] = merged_freq_fi
    multi_hot_frequency_bins[model_name] = freq_bins

    print("\nÜbersicht nach Feature-Typ:")
    print(summary)

    print("\nHäufigkeitsschwellen:")
    print(freq_thresholds)

    print("\nImportance nach Häufigkeits-Bins, nur Code-Features:")
    print(
        freq_bins[
            freq_bins["feature_type"].isin(
                ["diag_code", "proc_code"]
            )
        ]
    )


# ---------------------------------------------------------
# Einzelne Variablen für den Variable Explorer
# ---------------------------------------------------------

multi_hot_summary_baseline_with_icd10 = multi_hot_summary["baseline_with_icd10"]
multi_hot_summary_semantic_model_5 = multi_hot_summary["semantic_model_5"]
multi_hot_summary_baseline_with_icd10_no_leakage = multi_hot_summary["baseline_with_icd10_no_leakage"]
multi_hot_summary_semantic_model_5_no_leakage = multi_hot_summary["semantic_model_5_no_leakage"]

multi_hot_freq_bins_baseline_with_icd10 = multi_hot_frequency_bins["baseline_with_icd10"]
multi_hot_freq_bins_semantic_model_5 = multi_hot_frequency_bins["semantic_model_5"]
multi_hot_freq_bins_baseline_with_icd10_no_leakage = multi_hot_frequency_bins["baseline_with_icd10_no_leakage"]
multi_hot_freq_bins_semantic_model_5_no_leakage = multi_hot_frequency_bins["semantic_model_5_no_leakage"]


# =========================================================
# Export der Multi-Hot-Redundanzanalyse
# =========================================================

output_file_multi_hot = os.path.join(output_dir, "multi_hot_redundancy_analysis.xlsx")

with pd.ExcelWriter(output_file_multi_hot, engine="openpyxl") as writer:

    # -----------------------------------------------------
    # 1. Zusammenfassung nach Feature-Typ
    # -----------------------------------------------------
    multi_hot_summary_all = pd.concat(
        [
            df.assign(model=model_name)
            for model_name, df in multi_hot_summary.items()
        ],
        ignore_index=True
    )

    multi_hot_summary_all.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    # -----------------------------------------------------
    # 2. Häufigkeitsschwellen
    # -----------------------------------------------------
    multi_hot_frequency_thresholds_all = pd.concat(
        [
            df.assign(model=model_name)
            for model_name, df in multi_hot_frequency_thresholds.items()
        ],
        ignore_index=True
    )

    multi_hot_frequency_thresholds_all.to_excel(
        writer,
        sheet_name="Frequency_thresholds",
        index=False
    )

    # -----------------------------------------------------
    # 3. Importance nach Häufigkeits-Bins
    # -----------------------------------------------------
    multi_hot_frequency_bins_all = pd.concat(
        [
            df.assign(model=model_name)
            for model_name, df in multi_hot_frequency_bins.items()
        ],
        ignore_index=True
    )

    multi_hot_frequency_bins_all.to_excel(
        writer,
        sheet_name="Frequency_bins",
        index=False
    )

    # -----------------------------------------------------
    # 4. Feature-Level-Tabelle
    # -----------------------------------------------------
    # Enthält jedes einzelne Feature mit Häufigkeit und Importance.
    multi_hot_feature_level_all = pd.concat(
        [
            df.assign(model=model_name)
            for model_name, df in multi_hot_merged_freq_fi.items()
        ],
        ignore_index=True
    )

    multi_hot_feature_level_all.to_excel(
        writer,
        sheet_name="Feature_level",
        index=False
    )

    # -----------------------------------------------------
    # 5. Nur Code-Features auf Feature-Level
    # -----------------------------------------------------
    multi_hot_code_features_only = multi_hot_feature_level_all[
        multi_hot_feature_level_all["feature_type"].isin(
            ["diag_code", "proc_code"]
        )
    ].copy()

    multi_hot_code_features_only.to_excel(
        writer,
        sheet_name="Code_features_only",
        index=False
    )

print(f"Export abgeschlossen: {output_file_multi_hot}")





# =========================================================
# SHAP-Analyse
# =========================================================

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------

shap_output_dir = os.path.join(
    output_dir,
    "shap_analysis"
)

os.makedirs(shap_output_dir, exist_ok=True)

SHAP_CONFIG = {
    "sample_size": 5000,
    "max_display": 25, # Anzahl der Features in den Grafiken
    "random_state": 42
}

shap_model_names = [
    # Mit Temporal Leakage
    "baseline_with_icd10",
    "semantic_model_1",
    "semantic_model_2",
    "semantic_model_3",
    "semantic_model_4",
    "semantic_model_5",
    "semantic_model_6",

    # Ohne Temporal Leakage
    "baseline_with_icd10_no_leakage",
    "semantic_model_1_no_leakage",
    "semantic_model_2_no_leakage",
    "semantic_model_3_no_leakage",
    "semantic_model_4_no_leakage",
    "semantic_model_5_no_leakage",
    "semantic_model_6_no_leakage",

    # CCSR-Benchmark
    "baseline_with_ccsr"
]

# Für die globale SHAP-Analyse werden mehr Modelle betrachtet als
# in der vorherigen Feature-Importance-Analyse.
shap_feature_sets = {
    model_name: feature_sets[model_name]
    for model_name in shap_model_names
}

# ---------------------------------------------------------
# Stratifizierte Stichprobe aus dem Testset ziehen
# ---------------------------------------------------------

def get_stratified_shap_sample_indices(
    y,
    candidate_indices,
    sample_size=1000,
    random_state=42
):
    """
    Zieht eine nach dem Outcome stratifizierte Stichprobe aus den
    angegebenen Kandidatenindizes.

    Die Mortalitätsrate der Stichprobe entspricht ungefähr derjenigen
    im Testset.
    """

    rng = np.random.default_rng(random_state)

    candidate_indices = np.asarray(candidate_indices)
    y_candidate = y[candidate_indices]

    positive_indices = candidate_indices[y_candidate == 1]
    negative_indices = candidate_indices[y_candidate == 0]

    if sample_size >= len(candidate_indices):
        return np.sort(candidate_indices)

    positive_share = len(positive_indices) / len(candidate_indices)

    n_positive_sample = int(
        round(sample_size * positive_share)
    )

    n_positive_sample = min(
        n_positive_sample,
        len(positive_indices)
    )

    n_negative_sample = sample_size - n_positive_sample

    n_negative_sample = min(
        n_negative_sample,
        len(negative_indices)
    )

    sampled_positive = rng.choice(
        positive_indices,
        size=n_positive_sample,
        replace=False
    )

    sampled_negative = rng.choice(
        negative_indices,
        size=n_negative_sample,
        replace=False
    )

    sampled_indices = np.concatenate([
        sampled_positive,
        sampled_negative
    ])

    rng.shuffle(sampled_indices)

    return sampled_indices


shap_sample_indices = get_stratified_shap_sample_indices(
    y=y,
    candidate_indices=test_idx,
    sample_size=SHAP_CONFIG["sample_size"],
    random_state=SHAP_CONFIG["random_state"]
)

print("\n=== SHAP-Stichprobe ===")
print("Anzahl Fälle:", len(shap_sample_indices))
print(
    "Mortalitätsrate SHAP-Stichprobe:",
    y[shap_sample_indices].mean()
)
print(
    "Mortalitätsrate gesamtes Testset:",
    y[test_idx].mean()
)


# ---------------------------------------------------------
# Hilfsfunktion: Matrix in für SHAP geeignete Form bringen
# ---------------------------------------------------------

def prepare_shap_matrix(X, row_indices):
    """
    Extrahiert die benötigten Zeilen und erhält die ursprüngliche
    Sparse-Repräsentation für eine konsistente XGBoost-Auswertung.
    """

    X_sample = get_subset(
        X,
        row_indices
    )

    if sparse.issparse(X_sample):
        return X_sample.astype(
            np.float32
        ).tocsr()

    return np.asarray(
        X_sample,
        dtype=np.float32
    )

# ---------------------------------------------------------
# Globale SHAP-Auswertung für ein Modell
# ---------------------------------------------------------

def run_global_shap_analysis(
    model_name,
    result,
    X,
    feature_names,
    sample_indices,
    output_directory,
    max_display=25
):

    model = result["final_model"]

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{model_name}: X besitzt {X.shape[1]} Features, "
            f"aber es wurden {len(feature_names)} Feature-Namen übergeben."
        )

    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            f"{model_name}: Das Modell erwartet {model.n_features_in_} "
            f"Features, aber es liegen {len(feature_names)} Namen vor."
        )

    # Nur Teststichprobe materialisieren
    X_shap = prepare_shap_matrix(
        X=X,
        row_indices=sample_indices
    )

    # TreeExplainer für XGBoost
    explainer = shap.TreeExplainer(
        model,
        feature_names=feature_names
    )
    
    shap_values_raw = explainer(
        X_shap,
        check_additivity=False
    )
    
    # Dichte Featurewerte nur für die grafische Darstellung erzeugen
    if sparse.issparse(X_shap):
        X_shap_plot_data = X_shap.toarray()
    else:
        X_shap_plot_data = np.asarray(
            X_shap
        )
    
    X_shap_plot_data = np.asarray(
        X_shap_plot_data,
        dtype=np.float32
    )
    
    shap_values = shap.Explanation(
        values=np.asarray(
            shap_values_raw.values
        ),
        base_values=np.asarray(
            shap_values_raw.base_values
        ),
        data=X_shap_plot_data,
        feature_names=list(feature_names)
    )
    
    # Sicherheitschecks
    if shap_values.values.shape[1] != len(feature_names):
        raise ValueError(
            f"{model_name}: SHAP besitzt "
            f"{shap_values.values.shape[1]} Feature-Spalten, "
            f"aber es liegen {len(feature_names)} Feature-Namen vor."
        )
    
    if shap_values.data.shape != shap_values.values.shape:
        raise ValueError(
            f"{model_name}: SHAP-Werte und Darstellungsdaten "
            f"haben unterschiedliche Shapes: "
            f"{shap_values.values.shape} gegenüber "
            f"{shap_values.data.shape}."
        )

    # -----------------------------------------------------
    # Additivitätscheck für eine kleine Teilmenge
    # -----------------------------------------------------
    
    n_additivity_checks = min(
        10,
        X_shap.shape[0]
    )
    
    check_positions = np.arange(
        n_additivity_checks
    )
    
    X_check = X_shap[
        check_positions
    ]
    
    model_margins_check = np.asarray(
        model.predict(
            X_check,
            output_margin=True
        )
    ).reshape(-1)
    
    shap_margins_check = (
        np.asarray(
            shap_values.base_values
        ).reshape(-1)[check_positions]
        + shap_values.values[
            check_positions
        ].sum(axis=1)
    )
    
    max_additivity_difference = float(
        np.max(
            np.abs(
                model_margins_check
                - shap_margins_check
            )
        )
    )
    
    print(
        f"{model_name}: maximale SHAP-Additivitätsabweichung "
        f"bei {n_additivity_checks} Prüffällen: "
        f"{max_additivity_difference:.8f}"
    )
    
    assert np.allclose(
        model_margins_check,
        shap_margins_check,
        atol=1e-3,
        rtol=1e-5
    ), (
        f"{model_name}: Die globalen SHAP-Werte rekonstruieren "
        "den XGBoost-Margin nicht ausreichend genau."
    )
        
    # -----------------------------------------------------
    # Globale Feature-Wichtigkeit:
    # Mittelwert der absoluten SHAP-Werte
    # -----------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(axis=0)

    mean_signed_shap = shap_values.values.mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature_set": model_name,
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap
    })

    shap_importance_df = shap_importance_df.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    total_mean_abs_shap = shap_importance_df[
        "mean_abs_shap"
    ].sum()

    if total_mean_abs_shap > 0:
        shap_importance_df["relative_mean_abs_shap"] = (
            shap_importance_df["mean_abs_shap"]
            / total_mean_abs_shap
        )
    else:
        shap_importance_df["relative_mean_abs_shap"] = 0

    shap_importance_df["cumulative_relative_mean_abs_shap"] = (
        shap_importance_df["relative_mean_abs_shap"].cumsum()
    )

    # CSV speichern
    csv_path = os.path.join(
        output_directory,
        f"shap_importance_{model_name}.csv"
    )

    shap_importance_df.to_csv(
        csv_path,
        index=False
    )

    # -----------------------------------------------------
    # 1. Globaler Top-SHAP-Barplot ohne Sammelbalken
    # -----------------------------------------------------
    
    top_shap_df = (
        shap_importance_df
        .head(max_display)
        .sort_values(
            "mean_abs_shap",
            ascending=True
        )
    )
    
    plt.figure(figsize=(10, 8))
    
    plt.barh(
        top_shap_df["feature"],
        top_shap_df["mean_abs_shap"]
    )
    
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(
        f"Top {max_display} global SHAP features – {model_name}"
    )
    
    plt.tight_layout()
    
    barplot_path = os.path.join(
        output_directory,
        f"shap_bar_{model_name}.png"
    )
    
    plt.savefig(
        barplot_path,
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.close()    
    
    # -----------------------------------------------------
    # 2. SHAP-Beeswarm-Plot
    # -----------------------------------------------------

    # Zoomgrenze nur anhand der wichtigsten dargestellten Features bestimmen
    mean_abs_shap_for_zoom = np.abs(
        shap_values.values
    ).mean(axis=0)
    
    top_feature_indices_for_zoom = np.argsort(
        mean_abs_shap_for_zoom
    )[-max_display:]
    
    abs_shap_values_for_zoom = np.abs(
        shap_values.values[:, top_feature_indices_for_zoom]
    ).ravel()
    
    robust_x_limit = float(
        np.quantile(
            abs_shap_values_for_zoom,
            0.995
        )
    )
    
    if not np.isfinite(robust_x_limit) or robust_x_limit <= 0:
        robust_x_limit = float(
            np.max(abs_shap_values_for_zoom)
        )
    
    # Etwas Abstand zum Rand
    robust_x_limit *= 1.10    
    
    # -----------------------------------------------------
    # A) Vollständiger Beeswarm ohne abgeschnittene Werte
    # -----------------------------------------------------
    
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        plot_size=(13, 10),
        show=False
    )
    
    ax = plt.gca()
    
    # Null-Linie stärker hervorheben
    ax.axvline(
        0,
        color="black",
        linewidth=1.2,
        alpha=0.8,
        zorder=0
    )
    
    # Zusätzliche vertikale Hilfslinien
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )
    
    ax.set_title(
        f"SHAP summary – {model_name}",
        fontsize=14,
        pad=15
    )
    
    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=12
    )
    
    ax.tick_params(
        axis="x",
        labelsize=10
    )
    
    ax.tick_params(
        axis="y",
        labelsize=10
    )
    
    plt.tight_layout()
    
    beeswarm_path = os.path.join(
        output_directory,
        f"shap_beeswarm_{model_name}.png"
    )
    
    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.close()
    
    
    # -----------------------------------------------------
    # B) Zusätzlicher gezoomter Beeswarm
    # -----------------------------------------------------
    
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        plot_size=(13, 10),
        show=False
    )
    
    ax = plt.gca()
    
    ax.axvline(
        0,
        color="black",
        linewidth=1.2,
        alpha=0.8,
        zorder=0
    )
    
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )
    
    # Symmetrischer Zoom um 0
    ax.set_xlim(
        -robust_x_limit,
        robust_x_limit
    )
    
    ax.set_title(
        f"SHAP summary – {model_name}\n"
        f"Zoomed to 99.5th percentile of SHAP values among top {max_display} features",
        fontsize=14,
        pad=15
    )
    
    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=12
    )
    
    ax.tick_params(
        axis="x",
        labelsize=10
    )
    
    ax.tick_params(
        axis="y",
        labelsize=10
    )
    
    plt.tight_layout()
    
    beeswarm_zoomed_path = os.path.join(
        output_directory,
        f"shap_beeswarm_zoomed_{model_name}.png"
    )
    
    plt.savefig(
        beeswarm_zoomed_path,
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.close()
    # -----------------------

    del shap_values_raw
    del X_shap
    del X_shap_plot_data
    
    gc.collect()

    return {
        "model_name": model_name,
        "explainer": explainer,
        "shap_values": shap_values,
        "sample_indices": np.asarray(sample_indices),
        "importance_df": shap_importance_df
    }

# ---------------------------------------------------------
# Analyse für alle ausgewählten Modelle durchführen
# ---------------------------------------------------------

shap_results = {}
shap_importance_dfs = {}

for model_name in shap_model_names:

    if model_name not in results_dict:
        print(
            f"Übersprungen, da kein Modellergebnis vorhanden ist: "
            f"{model_name}"
        )
        continue

    if model_name not in shap_feature_sets:
        print(
            f"Übersprungen, da kein Feature-Set vorhanden ist: "
            f"{model_name}"
        )
        continue

    if model_name not in feature_names_dict:
        print(
            f"Übersprungen, da keine Feature-Namen vorhanden sind: "
            f"{model_name}"
        )
        continue

    shap_result = run_global_shap_analysis(
        model_name=model_name,
        result=results_dict[model_name],
        X=shap_feature_sets[model_name],
        feature_names=feature_names_dict[model_name],
        sample_indices=shap_sample_indices,
        output_directory=shap_output_dir,
        max_display=SHAP_CONFIG["max_display"]
    )

    shap_results[model_name] = shap_result
    shap_importance_dfs[model_name] = shap_result[
        "importance_df"
    ]

    # Nicht benötigte temporäre Objekte freigeben
    gc.collect()


# ---------------------------------------------------------
# Alle globalen SHAP-Importances zusammenführen
# ---------------------------------------------------------

if shap_importance_dfs:

    all_shap_importance = pd.concat(
        shap_importance_dfs.values(),
        ignore_index=True
    )

    all_shap_importance.to_csv(
        os.path.join(
            shap_output_dir,
            "all_models_shap_importance.csv"
        ),
        index=False
    )

else:
    all_shap_importance = pd.DataFrame()


print("\nGlobale SHAP-Analyse abgeschlossen.")


# ---------------------------------------------------------
# SHAP-Grafiken ohne einzelne ICD-Code-Dummies
# ---------------------------------------------------------

def is_direct_code_feature(feature):
    return str(feature).startswith((
        "diag_code_",
        "proc_code_",
        "last_diag_code_",
        "last_proc_code_"
    ))


def create_shap_plots_without_codes(
    shap_result,
    output_directory,
    max_display=25
):
    model_name = shap_result["model_name"]
    shap_values = shap_result["shap_values"]

    # Feature-Namen als normale Python-Liste
    all_feature_names = list(shap_values.feature_names)

    if len(all_feature_names) != shap_values.values.shape[1]:
        raise ValueError(
            f"{model_name}: Anzahl der Feature-Namen passt nicht zu den "
            f"SHAP-Spalten: {len(all_feature_names)} Namen gegenüber "
            f"{shap_values.values.shape[1]} Spalten."
        )

    # Zu behaltende Features
    keep_indices = [
        i
        for i, feature in enumerate(all_feature_names)
        if not is_direct_code_feature(feature)
    ]

    filtered_feature_names = [
        all_feature_names[i]
        for i in keep_indices
    ]

    if len(keep_indices) == 0:
        print(
            f"{model_name}: Nach Ausschluss der Code-Features "
            "sind keine Features mehr vorhanden."
        )
        return None

    # SHAP-Werte und Feature-Daten explizit filtern
    filtered_values = shap_values.values[:, keep_indices]

    if shap_values.data is not None:
        filtered_data = np.asarray(
            shap_values.data
        )[:, keep_indices]
    else:
        filtered_data = None

    # Base Values unverändert übernehmen
    filtered_base_values = shap_values.base_values

    # Neues Explanation-Objekt erzeugen
    shap_values_no_codes = shap.Explanation(
        values=filtered_values,
        base_values=filtered_base_values,
        data=filtered_data,
        feature_names=filtered_feature_names
    )

    print("\n----------------------------------------")
    print(f"SHAP ohne Codes: {model_name}")
    print("Features vorher:", len(all_feature_names))
    print("Features nachher:", len(filtered_feature_names))

    assert (
        shap_values_no_codes.values.shape[1]
        == len(shap_values_no_codes.feature_names)
    )

    # -----------------------------------------------------
    # Importance-Tabelle ohne Code-Dummies
    # -----------------------------------------------------

    mean_abs_shap_no_codes = np.abs(
        shap_values_no_codes.values
    ).mean(axis=0)

    mean_signed_shap_no_codes = (
        shap_values_no_codes.values.mean(axis=0)
    )

    shap_importance_no_codes_df = pd.DataFrame({
        "feature_set": model_name,
        "feature": filtered_feature_names,
        "mean_abs_shap": mean_abs_shap_no_codes,
        "mean_signed_shap": mean_signed_shap_no_codes
    })

    shap_importance_no_codes_df = (
        shap_importance_no_codes_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_mean_abs_shap = (
        shap_importance_no_codes_df["mean_abs_shap"].sum()
    )

    if total_mean_abs_shap > 0:
        shap_importance_no_codes_df[
            "relative_mean_abs_shap"
        ] = (
            shap_importance_no_codes_df["mean_abs_shap"]
            / total_mean_abs_shap
        )
    else:
        shap_importance_no_codes_df[
            "relative_mean_abs_shap"
        ] = 0

    shap_importance_no_codes_df[
        "cumulative_relative_mean_abs_shap"
    ] = (
        shap_importance_no_codes_df[
            "relative_mean_abs_shap"
        ].cumsum()
    )

    # Tabelle exportieren
    shap_importance_no_codes_df.to_csv(
        os.path.join(
            output_directory,
            f"shap_importance_without_codes_{model_name}.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # Top-SHAP-Barplot ohne Codes
    # -----------------------------------------------------

    top_shap_no_codes_df = (
        shap_importance_no_codes_df
        .head(max_display)
        .sort_values(
            "mean_abs_shap",
            ascending=True
        )
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_shap_no_codes_df["feature"],
        top_shap_no_codes_df["mean_abs_shap"]
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(
        f"Top {max_display} SHAP features without code dummies – "
        f"{model_name}"
    )

    plt.tight_layout()

    barplot_path = os.path.join(
        output_directory,
        f"shap_bar_without_codes_{model_name}.png"
    )

    plt.savefig(
        barplot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------
    # Robuste x-Achsen-Grenze für gezoomten Beeswarm
    # -----------------------------------------------------

    # Zoomgrenze nur anhand der wichtigsten dargestellten Features bestimmen
    mean_abs_shap_for_zoom = np.abs(
        shap_values_no_codes.values
    ).mean(axis=0)
    
    top_feature_indices_for_zoom = np.argsort(
        mean_abs_shap_for_zoom
    )[-max_display:]
    
    abs_shap_values_for_zoom = np.abs(
        shap_values_no_codes.values[:, top_feature_indices_for_zoom]
    ).ravel()
    
    robust_x_limit = float(
        np.quantile(
            abs_shap_values_for_zoom,
            0.995
        )
    )
    
    if not np.isfinite(robust_x_limit) or robust_x_limit <= 0:
        robust_x_limit = float(
            np.max(abs_shap_values_for_zoom)
        )
    
    robust_x_limit *= 1.10

    # -----------------------------------------------------
    # A) Vollständiger Beeswarm ohne Code-Dummies
    # -----------------------------------------------------

    shap.plots.beeswarm(
        shap_values_no_codes,
        max_display=max_display,
        plot_size=(13, 10),
        show=False
    )

    ax = plt.gca()

    ax.axvline(
        0,
        color="black",
        linewidth=1.2,
        alpha=0.8,
        zorder=0
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    ax.set_title(
        f"SHAP summary without code dummies – {model_name}",
        fontsize=14,
        pad=15
    )

    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=12
    )

    ax.tick_params(
        axis="x",
        labelsize=10
    )

    ax.tick_params(
        axis="y",
        labelsize=10
    )

    plt.tight_layout()

    beeswarm_path = os.path.join(
        output_directory,
        f"shap_beeswarm_without_codes_{model_name}.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------
    # B) Gezoomter Beeswarm ohne Code-Dummies
    # -----------------------------------------------------

    shap.plots.beeswarm(
        shap_values_no_codes,
        max_display=max_display,
        plot_size=(13, 10),
        show=False
    )

    ax = plt.gca()

    ax.axvline(
        0,
        color="black",
        linewidth=1.2,
        alpha=0.8,
        zorder=0
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    ax.set_xlim(
        -robust_x_limit,
        robust_x_limit
    )

    ax.set_title(
        f"SHAP summary without code dummies – {model_name}\n"
        f"Zoomed to 99.5th percentile of SHAP values among top {max_display} features",
        fontsize=14,
        pad=15
    )

    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=12
    )

    ax.tick_params(
        axis="x",
        labelsize=10
    )

    ax.tick_params(
        axis="y",
        labelsize=10
    )

    plt.tight_layout()

    beeswarm_zoomed_path = os.path.join(
        output_directory,
        f"shap_beeswarm_without_codes_zoomed_{model_name}.png"
    )

    plt.savefig(
        beeswarm_zoomed_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "shap_values": shap_values_no_codes,
        "importance_df": shap_importance_no_codes_df
    }

shap_without_codes_results = {}

for model_name, shap_result in shap_results.items():

    shap_without_codes_results[model_name] = (
        create_shap_plots_without_codes(
            shap_result=shap_result,
            output_directory=shap_output_dir,
            max_display=SHAP_CONFIG["max_display"]
        )
    )    

# =========================================================
# SHAP-Auswertung nach Feature-Gruppen
# =========================================================

def summarize_shap_by_feature_group(
    shap_importance_df,
    proc_metric_feature_names=None
):
    temp = shap_importance_df.copy()

    temp["feature_group"] = temp["feature"].apply(
        lambda feature: assign_feature_group(
            feature,
            proc_metric_feature_names=proc_metric_feature_names
        )
    )

    group_summary = (
        temp
        .groupby("feature_group")
        .agg(
            n_features=("feature", "count"),
            group_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_feature_abs_shap=("mean_abs_shap", "mean"),
            max_feature_abs_shap=("mean_abs_shap", "max")
        )
        .reset_index()
    )

    total_shap = group_summary[
        "group_mean_abs_shap"
    ].sum()

    if total_shap > 0:
        group_summary["relative_group_mean_abs_shap"] = (
            group_summary["group_mean_abs_shap"]
            / total_shap
        )
    else:
        group_summary["relative_group_mean_abs_shap"] = 0

    return group_summary.sort_values(
        "relative_group_mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)


shap_group_summaries = {}

for model_name, importance_df in shap_importance_dfs.items():

    shap_group_summaries[model_name] = (
        summarize_shap_by_feature_group(
            shap_importance_df=importance_df,
            proc_metric_feature_names=(
                get_proc_metric_names_for_feature_set(model_name)
            )
        )
    )


all_shap_group_summaries = pd.concat(
    [
        summary.assign(feature_set=model_name)
        for model_name, summary in shap_group_summaries.items()
    ],
    ignore_index=True
)

all_shap_group_summaries.to_csv(
    os.path.join(
        shap_output_dir,
        "all_models_shap_group_summary.csv"
    ),
    index=False
)

# ---------------------------------------------------------
# Sicherheitscheck: Die relativen Gruppenwerte sollten pro Modell ungefähr 1 ergeben.
# ---------------------------------------------------------

for model_name, summary in shap_group_summaries.items():
    relative_sum = summary[
        "relative_group_mean_abs_shap"
    ].sum()

    assert np.isclose(
        relative_sum,
        1.0,
        atol=1e-8
    ), (
        f"Relative SHAP-Gruppenwerte summieren sich bei "
        f"{model_name} nicht zu 1: {relative_sum}"
    )

other_check = all_shap_group_summaries[
    all_shap_group_summaries["feature_group"] == "other"
]

print("\nZusammenfassung der Gruppe 'other':")
print(other_check)


# ---------------------------------------------------------
# Grafik für Feature-Gruppen
# ---------------------------------------------------------

def plot_shap_group_importance(
    group_summary,
    model_name,
    output_directory
):
    plot_df = group_summary.sort_values(
        "relative_group_mean_abs_shap",
        ascending=True
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_df["feature_group"],
        plot_df["relative_group_mean_abs_shap"]
    )

    plt.xlabel("Relative sum of mean absolute SHAP values")
    plt.ylabel("Feature group")
    plt.title(f"SHAP importance by feature group – {model_name}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"shap_group_importance_{model_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# Durchschnittliche SHAP-Importance pro Feature
def plot_mean_shap_per_feature_group(
    group_summary,
    model_name,
    output_directory
):
    """
    Stellt die durchschnittliche globale SHAP-Importance eines einzelnen
    Features innerhalb jeder Feature-Gruppe dar.

    Anders als relative_group_mean_abs_shap wird der Wert nicht durch
    Aufsummieren aller Features einer Gruppe gebildet. Dadurch wird der
    Einfluss unterschiedlicher Gruppengrößen reduziert.
    """

    plot_df = (
        group_summary
        .sort_values(
            "mean_feature_abs_shap",
            ascending=True
        )
        .copy()
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_df["feature_group"],
        plot_df["mean_feature_abs_shap"]
    )

    plt.xlabel("Mean absolute SHAP value per feature")
    plt.ylabel("Feature group")
    plt.title(
        f"Mean SHAP importance per feature group – {model_name}"
    )

    # Vertikale Hilfslinien
    plt.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"shap_mean_per_feature_group_{model_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# Ausführen
for model_name, group_summary in shap_group_summaries.items():
    
    # 1. Gesamter SHAP-Anteil der Feature-Gruppe
    plot_shap_group_importance(
        group_summary=group_summary,
        model_name=model_name,
        output_directory=shap_output_dir
    )
    
    # 2. Durchschnittliche SHAP-Importance pro Feature
    plot_mean_shap_per_feature_group(
        group_summary=group_summary,
        model_name=model_name,
        output_directory=shap_output_dir
    )
    
# ---------------------------------------------------------
# Tabelle für Feature-Gruppen
# ---------------------------------------------------------

group_table_columns = [
    "feature_set",
    "feature_group",
    "n_features",
    "group_mean_abs_shap",
    "relative_group_mean_abs_shap",
    "mean_feature_abs_shap",
    "max_feature_abs_shap"
]

shap_group_table = all_shap_group_summaries[
    group_table_columns
].copy()

shap_group_table = shap_group_table.sort_values(
    [
        "feature_set",
        "relative_group_mean_abs_shap"
    ],
    ascending=[True, False]
)

columns_to_round = [
    "group_mean_abs_shap",
    "relative_group_mean_abs_shap",
    "mean_feature_abs_shap",
    "max_feature_abs_shap"
]

shap_group_table[
    columns_to_round
] = shap_group_table[
    columns_to_round
].round(4)
    
shap_group_table.to_excel(
    os.path.join(
        shap_output_dir,
        "shap_group_analysis.xlsx"
    ),
    index=False
)



# =========================================================
# SHAP Interaction Values
# =========================================================
# Nur für:
# - Semantik-Modell 6 mit Temporal Leakage
# - Semantik-Modell 6 ohne Temporal Leakage
#
# Die Stichprobe ist bewusst kleiner als bei der normalen globalen SHAP-Analyse, 
# da Interaction Values quadratisch mit der Anzahl der Features wachsen.

SHAP_INTERACTION_CONFIG = {
    "sample_size": 1000,
    "top_features_heatmap": 20,
    "top_pairs": 25,
    "random_state": 42
}

shap_interaction_model_names = [
    "semantic_model_6",
    "semantic_model_6_no_leakage"
]


# ---------------------------------------------------------
# Eigene stratifizierte Interaction-Stichprobe
# ---------------------------------------------------------

shap_interaction_sample_indices = (
    get_stratified_shap_sample_indices(
        y=y,
        candidate_indices=test_idx,
        sample_size=SHAP_INTERACTION_CONFIG["sample_size"],
        random_state=SHAP_INTERACTION_CONFIG["random_state"]
    )
)

print("\n=== SHAP-Interaction-Stichprobe ===")
print(
    "Anzahl Fälle:",
    len(shap_interaction_sample_indices)
)
print(
    "Mortalitätsrate Interaction-Stichprobe:",
    y[shap_interaction_sample_indices].mean()
)


# ---------------------------------------------------------
# Interaction Values eines Modells berechnen
# ---------------------------------------------------------

def run_shap_interaction_analysis(
    model_name,
    result,
    X,
    feature_names,
    sample_indices,
    output_directory,
    top_features_heatmap=20,
    top_pairs=25
):
    """
    Berechnet und visualisiert SHAP Interaction Values.

    Ausgaben:
    1. CSV mit allen Feature-Paaren und mittlerer absoluter Interaktion
    2. CSV mit Interaktionsstärke je Feature
    3. CSV mit der vollständigen mittleren absoluten Interaktionsmatrix
    4. Heatmap der wichtigsten Features
    5. Balkendiagramm der wichtigsten Feature-Paare

    Die Diagonale der Interaction-Matrix entspricht den Main Effects.
    Für die Paaranalyse werden nur Off-Diagonal-Elemente verwendet.
    """

    print("\n==================================================")
    print(f"SHAP Interaction Values: {model_name}")
    print("==================================================")

    model = result["final_model"]

    # -----------------------------------------------------
    # Sicherheitschecks
    # -----------------------------------------------------

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{model_name}: X besitzt {X.shape[1]} Features, "
            f"aber es liegen {len(feature_names)} Feature-Namen vor."
        )

    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            f"{model_name}: Modell erwartet "
            f"{model.n_features_in_} Features, "
            f"aber es liegen {len(feature_names)} Namen vor."
        )

    # -----------------------------------------------------
    # Stichprobe vorbereiten
    # -----------------------------------------------------

    X_interaction = prepare_shap_matrix(
        X=X,
        row_indices=sample_indices
    )

    # Für die Interaction-Matrix wird eine dichte Matrix benötigt.
    # Bei Semantik-Modell 5 sollte dies vertretbar sein, weil die
    # direkten Diagnose- und Prozedurcode-Dummies entfernt wurden.
    if sparse.issparse(X_interaction):
        X_interaction_dense = X_interaction.toarray()
    else:
        X_interaction_dense = np.asarray(
            X_interaction
        )

    X_interaction_dense = np.asarray(
        X_interaction_dense,
        dtype=np.float32
    )

    n_samples = X_interaction_dense.shape[0]
    n_features = X_interaction_dense.shape[1]

    estimated_memory_gb = (
        n_samples
        * n_features
        * n_features
        * np.dtype(np.float32).itemsize
        / 1024**3
    )

    print("Interaction-Fälle:", n_samples)
    print("Features:", n_features)
    print(
        "Geschätzter Speicherbedarf der reinen "
        f"Interaction-Matrix: {estimated_memory_gb:.3f} GB"
    )

    # Begrenzung des Speicherbedarfs
    if estimated_memory_gb > 8:
        raise MemoryError(
            f"{model_name}: Die Interaction-Matrix würde ungefähr "
            f"{estimated_memory_gb:.2f} GB benötigen. "
            "Reduziere SHAP_INTERACTION_CONFIG['sample_size']."
        )

    # -----------------------------------------------------
    # SHAP Interaction Values berechnen
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        model,
        feature_names=feature_names
    )

    interaction_values_raw = (
        explainer.shap_interaction_values(
            X_interaction_dense
        )
    )

    # Manche SHAP-Versionen liefern bei Klassifikationsmodellen
    # eine Liste. Bei binärer Klassifikation wird dann die einzige
    # enthaltene Matrix verwendet.
    if isinstance(interaction_values_raw, list):

        if len(interaction_values_raw) != 1:
            raise ValueError(
                f"{model_name}: Unerwartete Anzahl von "
                f"Interaction-Ausgaben: "
                f"{len(interaction_values_raw)}"
            )

        interaction_values_raw = (
            interaction_values_raw[0]
        )

    interaction_values = np.asarray(
        interaction_values_raw,
        dtype=np.float32
    )

    # Neuere SHAP-Versionen können eine zusätzliche
    # Output-Dimension der Länge 1 zurückgeben.
    if (
        interaction_values.ndim == 4
        and interaction_values.shape[-1] == 1
    ):
        interaction_values = (
            interaction_values[..., 0]
        )

    expected_shape = (
        n_samples,
        n_features,
        n_features
    )

    if interaction_values.shape != expected_shape:
        raise ValueError(
            f"{model_name}: Unerwartete Shape der "
            f"Interaction Values: {interaction_values.shape}; "
            f"erwartet wurde {expected_shape}."
        )

    # -----------------------------------------------------
    # Symmetrie prüfen
    # -----------------------------------------------------

    max_symmetry_difference = float(
        np.max(
            np.abs(
                interaction_values
                - np.swapaxes(
                    interaction_values,
                    1,
                    2
                )
            )
        )
    )

    print(
        "Maximale Symmetrieabweichung:",
        max_symmetry_difference
    )

    assert np.allclose(
        interaction_values,
        np.swapaxes(
            interaction_values,
            1,
            2
        ),
        atol=1e-5,
        rtol=1e-5
    ), (
        f"{model_name}: Die SHAP-Interaction-Matrix "
        "ist nicht ausreichend symmetrisch."
    )

    # -----------------------------------------------------
    # Additivitätscheck
    # -----------------------------------------------------
    # Summe über die vollständige Interaction-Matrix
    # plus Expected Value muss dem Modell-Margin entsprechen.
    # -----------------------------------------------------

    n_checks = min(
        10,
        n_samples
    )

    model_margins = np.asarray(
        model.predict(
            X_interaction_dense[:n_checks],
            output_margin=True
        )
    ).reshape(-1)

    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    if len(expected_value) == 1:
        expected_value_scalar = float(
            expected_value[0]
        )
    else:
        raise ValueError(
            f"{model_name}: Unerwartete Form des "
            f"SHAP Expected Value: {expected_value.shape}"
        )

    reconstructed_margins = (
        expected_value_scalar
        + interaction_values[
            :n_checks
        ].sum(axis=(1, 2))
    )

    max_additivity_difference = float(
        np.max(
            np.abs(
                model_margins
                - reconstructed_margins
            )
        )
    )

    print(
        "Maximale Additivitätsabweichung:",
        max_additivity_difference
    )

    assert np.allclose(
        model_margins,
        reconstructed_margins,
        atol=1e-3,
        rtol=1e-5
    ), (
        f"{model_name}: Die Interaction Values rekonstruieren "
        "den Modell-Margin nicht ausreichend genau."
    )

    # -----------------------------------------------------
    # Mittlere absolute Interaction-Matrix
    # -----------------------------------------------------

    mean_abs_interaction_matrix = np.mean(
        np.abs(interaction_values),
        axis=0
    )

    mean_signed_interaction_matrix = np.mean(
        interaction_values,
        axis=0
    )

    # Main Effects befinden sich auf der Diagonale.
    main_effects = np.diag(
        mean_abs_interaction_matrix
    ).copy()

    # Für die reine Interaktionsanalyse Diagonale entfernen.
    mean_abs_off_diagonal = (
        mean_abs_interaction_matrix.copy()
    )

    np.fill_diagonal(
        mean_abs_off_diagonal,
        0
    )

    # -----------------------------------------------------
    # Interaktionsstärke je Feature
    # -----------------------------------------------------
    # Da die Matrix symmetrisch ist, steht jede Interaktion
    # zweimal darin. Für das Ranking ist das unproblematisch.
    # -----------------------------------------------------

    feature_interaction_strength = (
        mean_abs_off_diagonal.sum(axis=1)
    )

    feature_interaction_df = pd.DataFrame({
        "feature_set": model_name,
        "feature": feature_names,
        "mean_abs_main_effect": main_effects,
        "total_mean_abs_interaction": (
            feature_interaction_strength
        )
    })

    feature_interaction_df = (
        feature_interaction_df
        .sort_values(
            "total_mean_abs_interaction",
            ascending=False
        )
        .reset_index(drop=True)
    )

    feature_interaction_df.to_csv(
        os.path.join(
            output_directory,
            f"shap_interaction_features_{model_name}.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # Tabelle aller Feature-Paare
    # -----------------------------------------------------

    pair_rows = []

    for i in range(n_features):
        for j in range(i + 1, n_features):

            pair_rows.append({
                "feature_set": model_name,
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "mean_abs_interaction": (
                    mean_abs_interaction_matrix[i, j]
                ),
                "mean_signed_interaction": (
                    mean_signed_interaction_matrix[i, j]
                )
            })

    interaction_pairs_df = pd.DataFrame(
        pair_rows
    )

    interaction_pairs_df = (
        interaction_pairs_df
        .sort_values(
            "mean_abs_interaction",
            ascending=False
        )
        .reset_index(drop=True)
    )

    interaction_pairs_df.to_csv(
        os.path.join(
            output_directory,
            f"shap_interaction_pairs_{model_name}.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # Vollständige mittlere Interaktionsmatrix exportieren
    # -----------------------------------------------------

    mean_abs_interaction_df = pd.DataFrame(
        mean_abs_interaction_matrix,
        index=feature_names,
        columns=feature_names
    )

    mean_abs_interaction_df.to_csv(
        os.path.join(
            output_directory,
            f"shap_interaction_matrix_{model_name}.csv"
        )
    )

    # -----------------------------------------------------
    # 1. Heatmap der wichtigsten Interaktionsfeatures
    # -----------------------------------------------------

    n_heatmap_features = min(
        top_features_heatmap,
        n_features
    )

    top_feature_indices = np.argsort(
        feature_interaction_strength
    )[
        -n_heatmap_features:
    ]

    # Wichtigstes Feature oben/links darstellen
    top_feature_indices = (
        top_feature_indices[::-1]
    )

    heatmap_matrix = (
        mean_abs_off_diagonal[
            np.ix_(
                top_feature_indices,
                top_feature_indices
            )
        ]
    )

    heatmap_feature_names = [
        feature_names[i]
        for i in top_feature_indices
    ]

    figure_size = max(
        10,
        0.55 * n_heatmap_features
    )

    plt.figure(
        figsize=(
            figure_size,
            figure_size
        )
    )

    image = plt.imshow(
        heatmap_matrix,
        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Mean absolute SHAP interaction value"
    )

    plt.xticks(
        ticks=np.arange(
            n_heatmap_features
        ),
        labels=heatmap_feature_names,
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        ticks=np.arange(
            n_heatmap_features
        ),
        labels=heatmap_feature_names,
        fontsize=8
    )

    plt.title(
        f"SHAP interaction heatmap – {model_name}\n"
        "Diagonal excluded"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"shap_interaction_heatmap_{model_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------
    # 2. Balkendiagramm der stärksten Feature-Paare
    # -----------------------------------------------------

    top_pairs_df = (
        interaction_pairs_df
        .head(top_pairs)
        .copy()
    )

    top_pairs_df["feature_pair"] = (
        top_pairs_df["feature_1"]
        + " × "
        + top_pairs_df["feature_2"]
    )

    top_pairs_plot_df = (
        top_pairs_df
        .sort_values(
            "mean_abs_interaction",
            ascending=True
        )
    )

    plt.figure(
        figsize=(12, 10)
    )

    plt.barh(
        top_pairs_plot_df["feature_pair"],
        top_pairs_plot_df[
            "mean_abs_interaction"
        ]
    )

    plt.xlabel(
        "Mean absolute SHAP interaction value"
    )

    plt.ylabel(
        "Feature pair"
    )

    plt.title(
        f"Top {len(top_pairs_plot_df)} "
        f"SHAP interactions – {model_name}"
    )

    plt.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"shap_interaction_top_pairs_{model_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"{model_name}: SHAP Interaction Values abgeschlossen."
    )

    # Die vollständigen fallbezogenen Interaction Values
    # werden nicht als CSV gespeichert, da sie sehr groß sind.
    # Sie bleiben jedoch im Ergebnis-Dictionary verfügbar.
    return {
        "model_name": model_name,
        "sample_indices": np.asarray(
            sample_indices
        ),
        "interaction_values": interaction_values,
        "mean_abs_interaction_matrix": (
            mean_abs_interaction_matrix
        ),
        "mean_signed_interaction_matrix": (
            mean_signed_interaction_matrix
        ),
        "feature_interaction_df": (
            feature_interaction_df
        ),
        "interaction_pairs_df": (
            interaction_pairs_df
        ),
        "estimated_memory_gb": (
            estimated_memory_gb
        )
    }


# ---------------------------------------------------------
# Interaction-Analyse ausführen
# ---------------------------------------------------------

shap_interaction_results = {}

for model_name in shap_interaction_model_names:

    shap_interaction_results[model_name] = (
        run_shap_interaction_analysis(
            model_name=model_name,
            result=results_dict[model_name],
            X=shap_feature_sets[model_name],
            feature_names=feature_names_dict[
                model_name
            ],
            sample_indices=(
                shap_interaction_sample_indices
            ),
            output_directory=shap_output_dir,
            top_features_heatmap=(
                SHAP_INTERACTION_CONFIG[
                    "top_features_heatmap"
                ]
            ),
            top_pairs=(
                SHAP_INTERACTION_CONFIG[
                    "top_pairs"
                ]
            )
        )
    )

    gc.collect()


# ---------------------------------------------------------
# Interaction-Ergebnisse beider Modelle zusammenführen
# ---------------------------------------------------------

all_shap_interaction_pairs = pd.concat(
    [
        result["interaction_pairs_df"]
        for result in (
            shap_interaction_results.values()
        )
    ],
    ignore_index=True
)

all_shap_interaction_features = pd.concat(
    [
        result["feature_interaction_df"]
        for result in (
            shap_interaction_results.values()
        )
    ],
    ignore_index=True
)

all_shap_interaction_pairs.to_csv(
    os.path.join(
        shap_output_dir,
        "all_models_shap_interaction_pairs.csv"
    ),
    index=False
)

all_shap_interaction_features.to_csv(
    os.path.join(
        shap_output_dir,
        "all_models_shap_interaction_features.csv"
    ),
    index=False
)

print(
    "\nSHAP-Interaction-Analyse abgeschlossen."
)








# =========================================================
# Lokale SHAP-Fallanalyse
# =========================================================

# ---------------------------------------------------------
# Modelle für die lokale Analyse auswählen
# ---------------------------------------------------------

local_shap_model_names = [
    "baseline_with_icd10",
    "semantic_model_5",
    "semantic_model_6"
]

baseline_local_name = "baseline_with_icd10"

# Semantik-Modell, anhand dessen die kontrastierenden Fälle
# gegenüber der Baseline ausgewählt werden
semantic_local_name = "semantic_model_6"

# ---------------------------------------------------------
# Falltyp für jede Testposition bestimmen
# ---------------------------------------------------------

def get_case_type_array(
    result,
    y,
    test_idx
):
    """
    Bestimmt für jede Position im Testset den Klassifikationstyp:
    true_positive, false_negative, false_positive oder true_negative.
    """

    y_test_local = np.asarray(y[test_idx])
    proba = np.asarray(result["test_proba"])
    threshold = float(result["best_threshold"])

    if len(proba) != len(test_idx):
        raise ValueError(
            "Die Anzahl der Testwahrscheinlichkeiten stimmt nicht "
            "mit der Länge von test_idx überein."
        )

    prediction = (
        proba >= threshold
    ).astype(int)

    case_types = np.empty(
        len(test_idx),
        dtype=object
    )

    case_types[
        (y_test_local == 1)
        & (prediction == 1)
    ] = "true_positive"

    case_types[
        (y_test_local == 1)
        & (prediction == 0)
    ] = "false_negative"

    case_types[
        (y_test_local == 0)
        & (prediction == 1)
    ] = "false_positive"

    case_types[
        (y_test_local == 0)
        & (prediction == 0)
    ] = "true_negative"

    return case_types


case_types_by_model = {
    model_name: get_case_type_array(
        result=results_dict[model_name],
        y=y,
        test_idx=test_idx
    )
    for model_name in local_shap_model_names
}

# ---------------------------------------------------------
# Repräsentativen Fall aus einer Kandidatengruppe auswählen
# ---------------------------------------------------------

def select_representative_position(
    candidate_positions,
    results_dict,
    model_names
):
    """
    Wählt aus den Kandidaten den Fall, dessen vorhergesagte
    Wahrscheinlichkeiten bei den betrachteten Modellen möglichst nahe
    an den jeweiligen Gruppenmedianen liegen.

    Die Distanz wird durch den Interquartilsabstand standardisiert.
    """

    candidate_positions = np.asarray(
        candidate_positions,
        dtype=int
    )

    if len(candidate_positions) == 0:
        return None, None

    total_distance = np.zeros(
        len(candidate_positions),
        dtype=float
    )

    median_probabilities = {}

    for model_name in model_names:

        probabilities = np.asarray(
            results_dict[model_name]["test_proba"]
        )[candidate_positions]

        median_probability = float(
            np.median(probabilities)
        )

        q25, q75 = np.quantile(
            probabilities,
            [0.25, 0.75]
        )

        iqr = float(q75 - q25)

        if iqr > 0:
            scale = iqr
        else:
            scale = float(
                np.std(probabilities)
            )

        if scale <= 0:
            scale = 1.0

        total_distance += (
            np.abs(
                probabilities
                - median_probability
            )
            / scale
        )

        median_probabilities[
            model_name
        ] = median_probability

    best_local_index = int(
        np.argmin(total_distance)
    )

    best_test_position = int(
        candidate_positions[
            best_local_index
        ]
    )

    return (
        best_test_position,
        {
            "selection_distance": float(
                total_distance[
                    best_local_index
                ]
            ),
            "median_probabilities": median_probabilities
        }
    )

# ---------------------------------------------------------
# Drei relevante Fallkonstellationen auswählen
# ---------------------------------------------------------

def select_local_comparison_cases(
    results_dict,
    case_types_by_model,
    local_model_names,
    baseline_name,
    semantic_name,
    y,
    test_idx
):
    """
    Wählt drei lokale Vergleichsfälle:

    1. Gemeinsamer repräsentativer TP aller Modelle;
       falls nicht vorhanden: gemeinsamer TN.

    2. Semantik-Modell 6 korrekt, Baseline falsch.
       - Semantik TP, Baseline FN
       - sonst Semantik TN, Baseline FP

    3. Baseline korrekt, Semantik-Modell 6 falsch.
       - Baseline TP, Semantik FN
       - sonst Baseline TN, Semantik FP
    """

    test_positions = np.arange(len(test_idx))

    selected_rows = []
    candidate_count_rows = []

    # =====================================================
    # Fall 1: Gemeinsame korrekte Entscheidung
    # =====================================================

    common_choice = None

    for preferred_case_type in [
        "true_positive",
        "true_negative"
    ]:

        common_mask = np.ones(len(test_idx), dtype=bool)

        for model_name in local_model_names:
            common_mask &= (
                case_types_by_model[
                    model_name
                ]
                == preferred_case_type
            )

        candidate_positions = test_positions[common_mask]

        candidate_count_rows.append({
            "scenario": (
                f"common_{preferred_case_type}"
            ),
            "n_candidates": len(
                candidate_positions
            )
        })

        if len(candidate_positions) > 0:

            selected_position, selection_info = (
                select_representative_position(
                    candidate_positions=(
                        candidate_positions
                    ),
                    results_dict=results_dict,
                    model_names=local_model_names
                )
            )

            common_choice = {
                "scenario": ("common_correct_case"),
                "scenario_description": (
                    f"Alle Modelle: "
                    f"{preferred_case_type}"
                ),
                "test_position": (selected_position),
                "n_candidates": len(candidate_positions),
                "selection_distance": (
                    selection_info[ "selection_distance"]
                ),
                "expected_baseline_type": (preferred_case_type),
                "expected_semantic_type": (preferred_case_type)
            }

            break

    if common_choice is not None:
        selected_rows.append(common_choice)
    else:
        print(
            "WARNUNG: Es gibt weder einen gemeinsamen TP "
            "noch einen gemeinsamen TN."
        )

    # =====================================================
    # Fall 2: Semantik korrekt, Baseline falsch
    # =====================================================

    semantic_better_patterns = [
        {
            "baseline_type": "false_negative",
            "semantic_type": "true_positive",
            "description": (
                "Semantik-Modell 6 TP, Baseline FN"
            )
        },
        {
            "baseline_type": "false_positive",
            "semantic_type": "true_negative",
            "description": (
                "Semantik-Modell 6 TN, Baseline FP"
            )
        }
    ]

    semantic_better_choice = None

    for pattern in semantic_better_patterns:

        mask = (
            case_types_by_model[
                baseline_name
            ]
            == pattern["baseline_type"]
        ) & (
            case_types_by_model[
                semantic_name
            ]
            == pattern["semantic_type"]
        )

        candidate_positions = test_positions[
            mask
        ]

        candidate_count_rows.append({
            "scenario": (
                "semantic_correct_baseline_wrong_"
                + pattern["description"]
            ),
            "n_candidates": len(
                candidate_positions
            )
        })

        if len(candidate_positions) > 0:

            selected_position, selection_info = (
                select_representative_position(
                    candidate_positions=(
                        candidate_positions
                    ),
                    results_dict=results_dict,
                    model_names=[
                        baseline_name,
                        semantic_name
                    ]
                )
            )

            semantic_better_choice = {
                "scenario": (
                    "semantic_correct_"
                    "baseline_wrong"
                ),
                "scenario_description": (
                    pattern["description"]
                ),
                "test_position": (
                    selected_position
                ),
                "n_candidates": len(
                    candidate_positions
                ),
                "selection_distance": (
                    selection_info[
                        "selection_distance"
                    ]
                ),
                "expected_baseline_type": (
                    pattern["baseline_type"]
                ),
                "expected_semantic_type": (
                    pattern["semantic_type"]
                )
            }

            break

    if semantic_better_choice is not None:
        selected_rows.append(
            semantic_better_choice
        )
    else:
        print(
            "WARNUNG: Kein Fall gefunden, bei dem "
            f"{semantic_name} korrekt und die Baseline "
            "falsch klassifiziert."
        )

    # =====================================================
    # Fall 3: Baseline korrekt, Semantik falsch
    # =====================================================

    baseline_better_patterns = [
        {
            "baseline_type": "true_positive",
            "semantic_type": "false_negative",
            "description": (
                "Baseline TP, Semantik-Modell 6 FN"
            )
        },
        {
            "baseline_type": "true_negative",
            "semantic_type": "false_positive",
            "description": (
                "Baseline TN, Semantik-Modell 6 FP"
            )
        }
    ]

    baseline_better_choice = None

    for pattern in baseline_better_patterns:

        mask = (
            case_types_by_model[
                baseline_name
            ]
            == pattern["baseline_type"]
        ) & (
            case_types_by_model[
                semantic_name
            ]
            == pattern["semantic_type"]
        )

        candidate_positions = test_positions[
            mask
        ]

        candidate_count_rows.append({
            "scenario": (
                "baseline_correct_semantic_wrong_"
                + pattern["description"]
            ),
            "n_candidates": len(
                candidate_positions
            )
        })

        if len(candidate_positions) > 0:

            selected_position, selection_info = (
                select_representative_position(
                    candidate_positions=(
                        candidate_positions
                    ),
                    results_dict=results_dict,
                    model_names=[
                        baseline_name,
                        semantic_name
                    ]
                )
            )

            baseline_better_choice = {
                "scenario": (
                    "baseline_correct_"
                    "semantic_wrong"
                ),
                "scenario_description": (
                    pattern["description"]
                ),
                "test_position": (
                    selected_position
                ),
                "n_candidates": len(
                    candidate_positions
                ),
                "selection_distance": (
                    selection_info[
                        "selection_distance"
                    ]
                ),
                "expected_baseline_type": (
                    pattern["baseline_type"]
                ),
                "expected_semantic_type": (
                    pattern["semantic_type"]
                )
            }

            break

    if baseline_better_choice is not None:
        selected_rows.append(
            baseline_better_choice
        )
    else:
        print(
            "WARNUNG: Kein Fall gefunden, bei dem "
            f"die Baseline korrekt und {semantic_name} "
            "falsch klassifiziert."
        )

    selected_cases_df = pd.DataFrame(
        selected_rows
    )

    candidate_counts_df = pd.DataFrame(
        candidate_count_rows
    )

    if not selected_cases_df.empty:

        selected_cases_df[
            "row_index"
        ] = selected_cases_df[
            "test_position"
        ].apply(
            lambda position: int(
                test_idx[int(position)]
            )
        )

        selected_cases_df[
            "hadm_id"
        ] = selected_cases_df[
            "row_index"
        ].apply(
            lambda row_index: hadm_id[
                int(row_index)
            ]
        )

        selected_cases_df[
            "actual_outcome"
        ] = selected_cases_df[
            "row_index"
        ].apply(
            lambda row_index: int(
                y[int(row_index)]
            )
        )

        for model_name in local_model_names:

            selected_cases_df[
                f"case_type_{model_name}"
            ] = selected_cases_df[
                "test_position"
            ].apply(
                lambda position: (
                    case_types_by_model[
                        model_name
                    ][int(position)]
                )
            )

            selected_cases_df[
                f"probability_{model_name}"
            ] = selected_cases_df[
                "test_position"
            ].apply(
                lambda position: float(
                    results_dict[
                        model_name
                    ]["test_proba"][
                        int(position)
                    ]
                )
            )

    return (
        selected_cases_df,
        candidate_counts_df
    )


(
    selected_local_cases_df,
    local_candidate_counts_df
) = select_local_comparison_cases(
    results_dict=results_dict,
    case_types_by_model=case_types_by_model,
    local_model_names=local_shap_model_names,
    baseline_name=baseline_local_name,
    semantic_name=semantic_local_name,
    y=y,
    test_idx=test_idx
)


print("\n=== Anzahl möglicher lokaler Vergleichsfälle ===")
print(local_candidate_counts_df)

print("\n=== Ausgewählte lokale Vergleichsfälle ===")
print(selected_local_cases_df)


selected_local_cases_df.to_csv(
    os.path.join(
        shap_output_dir,
        "selected_local_comparison_cases.csv"
    ),
    index=False
)

# ---------------------------------------------------------
# Wasserfall-Plots für die drei ausgewählten Fälle
# ---------------------------------------------------------

def create_local_shap_comparison_plots(
    model_name,
    result,
    X,
    feature_names,
    selected_cases_df,
    y,
    test_idx,
    hadm_id,
    baseline_name,
    semantic_name,
    output_directory,
    max_display=20
):
    model = result["final_model"]

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{model_name}: Anzahl der Features und "
            "Feature-Namen stimmt nicht überein."
        )

    explainer = shap.TreeExplainer(
        model,
        feature_names=feature_names
    )

    case_rows = []

    for _, selected_case in (
        selected_cases_df.iterrows()
    ):

        scenario = str(
            selected_case["scenario"]
        )

        scenario_description = str(
            selected_case[
                "scenario_description"
            ]
        )

        test_position = int(
            selected_case[
                "test_position"
            ]
        )

        original_row_index = int(
            test_idx[test_position]
        )

        X_case_model = get_subset(
            X,
            np.array([
                original_row_index
            ])
        )
        
        if sparse.issparse(X_case_model):
            X_case_model = X_case_model.astype(
                np.float32
            ).tocsr()
        else:
            X_case_model = np.asarray(
                X_case_model,
                dtype=np.float32
            )
        
        shap_case_raw = explainer(
            X_case_model,
            check_additivity=False
        )
        
        # Nur für die Darstellung explizite Featurewerte erzeugen.
        # Die SHAP-Werte selbst wurden vorher auf der korrekten Eingabe berechnet.
        if sparse.issparse(X_case_model):
            X_case_plot_data = X_case_model.toarray()
        else:
            X_case_plot_data = np.asarray(
                X_case_model
            )
        
        shap_case = shap.Explanation(
            values=shap_case_raw.values,
            base_values=shap_case_raw.base_values,
            data=X_case_plot_data,
            feature_names=list(feature_names)
        )
        
        actual_outcome = int(
            y[original_row_index]
        )

        admission_id = (
            hadm_id[
                original_row_index
            ]
        )

        stored_probability = float(
            result["test_proba"][
                test_position
            ]
        )

        model_threshold = float(
            result["best_threshold"]
        )

        predicted_class = int(
            stored_probability
            >= model_threshold
        )

        if (
            actual_outcome == 1
            and predicted_class == 1
        ):
            calculated_case_type = (
                "true_positive"
            )

        elif (
            actual_outcome == 1
            and predicted_class == 0
        ):
            calculated_case_type = (
                "false_negative"
            )

        elif (
            actual_outcome == 0
            and predicted_class == 1
        ):
            calculated_case_type = (
                "false_positive"
            )

        else:
            calculated_case_type = (
                "true_negative"
            )

        expected_case_type = str(
            selected_case[
                f"case_type_{model_name}"
            ]
        )

        assert (
            calculated_case_type
            == expected_case_type
        ), (
            f"Falltyp stimmt bei {model_name} nicht: "
            f"erwartet {expected_case_type}, "
            f"berechnet {calculated_case_type}."
        )

        # ---------------------------------------------
        # Spezifische Plausibilitätschecks der Szenarien
        # ---------------------------------------------

        if (
            scenario
            == "semantic_correct_baseline_wrong"
        ):

            baseline_type = str(
                selected_case[
                    f"case_type_{baseline_name}"
                ]
            )

            semantic_type = str(
                selected_case[
                    f"case_type_{semantic_name}"
                ]
            )

            assert baseline_type in [
                "false_negative",
                "false_positive"
            ]

            assert semantic_type in [
                "true_positive",
                "true_negative"
            ]

        elif (
            scenario
            == "baseline_correct_semantic_wrong"
        ):

            baseline_type = str(
                selected_case[
                    f"case_type_{baseline_name}"
                ]
            )

            semantic_type = str(
                selected_case[
                    f"case_type_{semantic_name}"
                ]
            )

            assert baseline_type in [
                "true_positive",
                "true_negative"
            ]

            assert semantic_type in [
                "false_negative",
                "false_positive"
            ]

        # ---------------------------------------------
        # SHAP-Konsistenzcheck
        # ---------------------------------------------

        proba_direct = float(
            model.predict_proba(
                X_case_model
            )[0, 1]
        )

        margin_direct = float(
            model.predict(
                X_case_model,
                output_margin=True
            )[0]
        )

        shap_reconstructed_output = float(
            np.asarray(
                shap_case.base_values
            ).reshape(-1)[0]
            + shap_case.values[0].sum()
        )

        proba_from_margin = float(
            expit(margin_direct)
        )

        proba_from_shap = float(
            expit(
                shap_reconstructed_output
            )
        )

        score_difference = abs(
            proba_direct
            - stored_probability
        )
        
        print(
            f"{model_name} | {scenario} | "
            f"gespeicherter Score={stored_probability:.10f} | "
            f"direkter Score={proba_direct:.10f} | "
            f"Differenz={score_difference:.10e}"
        )

        assert np.isclose(
            proba_direct,
            stored_probability,
            atol=1e-6
        ), (
            "Gespeicherte und direkt berechnete "
            "Wahrscheinlichkeit stimmen nicht überein."
        )

        assert np.isclose(
            margin_direct,
            shap_reconstructed_output,
            atol=1e-3
        ), (
            "SHAP-Werte rekonstruieren nicht den "
            "rohen Modelloutput."
        )

        assert np.isclose(
            proba_direct,
            proba_from_margin,
            atol=1e-6
        ), (
            "predict_proba und Margin sind inkonsistent."
        )

        assert np.isclose(
            proba_direct,
            proba_from_shap,
            atol=1e-5
        ), (
            "Die aus den SHAP-Werten rekonstruierte "
            "Wahrscheinlichkeit stimmt nicht."
        )

        # ---------------------------------------------
        # Waterfall-Plot
        # ---------------------------------------------

        plt.figure()

        shap.plots.waterfall(
            shap_case[0],
            max_display=max_display,
            show=False
        )

        plt.title(
            f"{model_name}: "
            f"{calculated_case_type}\n"
            f"{scenario_description}\n"
            f"Outcome={actual_outcome}, "
            f"score={proba_direct:.3f}, "
            f"threshold={model_threshold:.3f}, "
            f"margin={margin_direct:.3f}, "
            f"hadm_id={admission_id}"
        )

        plt.tight_layout()

        plot_path = os.path.join(
            output_directory,
            f"shap_local_{scenario}_"
            f"{model_name}.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        case_rows.append({
            "scenario": scenario,
            "scenario_description": scenario_description,
            "feature_set": model_name,
            "row_index": original_row_index,
            "test_position": test_position,
            "hadm_id": admission_id,
            "actual_outcome": actual_outcome,
            "case_type": calculated_case_type,
            "predicted_class": predicted_class,
            "predicted_probability": stored_probability,
            "predicted_probability_direct": proba_direct,
            "threshold": model_threshold,
            "output_margin": margin_direct,
            "shap_reconstructed_output": shap_reconstructed_output,
            "probability_from_shap": proba_from_shap,
            "plot_path": plot_path
        })

    return pd.DataFrame(
        case_rows
    )


# ---------------------------------------------------------
# Plots für alle drei Modelle erzeugen
# ---------------------------------------------------------

local_shap_case_dfs = []

for model_name in local_shap_model_names:

    local_case_df = (
        create_local_shap_comparison_plots(
            model_name=model_name,
            result=results_dict[
                model_name
            ],
            X=selected_feature_sets[
                model_name
            ],
            feature_names=feature_names_dict[
                model_name
            ],
            selected_cases_df=(
                selected_local_cases_df
            ),
            y=y,
            test_idx=test_idx,
            hadm_id=hadm_id,
            baseline_name=(
                baseline_local_name
            ),
            semantic_name=(
                semantic_local_name
            ),
            output_directory=(
                shap_output_dir
            ),
            max_display=20
        )
    )

    local_shap_case_dfs.append(
        local_case_df
    )


if local_shap_case_dfs:

    all_local_shap_cases = pd.concat(
        local_shap_case_dfs,
        ignore_index=True
    )

else:
    all_local_shap_cases = pd.DataFrame()


all_local_shap_cases.to_csv(
    os.path.join(
        shap_output_dir,
        "local_shap_comparison_results.csv"
    ),
    index=False
)















# =========================================================
# Feature-Gruppen-basierte Permutation Importance
# =========================================================
# Untersucht, wie stark sich die Testperformance verschlechtert,
# wenn alle Features einer Merkmalsgruppe gemeinsam permutiert werden.
#
# Wichtig:
# Innerhalb einer Feature-Gruppe wird für alle Spalten dieselbe
# Zeilenpermutation verwendet. Dadurch bleiben die Beziehungen zwischen
# den Features derselben Gruppe erhalten, während ihre Zuordnung zu
# Outcome und anderen Merkmalsgruppen zerstört wird.

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------

GROUP_PERMUTATION_CONFIG = {
    "n_repeats": 10,
    "random_state": 42
}

group_permutation_output_dir = os.path.join(
    output_dir,
    "group_permutation_importance"
)

os.makedirs(
    group_permutation_output_dir,
    exist_ok=True
)

# ---------------------------------------------------------
# Hilfsfunktion: Bewertungsmetrik
# ---------------------------------------------------------

def calculate_group_permutation_score(
    model,
    X,
    y_true,
    scoring="average_precision"
):
    """
    Berechnet die gewünschte diskriminative Modellmetrik.

    Unterstützte Werte:
    - average_precision
    - roc_auc
    - pr_auc
    """

    y_proba = model.predict_proba(X)[:, 1]

    if scoring == "average_precision":
        return average_precision_score(
            y_true,
            y_proba
        )

    elif scoring == "roc_auc":
        return roc_auc_score(
            y_true,
            y_proba
        )

    elif scoring == "pr_auc":
        return calculate_pr_auc(
            y_true,
            y_proba
        )

    else:
        raise ValueError(
            "scoring muss 'average_precision', "
            "'roc_auc' oder 'pr_auc' sein."
        )

# ---------------------------------------------------------
# Feature-Gruppen bestimmen
# ---------------------------------------------------------

def create_feature_group_indices(
    feature_names,
    proc_metric_feature_names=None
):
    """
    Ordnet alle Features mithilfe der bereits definierten Funktion
    assign_feature_group() einer Gruppe zu.

    Rückgabe:
        Dictionary:
        {
            "diagnosis_code": [Spaltenindizes],
            "procedure_code": [Spaltenindizes],
            ...
        }
    """

    if proc_metric_feature_names is None:
        proc_metric_feature_names = []

    group_indices = {}

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        feature_group = assign_feature_group(
            feature=feature_name,
            proc_metric_feature_names=(
                proc_metric_feature_names
            )
        )

        group_indices.setdefault(
            feature_group,
            []
        ).append(
            feature_index
        )

    return group_indices


# ---------------------------------------------------------
# Eine Feature-Gruppe gemeinsam permutieren
# ---------------------------------------------------------

def permute_feature_group(
    X,
    feature_indices,
    row_permutation
):
    """
    Permutiert alle Spalten einer Feature-Gruppe mit derselben
    Zeilenreihenfolge.

    Dadurch bleiben die Beziehungen der Features innerhalb der Gruppe
    weitgehend erhalten. Lediglich die Zuordnung der gesamten Gruppe
    zu den jeweiligen Aufenthalten wird zerstört.
    """

    feature_indices = np.asarray(
        feature_indices,
        dtype=int
    )

    row_permutation = np.asarray(
        row_permutation,
        dtype=int
    )

    if sparse.issparse(X):

        # CSC eignet sich besser für spaltenweise Änderungen
        X_original_csc = X.tocsc()

        # Die zu permutierende Gruppe aus der unveränderten Matrix lesen
        permuted_group = X_original_csc[
            row_permutation,
            :
        ][
            :,
            feature_indices
        ]

        X_permuted = X_original_csc.copy()

        # Gesamten Block ersetzen
        X_permuted[
            :,
            feature_indices
        ] = permuted_group

        # Für XGBoost wieder CSR verwenden
        X_permuted = X_permuted.tocsr()

        X_permuted.eliminate_zeros()

        return X_permuted

    else:

        X_permuted = np.asarray(
            X
        ).copy()

        X_permuted[
            :,
            feature_indices
        ] = X[
            row_permutation
        ][
            :,
            feature_indices
        ]

        return X_permuted

# ---------------------------------------------------------
# Gruppenbasierte Permutation Importance für ein Modell
# ---------------------------------------------------------

def permutation_importance_by_feature_group(
    result,
    X_test,
    y_test,
    feature_names,
    model_name,
    proc_metric_feature_names=None,
    scoring="average_precision",
    n_repeats=10,
    random_state=42
):
    """
    Berechnet eine gruppenbasierte Permutation Importance.

    Für jede Feature-Gruppe wird:
    1. dieselbe zufällige Zeilenpermutation auf alle Gruppenfeatures
       angewendet,
    2. die Modellperformance erneut berechnet,
    3. der Performanceverlust gegenüber der unveränderten Matrix
       bestimmt.

    score_drop = baseline_score - permuted_score

    Positive Werte:
        Die Modellperformance sinkt nach Permutation.
        Die Gruppe enthält relevante Modellinformation.

    Wert nahe 0:
        Die Gruppe liefert wenig zusätzliche Information oder ihre
        Information wird durch andere Gruppen ersetzt.

    Negative Werte:
        Das Modell schneidet nach der Permutation zufällig besser ab.
        Dies kann durch Stichprobenvariation, Redundanz oder Overfitting
        entstehen.
    """

    print("\n==================================================")
    print(
        f"Feature-Gruppen-Permutation: {model_name}"
    )
    print("==================================================")

    model = result["final_model"]

    # -----------------------------------------------------
    # Sicherheitschecks
    # -----------------------------------------------------

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            f"{model_name}: X_test besitzt "
            f"{X_test.shape[0]} Zeilen, y_test aber "
            f"{len(y_test)} Einträge."
        )

    if X_test.shape[1] != len(feature_names):
        raise ValueError(
            f"{model_name}: X_test besitzt "
            f"{X_test.shape[1]} Features, aber es liegen "
            f"{len(feature_names)} Feature-Namen vor."
        )

    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            f"{model_name}: Das trainierte Modell erwartet "
            f"{model.n_features_in_} Features, aber es liegen "
            f"{len(feature_names)} Feature-Namen vor."
        )

    if n_repeats < 1:
        raise ValueError(
            "n_repeats muss mindestens 1 sein."
        )

    # -----------------------------------------------------
    # Feature-Gruppen bilden
    # -----------------------------------------------------

    feature_group_indices = create_feature_group_indices(
        feature_names=feature_names,
        proc_metric_feature_names=(
            proc_metric_feature_names
        )
    )

    print("\nFeature-Gruppen:")

    for feature_group, indices in (
        feature_group_indices.items()
    ):
        print(
            f"{feature_group}: {len(indices)} Features"
        )

    # Features in der Fallback-Gruppe anzeigen
    if "other" in feature_group_indices:

        other_features = [
            feature_names[index]
            for index in feature_group_indices["other"]
        ]

        print("\nWARNUNG: Features in der Gruppe 'other':")
        print(other_features[:100])

        if len(other_features) > 100:
            print(
                f"... und {len(other_features) - 100} weitere."
            )

    # -----------------------------------------------------
    # Ausgangsperformance
    # -----------------------------------------------------

    baseline_score = calculate_group_permutation_score(
        model=model,
        X=X_test,
        y_true=y_test,
        scoring=scoring
    )

    print(
        f"\nBaseline-{scoring}: "
        f"{baseline_score:.6f}"
    )

    # -----------------------------------------------------
    # Permutationen
    # -----------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    repeat_rows = []

    n_test_cases = X_test.shape[0]

    for feature_group, feature_indices in (
        feature_group_indices.items()
    ):

        print(
            f"\nPermutiere Gruppe: {feature_group} "
            f"({len(feature_indices)} Features)"
        )

        for repeat in range(n_repeats):

            row_permutation = rng.permutation(
                n_test_cases
            )

            X_permuted = permute_feature_group(
                X=X_test,
                feature_indices=feature_indices,
                row_permutation=row_permutation
            )

            permuted_score = (
                calculate_group_permutation_score(
                    model=model,
                    X=X_permuted,
                    y_true=y_test,
                    scoring=scoring
                )
            )

            score_drop = (
                baseline_score
                - permuted_score
            )

            if baseline_score != 0:
                relative_score_drop = (
                    score_drop
                    / baseline_score
                )
            else:
                relative_score_drop = np.nan

            repeat_rows.append({
                "model": model_name,
                "feature_group": feature_group,
                "repeat": repeat + 1,
                "n_features": len(feature_indices),
                "scoring": scoring,
                "baseline_score": baseline_score,
                "permuted_score": permuted_score,
                "score_drop": score_drop,
                "relative_score_drop": (
                    relative_score_drop
                )
            })

            print(
                f"  Wiederholung {repeat + 1:02d}: "
                f"Score={permuted_score:.6f}, "
                f"Drop={score_drop:.6f}"
            )

            del X_permuted
            gc.collect()

    repeat_df = pd.DataFrame(
        repeat_rows
    )

    # -----------------------------------------------------
    # Ergebnisse je Feature-Gruppe zusammenfassen
    # -----------------------------------------------------

    summary_df = (
        repeat_df
        .groupby(
            [
                "model",
                "feature_group",
                "n_features",
                "scoring"
            ],
            as_index=False
        )
        .agg(
            baseline_score=(
                "baseline_score",
                "first"
            ),
            mean_permuted_score=(
                "permuted_score",
                "mean"
            ),
            std_permuted_score=(
                "permuted_score",
                "std"
            ),
            mean_score_drop=(
                "score_drop",
                "mean"
            ),
            std_score_drop=(
                "score_drop",
                "std"
            ),
            median_score_drop=(
                "score_drop",
                "median"
            ),
            min_score_drop=(
                "score_drop",
                "min"
            ),
            max_score_drop=(
                "score_drop",
                "max"
            ),
            ci_lower_score_drop=(
                "score_drop",
                lambda values: values.quantile(
                    0.025
                )
            ),
            ci_upper_score_drop=(
                "score_drop",
                lambda values: values.quantile(
                    0.975
                )
            ),
            mean_relative_score_drop=(
                "relative_score_drop",
                "mean"
            ),
            n_repeats=(
                "repeat",
                "count"
            )
        )
    )

    summary_df = (
        summary_df
        .sort_values(
            "mean_score_drop",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Anteil am gesamten positiven Performanceverlust.
    # Dies ist nur eine deskriptive Normierung und keine additive
    # Zerlegung der Modellperformance.
    positive_score_drops = (
        summary_df["mean_score_drop"]
        .clip(lower=0)
    )

    total_positive_score_drop = (
        positive_score_drops.sum()
    )

    if total_positive_score_drop > 0:
        summary_df[
            "relative_positive_importance"
        ] = (
            positive_score_drops
            / total_positive_score_drop
        )
    else:
        summary_df[
            "relative_positive_importance"
        ] = 0.0

    print("\nZusammenfassung:")
    print(
        summary_df[
            [
                "feature_group",
                "n_features",
                "baseline_score",
                "mean_permuted_score",
                "mean_score_drop",
                "std_score_drop",
                "mean_relative_score_drop"
            ]
        ]
    )

    return repeat_df, summary_df


# ---------------------------------------------------------
# Zu untersuchende Modelle auswählen
# ---------------------------------------------------------

group_permutation_model_names = [
    # Hauptansatz
    "semantic_model_5",
    "semantic_model_6",

    # Ohne Leakage
    "semantic_model_5_no_leakage",
    "semantic_model_6_no_leakage"
]

# ---------------------------------------------------------
# Analysen ausführen
# ---------------------------------------------------------

group_permutation_repeat_results = {}
group_permutation_summary_results = {}

for model_name in group_permutation_model_names:

    if model_name not in results_dict:
        print(
            f"Übersprungen: Kein Modellergebnis für "
            f"{model_name} vorhanden."
        )
        continue

    if model_name not in feature_sets:
        print(
            f"Übersprungen: Kein Feature-Set für "
            f"{model_name} vorhanden."
        )
        continue

    if model_name not in feature_names_dict:
        print(
            f"Übersprungen: Keine Feature-Namen für "
            f"{model_name} vorhanden."
        )
        continue

    X_test_current = get_subset(
        feature_sets[model_name],
        test_idx
    )

    if sparse.issparse(X_test_current):
        X_test_current = (
            X_test_current
            .astype(np.float32)
            .tocsr()
        )
    else:
        X_test_current = np.asarray(
            X_test_current,
            dtype=np.float32
        )

    (
        repeat_df,
        summary_df
    ) = permutation_importance_by_feature_group(
        result=results_dict[model_name],
        X_test=X_test_current,
        y_test=y_test,
        feature_names=(
            feature_names_dict[model_name]
        ),
        model_name=model_name,
        proc_metric_feature_names=(
            get_proc_metric_names_for_feature_set(
                model_name
            )
        ),
        scoring="average_precision",
        **GROUP_PERMUTATION_CONFIG
    )

    group_permutation_repeat_results[
        model_name
    ] = repeat_df

    group_permutation_summary_results[
        model_name
    ] = summary_df

    # Einzelne CSV-Dateien speichern
    repeat_df.to_csv(
        os.path.join(
            group_permutation_output_dir,
            f"group_permutation_repeats_{model_name}.csv"
        ),
        index=False
    )

    summary_df.to_csv(
        os.path.join(
            group_permutation_output_dir,
            f"group_permutation_summary_{model_name}.csv"
        ),
        index=False
    )

    del X_test_current
    gc.collect()

# ---------------------------------------------------------
# Ergebnisse aller Modelle zusammenführen
# ---------------------------------------------------------

if group_permutation_repeat_results:

    all_group_permutation_repeats = pd.concat(
        group_permutation_repeat_results.values(),
        ignore_index=True
    )

else:

    all_group_permutation_repeats = pd.DataFrame()


if group_permutation_summary_results:

    all_group_permutation_summary = pd.concat(
        group_permutation_summary_results.values(),
        ignore_index=True
    )

else:

    all_group_permutation_summary = pd.DataFrame()


all_group_permutation_repeats.to_csv(
    os.path.join(
        group_permutation_output_dir,
        "all_models_group_permutation_repeats.csv"
    ),
    index=False
)

all_group_permutation_summary.to_csv(
    os.path.join(
        group_permutation_output_dir,
        "all_models_group_permutation_summary.csv"
    ),
    index=False
)

# ---------------------------------------------------------
# Grafiken
# ---------------------------------------------------------

def plot_group_permutation_importance(
    summary_df,
    model_name,
    output_directory
):
    """
    Erstellt einen Balkenplot mit dem mittleren Verlust der
    Average Precision und Standardabweichung über die Wiederholungen.
    """

    plot_df = (
        summary_df
        .sort_values(
            "mean_score_drop",
            ascending=True
        )
        .copy()
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature_group"],
        plot_df["mean_score_drop"],
        xerr=plot_df["std_score_drop"].fillna(0),
        capsize=3
    )

    plt.axvline(
        0,
        linewidth=1,
        linestyle="--"
    )

    plt.xlabel(
        "Decrease in average precision after group permutation"
    )

    plt.ylabel(
        "Feature group"
    )

    plt.title(
        f"Permutation importance by feature group – {model_name}"
    )

    plt.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"group_permutation_importance_{model_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


for model_name, summary_df in (
    group_permutation_summary_results.items()
):

    plot_group_permutation_importance(
        summary_df=summary_df,
        model_name=model_name,
        output_directory=(
            group_permutation_output_dir
        )
    )

# ---------------------------------------------------------
# Excel-Export
# ---------------------------------------------------------

output_file_group_permutation = os.path.join(
    group_permutation_output_dir,
    "group_permutation_importance.xlsx"
)

# Kurze Sheet-Namen, da Excel maximal 31 Zeichen erlaubt
excel_sheet_names = {
    "semantic_model_5": "Semantic_5",
    "semantic_model_6": "Semantic_6",
    "semantic_model_5_no_leakage": "Semantic_5_noLeak",
    "semantic_model_6_no_leakage": "Semantic_6_noLeak"
}

with pd.ExcelWriter(
    output_file_group_permutation,
    engine="openpyxl"
) as writer:

    all_group_permutation_summary.to_excel(
        writer,
        sheet_name="All_models_summary",
        index=False
    )

    all_group_permutation_repeats.to_excel(
        writer,
        sheet_name="All_repetitions",
        index=False
    )

    for model_name, summary_df in (
        group_permutation_summary_results.items()
    ):

        sheet_name = excel_sheet_names.get(
            model_name,
            model_name[:31]
        )

        summary_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )


print(
    "\nFeature-Gruppen-basierte Permutation "
    "Importance abgeschlossen."
)

print(
    f"Ergebnisse gespeichert unter:\n"
    f"{output_file_group_permutation}"
)

all_group_permutation_summary




