#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NO-LEAKAGE-ANSATZ

Das Skript erzeugt die Datensätze für den No-Leakage-Ansatz unter Ausschluss 
codebasierter Informationen des aktuellen Aufenthalts. Codebasierte Merkmale 
werden ausschließlich aus dem unmittelbar vorherigen Aufenthalt desselben 
Patienten abgeleitet und mit den nicht-codebezogenen Merkmalen des aktuellen 
Aufenthalts kombiniert. Anschließend werden Baseline- und Full-Model-Datensätze 
als Sparse-Matrizen exportiert.
"""

import json
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.preprocessing import MultiLabelBinarizer

# PFAD ANPASSEN!
path = Path("Pfad_zu_zuvor_gespeicherten_Daten")

OUTPUT_BASELINE_MODEL = "baseline_no_leakage_last_stay"
OUTPUT_FULL_MODEL = "full_model_no_leakage_last_stay"

# ============================================================
# 1) Datenbasis laden
# ============================================================

# Aufenthaltsebene aus Hauptanalyse
df = pd.read_pickle(path / "full_model_staylevel_pre_encoding.pkl")

# Finale Analysepopulation aus Hauptanalyse
final_hadm_ids = np.load(path / "final_hadm_ids.npy")

# In Hauptanalyse definierte häufige Codes
with open(path / "frequent_diag_codes.pkl", "rb") as f:
    frequent_diag_codes = pickle.load(f)

with open(path / "frequent_proc_codes.pkl", "rb") as f:
    frequent_proc_codes = pickle.load(f)

print("Geladene Aufenthalte:", len(df))
print("Finale Analysepopulation:", len(final_hadm_ids))
print("Zulässige Diagnosecodes:", len(frequent_diag_codes))
print("Zulässige Behandlungscodes:", len(frequent_proc_codes))

assert df["hadm_id"].is_unique
assert set(final_hadm_ids).issubset(set(df["hadm_id"]))


# ============================================================
# 2) Zusätzliche Behandlungsmetriken laden
# ============================================================

procedure_metrics_path = path / "procedure_metrics.csv"

if not procedure_metrics_path.exists():
    raise FileNotFoundError(
        f"Behandlungsmetriken nicht gefunden: {procedure_metrics_path}"
    )

proc_metrics = pd.read_csv(procedure_metrics_path)

print("Procedure metrics geladen:")
print(proc_metrics.shape)

assert proc_metrics["hadm_id"].is_unique, (
    "procedure_metrics.csv ist nicht eindeutig auf hadm_id."
)

# ------------------------------------------------------------
# 2.1) PCS-Body-Systeme als Liste rekonstruieren
# ------------------------------------------------------------

proc_body_dummy_cols = [
    c for c in proc_metrics.columns
    if c.startswith("pcs_body_system_")
]

proc_metrics["pcs_body_system_list"] = proc_metrics[
    proc_body_dummy_cols
].apply(
    lambda row: [
        col.replace("pcs_body_system_", "", 1)
        for col in proc_body_dummy_cols
        if row[col] == 1
    ],
    axis=1
)

# Liste separat behalten
proc_body_lists = proc_metrics[
    ["hadm_id", "pcs_body_system_list"]
].copy()

# ------------------------------------------------------------
# 2.2) Numerische Behandlungsmetriken vorbereiten
# ------------------------------------------------------------

proc_metrics_numeric = proc_metrics.drop(
    columns=proc_body_dummy_cols + ["pcs_body_system_list"],
    errors="ignore"
)

proc_metrics_numeric = proc_metrics_numeric.rename(
    columns={
        c: f"proc_{c}"
        for c in proc_metrics_numeric.columns
        if c != "hadm_id"
    }
)

# ------------------------------------------------------------
# 2.3) An Aufenthalt-Datensatz mergen
# ------------------------------------------------------------

df = df.merge(
    proc_metrics_numeric,
    on="hadm_id",
    how="left"
)

df = df.merge(
    proc_body_lists,
    on="hadm_id",
    how="left"
)

# Aufenthalte ohne Prozeduren: keine Body-Systeme
df["pcs_body_system_list"] = df["pcs_body_system_list"].apply(
    lambda x: x if isinstance(x, list) else []
)

# Numerische Behandlungsmetriken bei Aufenthalten ohne Prozeduren = 0
proc_metric_cols = [
    c for c in proc_metrics_numeric.columns
    if c != "hadm_id"
]

# ------------------------------------------------------------
# 2.4) Fehlende Behandlungsmetriken differenziert behandeln
# ------------------------------------------------------------

# has_section_0_procedure ist ein binärer Indikator:
# Aufenthalte ohne Prozedur besitzen ebenfalls keine
# Section-0-Prozedur und erhalten daher 0.

df["proc_has_section_0_procedure"] = (
    df["proc_has_section_0_procedure"]
    .fillna(0)
    .astype(int)
)

# Die übrigen Section-0-basierten Metriken bleiben bewusst NaN,
# wenn keine Section-0-Prozedur vorhanden ist.
# Dadurch bleibt die Unterscheidung erhalten zwischen:
# 0 = tatsächlich beobachteter Nullwert innerhalb Section 0
# NaN = Metrik nicht definiert, da keine Section-0-Prozedur vorliegt

section0_proc_metric_cols = [
    "proc_body_system_diversity",
    "proc_intervention_diversity",
    "proc_invasiveness_share_high",
    "proc_invasiveness_share_medium",
    "proc_invasiveness_share_low",
    "proc_invasiveness_share_non_invasive",
    "proc_n_high_invasive",
]

print(
    "\nNaN-Werte in den Section-0-Behandlungsmetriken:"
)

print(
    df[section0_proc_metric_cols]
    .isna()
    .sum()
)

print("Anzahl Behandlungsmetriken:", len(proc_metric_cols))
print("Anzahl PCS-Body-System-Spalten:", len(proc_body_dummy_cols))


# ============================================================
# 3) Aufenthalte chronologisch sortieren
# ============================================================

for col in ["admittime", "dischtime"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df = df.sort_values(
    ["subject_id", "admittime", "dischtime", "hadm_id"],
    na_position="last"
).reset_index(drop=True)

# Reihenfolge innerhalb jedes Patienten
df["admission_order"] = (
    df.groupby("subject_id").cumcount()
)

df["has_previous_admission"] = (
    df["admission_order"] > 0
).astype(int)


# ============================================================
# 4) Code- und Kategorie-Listen des vorherigen Aufenthalts
# ============================================================

list_shift_map = {
    "diag_code_list": "last_diag_code_list",
    "proc_code_list": "last_proc_code_list",
    "icd10_chapter_list": "last_icd10_chapter_list",
    "pcs_body_system_list": "last_pcs_body_system_list",
}

for source_col, target_col in list_shift_map.items():

    df[target_col] = (
        df.groupby("subject_id", sort=False)[source_col]
          .shift(1)
    )

    # Erstaufenthalte: keine vorherigen Codes
    df[target_col] = df[target_col].apply(
        lambda x: x if isinstance(x, list) else []
    )

# ============================================================
# 5) Count-Merkmale des vorherigen Aufenthalts
# ============================================================

count_shift_map = {
    "n_diag_codes": "last_n_diag_codes",
    "n_proc_codes": "last_n_proc_codes",
    "n_diag_chapters": "last_n_diag_chapters",
}

for source_col, target_col in count_shift_map.items():

    df[target_col] = (
        df.groupby("subject_id", sort=False)[source_col]
          .shift(1)
    )

# ============================================================
# 6) Semantische Merkmale des vorherigen Aufenthalts
# ============================================================

# ------------------------------------------------------------
# 6.1) Diagnosemetriken und IC
# ------------------------------------------------------------

semantic_cols = [
    # Information Content
    "ic_diag_mean",
    "ic_diag_min",
    "ic_diag_max",
    "ic_diag_std",
    "ic_proc_mean",
    "ic_proc_min",
    "ic_proc_max",
    "ic_proc_std",

    # Tiefe
    "depth_mean",
    "depth_std",

    # Leaf
    "leaf_share",
    "leaf_count",

    # Siblings
    "siblings_max",
    "siblings_q25",
    "siblings_q75",
    "siblings_median",
    "siblings_mean",
    "siblings_std",

    # Subtree
    "subtree_rel_log_max",
    "subtree_rel_log_mean",
    "subtree_rel_log_std",
    "subtree_rel_log_internal_mean",

    # Kategorien
    "siblings_3cat_share_high",
    "siblings_3cat_share_low",
    "siblings_3cat_share_medium",

    "depth_ord_share_1.0",
    "depth_ord_share_2.0",
    "depth_ord_share_3.0",
    "depth_ord_share_4.0",
    "depth_ord_share_5.0",
]

# Nur tatsächlich vorhandene Spalten
semantic_cols = [
    c for c in semantic_cols
    if c in df.columns
]

last_semantic_cols = []

for col in semantic_cols:

    new_col = f"last_{col}"

    df[new_col] = (
        df.groupby("subject_id", sort=False)[col]
          .shift(1)
    )

    last_semantic_cols.append(new_col)

# ------------------------------------------------------------
# 6.2) Behandlungsmetriken 
# ------------------------------------------------------------

last_proc_metric_cols = []

for col in proc_metric_cols:

    new_col = f"last_{col}"

    df[new_col] = (
        df.groupby("subject_id", sort=False)[col]
          .shift(1)
    )

    last_proc_metric_cols.append(new_col)
    
    
# ============================================================
# 7) Gleiche Codeauswahl wie in der Hauptanalyse anwenden
# ============================================================

df["last_diag_code_list"] = (
    df["last_diag_code_list"]
    .apply(
        lambda codes: [
            code for code in codes
            if code in frequent_diag_codes
        ]
    )
)

df["last_proc_code_list"] = (
    df["last_proc_code_list"]
    .apply(
        lambda codes: [
            code for code in codes
            if code in frequent_proc_codes
        ]
    )
)


# ============================================================
# 8) Auf identische Analysepopulation beschränken wie im Hauptansatz
# ============================================================

n_before = len(df)

df = df[
    df["hadm_id"].isin(final_hadm_ids)
].copy()

print("Aufenthalte vor Beschränkung:", n_before)
print("Aufenthalte finale Analyse:", len(df))

assert len(df) == len(final_hadm_ids)
assert set(df["hadm_id"]) == set(final_hadm_ids)


# ============================================================
# 9) Aktuelle codebasierte Merkmale und Zeitvariablen entfernen
# ============================================================

current_code_feature_cols = (
    list(list_shift_map.keys())
    + list(count_shift_map.keys())
    + semantic_cols
    + proc_metric_cols
)

df = df.drop(
    columns=current_code_feature_cols,
    errors="ignore"
)

df = df.drop(
    columns=[
        "admittime",
        "dischtime",
        "admission_order",
    ],
    errors="ignore"
)

# Alte kumulative codebezogene historische Features entfernen
old_code_history_cols = [
    "prev_icd10_chapters",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters",
]

df = df.drop(
    columns=old_code_history_cols,
    errors="ignore"
)

# ============================================================
# 10) Erstaufenthalte behandeln
# ============================================================

last_count_cols = [
    "last_n_diag_codes",
    "last_n_proc_codes",
    "last_n_diag_chapters",
]

df[last_count_cols] = (
    df[last_count_cols]
    .fillna(0)
)


# ============================================================
# 11) One-Hot-Encoding der kategorialen Basisvariablen
# ============================================================

df_reduced = pd.get_dummies(
    df,
    columns=[
        "admission_type",
        "admission_location",
        "gender",
    ],
    drop_first=False,
    dummy_na=False,
    dtype=int,
)

# Nur gender_F verwenden
df_reduced = df_reduced.drop(
    columns=["gender_M"],
    errors="ignore"
)


# ============================================================
# 12) Sparse-Matrix-Funktionen
# ============================================================

def create_sparse_matrix(
    df,
    list_columns,
    target="hospital_expire_flag",
    id_cols=("hadm_id", "subject_id"),
):
    df_work = df.copy().reset_index(drop=True)

    y = df_work[target].astype(int).to_numpy(dtype=np.int8)
    subject_ids = df_work["subject_id"].to_numpy()
    hadm_ids = df_work["hadm_id"].to_numpy()

    exclude_cols = list(list_columns) + list(id_cols) + [target]

    base_cols = [
        c for c in df_work.columns
        if c not in exclude_cols
    ]

    X_base = df_work[base_cols].copy()

    bool_cols = X_base.select_dtypes(include="bool").columns
    X_base[bool_cols] = X_base[bool_cols].astype(int)

    non_numeric_cols = X_base.select_dtypes(exclude=[np.number]).columns
    print("Nicht-numerische Basisfeatures:", non_numeric_cols.tolist())
    assert len(non_numeric_cols) == 0, "Es gibt noch nicht-numerische Basisfeatures."

    X_parts = [
        csr_matrix(X_base.to_numpy(dtype=np.float32))
    ]

    feature_names = list(X_base.columns)
    mlb_dict = {}

    for col, prefix in list_columns.items():
        lists = df_work[col].apply(
            lambda x: x if isinstance(x, list) else []
        )

        mlb = MultiLabelBinarizer(sparse_output=True)
        X_col = mlb.fit_transform(lists).astype(np.float32)

        X_parts.append(X_col)

        feature_names.extend([
            f"{prefix}_{code}" for code in mlb.classes_
        ])

        mlb_dict[col] = mlb

    X_sparse = hstack(X_parts, format="csr").astype(np.float32)

    assert X_sparse.shape[0] == len(y)
    assert X_sparse.shape[0] == len(subject_ids)
    assert X_sparse.shape[0] == len(hadm_ids)
    assert X_sparse.shape[1] == len(feature_names)
    assert target not in feature_names
    assert "subject_id" not in feature_names
    assert "hadm_id" not in feature_names

    return {
        "X_sparse": X_sparse,
        "y": y,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "feature_names": feature_names,
        "base_cols": base_cols,
        "list_columns": list_columns,
        "mlb_dict": mlb_dict,
    }


def save_sparse_dataset(dataset, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    save_npz(output_path / "X_sparse.npz", dataset["X_sparse"])
    np.save(output_path / "y.npy", dataset["y"])
    np.save(output_path / "subject_id.npy", dataset["subject_ids"])
    np.save(output_path / "hadm_id.npy", dataset["hadm_ids"])

    with open(output_path / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(dataset["feature_names"], f, ensure_ascii=False, indent=2)

    with open(output_path / "base_cols.json", "w", encoding="utf-8") as f:
        json.dump(dataset["base_cols"], f, ensure_ascii=False, indent=2)

    with open(output_path / "list_columns.json", "w", encoding="utf-8") as f:
        json.dump(dataset["list_columns"], f, ensure_ascii=False, indent=2)

    with open(output_path / "mlb_dict.pkl", "wb") as f:
        pickle.dump(dataset["mlb_dict"], f)

    print("Gespeichert:", output_path)
    print("X_sparse shape:", dataset["X_sparse"].shape)
    print("y shape:", dataset["y"].shape)
    print("Anzahl Features:", len(dataset["feature_names"]))
    print("Mortalitätsrate:", dataset["y"].mean())
    print("Anzahl eindeutige Patienten:", len(np.unique(dataset["subject_ids"])))


# ============================================================
# 13) Check: keine kumulativen hist_*-Features mehr vorhanden
# ============================================================

remaining_hist_cols = [
    c for c in df_reduced.columns
    if c.startswith("hist_")
]

print("Verbliebene hist_*-Spalten:", remaining_hist_cols)
assert len(remaining_hist_cols) == 0, (
    "Es sind noch kumulative hist_*-Features vorhanden: "
    f"{remaining_hist_cols}"
)


# ============================================================
# 14) Konsistenzcheck der zeitlichen Featuredefinition
# ============================================================

required_list_and_count_cols = [
    "last_icd10_chapter_list",
    "last_pcs_body_system_list",
    "last_n_diag_codes",
    "last_n_proc_codes",
    "last_n_diag_chapters",
]

required_final_cols = sorted(set(
    required_list_and_count_cols
    + last_semantic_cols
    + last_proc_metric_cols
))

missing_final_cols = [
    col for col in required_final_cols
    if col not in df_reduced.columns
]

assert not missing_final_cols, (
    "Erwartete historische oder Last-Stay-Features fehlen: "
    f"{missing_final_cols}"
)

forbidden_old_count_cols = [
    "last_n_unique_diag_codes",
    "last_n_unique_proc_codes",
    "last_n_unique_icd10_chapters",
    "last_n_unique_pcs_body_systems",
]

remaining_old_count_cols = [
    col for col in forbidden_old_count_cols
    if col in df_reduced.columns
]

assert not remaining_old_count_cols, (
    "Nicht mehr vorgesehene Last-Unique-Counts sind noch vorhanden: "
    f"{remaining_old_count_cols}"
)

print("Zeitlicher Konsistenzcheck bestanden.")


# ============================================================
# 15) Sparse-Datensätze erstellen
# ============================================================

# ------------------------------------------------------------
# 15.1 Baseline-Modell der Sensitivitätsanalyse
# ------------------------------------------------------------

baseline_base_cols = [
    "age_at_admission",
    "gender_F",
    "n_prev_admissions",
    "n_prev_emergency_admissions",
    "time_since_last_discharge_days",
    "had_admission_last_30d",
    "max_past_los",
    "avg_past_los",
    "has_previous_admission",
]

# Alle erzeugten Dummies für Aufnahmeart und Aufnahmeort ergänzen.
baseline_base_cols += [
    c for c in df_reduced.columns
    if c.startswith("admission_type_")
]

baseline_base_cols += [
    c for c in df_reduced.columns
    if c.startswith("admission_location_")
]

# IDs, Zielvariable und die beiden benötigten Code-Listen ergänzen.
baseline_keep_cols = (
    [
        "hadm_id",
        "subject_id",
        "hospital_expire_flag",
        "last_diag_code_list",
        "last_proc_code_list",
    ]
    + baseline_base_cols
)

missing_baseline_cols = [
    c for c in baseline_keep_cols
    if c not in df_reduced.columns
]

assert not missing_baseline_cols, (
    "Folgende erwartete Baseline-Spalten fehlen: "
    f"{missing_baseline_cols}"
)

df_baseline = df_reduced[baseline_keep_cols].copy()

baseline_list_columns = {
    "last_diag_code_list": "last_diag_code",
    "last_proc_code_list": "last_proc_code",
}

baseline_dataset = create_sparse_matrix(
    df=df_baseline,
    list_columns=baseline_list_columns,
)

# Export
save_sparse_dataset(
    dataset=baseline_dataset,
    output_path=path / OUTPUT_BASELINE_MODEL
)


# ------------------------------------------------------------
# 15.2 Full-Modell der Sensitivitätsanalyse
# ------------------------------------------------------------
# Enthält Multi-Hot-Codes, Kapitel/Body-Systeme und semantische Metriken
# ausschließlich aus dem unmittelbar vorherigen Aufenthalt.

full_list_columns = {
    # ICD-Codes des unmittelbar vorherigen Aufenthalts
    "last_diag_code_list": "last_diag_code",
    "last_proc_code_list": "last_proc_code",

    # Verteilungsmerkmale des unmittelbar vorherigen Aufenthalts
    "last_icd10_chapter_list": "last_icd10_chapter",
    "last_pcs_body_system_list": "last_pcs_body_system",
}

full_dataset = create_sparse_matrix(
    df=df_reduced,
    list_columns=full_list_columns
)

# Export
save_sparse_dataset(
    dataset=full_dataset,
    output_path=path / OUTPUT_FULL_MODEL
)


# ============================================================
# 16) Checks der Sparse-Datensätze
# ============================================================

def check_sparse_dataset(
    dataset,
    df_reference,
    dataset_name="dataset",
    example_indices=(0, 6000),
    n_random_checks=10,
    random_state=42
):
    X_sparse = dataset["X_sparse"]
    feature_names = dataset["feature_names"]
    list_columns = dataset["list_columns"]

    df_reference = df_reference.reset_index(drop=True)

    print("=" * 80)
    print(f"Checks für: {dataset_name}")
    print("=" * 80)

    assert X_sparse.shape[0] == df_reference.shape[0], (
        f"{dataset_name}: X_sparse und df_reference haben unterschiedlich viele Zeilen."
    )

    assert X_sparse.shape[1] == len(feature_names), (
        f"{dataset_name}: Anzahl Feature-Namen passt nicht zur Matrix."
    )

    print("Shape:", X_sparse.shape)
    print("Anzahl Features:", len(feature_names))
    print("List Columns:", list_columns)

    multi_hot_prefixes = tuple(
        f"{prefix}_"
        for prefix in list_columns.values()
    )

    multi_hot_indices = [
        i for i, f in enumerate(feature_names)
        if f.startswith(multi_hot_prefixes)
    ]

    assert len(multi_hot_indices) > 0, (
        f"{dataset_name}: Keine Multi-Hot-Features gefunden."
    )

    X_multi_hot = X_sparse[:, multi_hot_indices]
    unique_values = set(X_multi_hot.data)

    print("\nUnique Non-Zero Values in Multi-Hot Features:")
    print(unique_values)

    assert unique_values.issubset({1, 1.0, np.float32(1.0)}), (
        f"{dataset_name}: Multi-Hot-Features enthalten andere Werte als 1."
    )

    print("Multi-Hot-Werte sind korrekt: nur 0/1.")

    for example_idx in example_indices:
        if example_idx >= X_sparse.shape[0]:
            print(f"\nBeispielindex {example_idx} übersprungen.")
            continue

        row = X_sparse[example_idx]
        active_feature_indices = row.indices
        active_features = [feature_names[i] for i in active_feature_indices]

        print("\n" + "-" * 80)
        print(f"Beispiel-Aufenthalt Index: {example_idx}")

        if "hadm_id" in df_reference.columns:
            print("HADM ID:")
            print(df_reference.iloc[example_idx]["hadm_id"])

        print("has_previous_admission:")
        print(df_reference.iloc[example_idx].get("has_previous_admission", "nicht vorhanden"))

        for col, prefix in list_columns.items():
            original_codes = df_reference.iloc[example_idx][col]

            if not isinstance(original_codes, list):
                original_codes = []

            expected_features = [f"{prefix}_{code}" for code in original_codes]
            active_features_for_col = [
                f for f in active_features
                if f.startswith(f"{prefix}_")
            ]

            missing = set(expected_features) - set(active_features_for_col)
            unexpected = set(active_features_for_col) - set(expected_features)

            print("\n", col)
            print("Original Codes/Kategorien des letzten Aufenthalts:")
            print(original_codes[:20], "..." if len(original_codes) > 20 else "")
            print("Anzahl:", len(original_codes))
            print("Fehlend:", missing)
            print("Unerwartet:", unexpected)

            assert len(missing) == 0, (
                f"{dataset_name}: Fehlende Features bei Index {example_idx}, "
                f"Spalte {col}: {missing}"
            )

            assert len(unexpected) == 0, (
                f"{dataset_name}: Unerwartete Features bei Index {example_idx}, "
                f"Spalte {col}: {unexpected}"
            )

    rng = random.Random(random_state)
    n_available = df_reference.shape[0]
    n_checks = min(n_random_checks, n_available)
    random_indices = rng.sample(range(n_available), n_checks)

    for example_idx in random_indices:
        row = X_sparse[example_idx]
        active_features = [feature_names[i] for i in row.indices]

        for col, prefix in list_columns.items():
            original_codes = df_reference.iloc[example_idx][col]

            if not isinstance(original_codes, list):
                original_codes = []

            expected_features = set(f"{prefix}_{code}" for code in original_codes)
            active_features_for_col = set(
                f for f in active_features
                if f.startswith(f"{prefix}_")
            )

            assert expected_features == active_features_for_col, (
                f"{dataset_name}: Mismatch bei Index {example_idx}, Spalte {col}. "
                f"Fehlend: {expected_features - active_features_for_col}. "
                f"Unerwartet: {active_features_for_col - expected_features}."
            )

    print("\nAlle Stichproben-Checks bestanden.")
    print("=" * 80)

check_sparse_dataset(
    dataset=baseline_dataset,
    df_reference=df_baseline,
    dataset_name="Baseline Model - No Current Codes",
    example_indices=(0, 6000),
    n_random_checks=10
)

check_sparse_dataset(
    dataset=full_dataset,
    df_reference=df_reduced,
    dataset_name="Full Model - No Current Codes",
    example_indices=(0, 6000),
    n_random_checks=10
)


# ============================================================
# 17) Finaler Leakage-Check
# ============================================================
# Prüft exemplarisch, dass aktuelle Codelisten nicht mehr als Feature-Listen im finalen df_reduced vorkommen.

for forbidden_col in [
    "diag_code_list",
    "proc_code_list",
    "icd10_chapter_list",
    "current_diag_code_list",
    "current_proc_code_list",
    "current_icd10_chapter_list",
    "current_pcs_body_system_list",
    "hist_diag_code_list",
    "hist_proc_code_list",
    "hist_icd10_chapter_list",
    "hist_pcs_body_system_list",
]:
    assert forbidden_col not in df_reduced.columns, f"Leakage-Risiko: {forbidden_col} ist noch im finalen DataFrame."

print("Finaler Leakage-Check bestanden: aktuelle Code-Listen sind nicht im finalen DataFrame enthalten.")

