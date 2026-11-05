#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELLAUSWAHL

Verglichen werden:
- Logistische Regression mit L2-Regularisierung
- Logistische Regression mit Elastic Net
- Random Forest
- XGBoost

Der Vergleich erfolgt anhand einer patientenbasierten Cross-Validation im Trainingsdatensatz. 
Hierzu wird zunächst ein einheitlicher Train-Test-Split erzeugt und für spätere Wiederverwendung
exportiert. 
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from scipy import sparse
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# PFAD ANPASSEN!
path = Path("Pfad_zu_zuvor_gespeicherten_Daten")
full_model_path = path / "full_model"
procedure_metrics = pd.read_csv(path / "procedure_metrics.csv")


# =========================================================
# 1) Funktion zum Laden eines Modells
# =========================================================

def load_model_data(model_path):
    model_path = Path(model_path)
  
    with open(model_path / "base_cols.json", "r") as f:
        base_cols = json.load(f)
    
    with open(model_path / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    with open(model_path / "list_columns.json", "r") as f:
        list_columns = json.load(f)
    
    with open(model_path / "mlb_dict.pkl", "rb") as f:
        mlb_dict = pickle.load(f)
    
    subject_id = np.load(model_path / "subject_id.npy")
    hadm_id = np.load(model_path / "hadm_id.npy")
    y = np.load(model_path / "y.npy")
    
    X_sparse = sparse.load_npz(
        model_path / "X_sparse.npz"
    )
    
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
# 2) Full Model laden
# =========================================================

full_data = load_model_data(full_model_path)

X_full = full_data["X_sparse"]
y_full = full_data["y"]
subject_id_full = full_data["subject_id"]
hadm_id_full = full_data["hadm_id"]

full_base_cols = full_data["base_cols"]
full_feature_names = full_data["feature_names"]
full_list_columns = full_data["list_columns"]
full_mlb_dict = full_data["mlb_dict"]

print("=== Full Model vor Behandlungsmetriken ===")
print("X shape:", X_full.shape)
print("y shape:", y_full.shape)
print("subject_id shape:", subject_id_full.shape)
print("hadm_id shape:", hadm_id_full.shape)


# =========================================================
# 3) Behandlungsmetriken an Full Model anhängen
# =========================================================

# procedure_metrics muss genau eine Zeile pro Aufenthalt besitzen
assert procedure_metrics["hadm_id"].is_unique, (
    "procedure_metrics.csv enthält doppelte hadm_id."
)

# ---------------------------------------------------------
# 3.1) Behandlungsmetriken an Reihenfolge von X_full ausrichten
# ---------------------------------------------------------

procedure_metrics_aligned = (
    pd.DataFrame(
        {"hadm_id": hadm_id_full}
    )
    .merge(
        procedure_metrics,
        on="hadm_id",
        how="left"
    )
)

# Sicherheitscheck:
# Reihenfolge der Aufenthalte muss unverändert sein
assert np.array_equal(
    procedure_metrics_aligned["hadm_id"].values,
    hadm_id_full
), (
    "Alignment der Behandlungsmetriken fehlgeschlagen."
)

# ---------------------------------------------------------
# 3.2) Alle Behandlungsmetriken übernehmen
# ---------------------------------------------------------

proc_metric_cols = [
    col
    for col in procedure_metrics_aligned.columns
    if col != "hadm_id"
]

print("\nBehandlungsmetriken für Modellauswahl:")
print(proc_metric_cols)
print("Anzahl:", len(proc_metric_cols))

# ---------------------------------------------------------
# 3.3) Prüfen, ob versehentlich nicht-numerische Spalten enthalten sind
# ---------------------------------------------------------

non_numeric_proc_cols = [
    col
    for col in proc_metric_cols
    if not pd.api.types.is_numeric_dtype(
        procedure_metrics_aligned[col]
    )
]

print("\nNicht-numerische Behandlungsmerkmale:", non_numeric_proc_cols)

assert len(non_numeric_proc_cols) == 0, (
    "procedure_metrics enthält nicht-numerische Feature-Spalten."
)

# ---------------------------------------------------------
# 3.4) Prüfen, ob Features bereits in X_full vorhanden sind
# ---------------------------------------------------------

duplicate_features = sorted(
    set(full_feature_names)
    .intersection(proc_metric_cols)
)

print(
    "\nBereits im Full Model vorhandene "
    "Behandlungsmerkmale:",
    duplicate_features
)

assert len(duplicate_features) == 0, (
    "Einige Behandlungsmetriken sind bereits im Full Model "
    "enthalten und würden doppelt aufgenommen."
)

# ---------------------------------------------------------
# 3.5) Behandlungsmetriken numerisch vorbereiten
# ---------------------------------------------------------

proc_metrics_df = (
    procedure_metrics_aligned[proc_metric_cols]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
    .astype(float)
)

X_proc_metrics = sparse.csr_matrix(proc_metrics_df.values)

# ---------------------------------------------------------
# 3.6) An Full Model anhängen
# ---------------------------------------------------------

X_full_with_proc = sparse.hstack(
    [
        X_full,
        X_proc_metrics
    ],
    format="csr"
)

full_feature_names_with_proc = (
    list(full_feature_names)
    + proc_metric_cols
)

print("\n=== Full Model inklusive Behandlungsmetriken ===")
print("X vorher:", X_full.shape)
print("X nachher:", X_full_with_proc.shape)
print("Zusätzliche Behandlungsmetriken:", len(proc_metric_cols))
print("Gesamtzahl Features:", len(full_feature_names_with_proc))

assert (X_full_with_proc.shape[1] == len(full_feature_names_with_proc))


# =========================================================
# 4) Daten für Modellauswahl definieren
# =========================================================

X = X_full_with_proc
y = y_full
groups = subject_id_full
hadm_id = hadm_id_full


print("\n=== Datensatz für Modellauswahl ===")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Mortalitätsrate:", y.mean())
print("Anzahl eindeutige Patient:innen:", len(np.unique(groups)))


# =========================================================
# 5) Einheitlicher patientenbasierter Train/Test-Split
# =========================================================
#
# Der Testdatensatz wird hier erzeugt und gespeichert,
# aber während der Modellauswahl NICHT verwendet.
#
# StratifiedGroupKFold:
# - Aufenthalte desselben Patienten bleiben zusammen
# - Mortalitätsrate wird möglichst ähnlich verteilt
#
# Erster Fold:
# ca. 80 % Training
# ca. 20 % Test

outer_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    outer_split.split(
        X=np.zeros(len(y)),
        y=y,
        groups=groups
    )
)


# =========================================================
# 6) Einheitlichen Split speichern
# =========================================================

split_path = path / "train_test_split"
split_path.mkdir(parents=True, exist_ok=True)

np.save(split_path / "train_idx.npy", train_idx)
np.save(split_path / "test_idx.npy", test_idx)

np.save(split_path / "train_hadm_id.npy", hadm_id[train_idx])
np.save(split_path / "test_hadm_id.npy", hadm_id[test_idx])

np.save(split_path / "train_subject_id.npy", groups[train_idx])
np.save(split_path / "test_subject_id.npy", groups[test_idx])

print("\nTrain/Test-Split gespeichert unter:")
print(split_path)


# =========================================================
# 7) Trainingsdatensatz definieren
# =========================================================
# Nur der Trainingsdatensatz wird für die Modellauswahl verwendet.

X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

# Testdaten werden nur für Checks definiert, NICHT für die Modellauswahl verwendet.
y_test = y[test_idx]

groups_test = groups[test_idx]

print("\n=== Train/Test-Split ===")
print("Train shape:", X_train.shape)
print("Test Anzahl Aufenthalte:", len(test_idx))
print("Train Mortalitätsrate:", y_train.mean())
print("Test Mortalitätsrate:", y_test.mean())

# Sicherheitscheck: Kein Patient darf in beiden Splits vorkommen.
overlap = set(groups_train).intersection(set(groups_test))
print("Patient:innen in beiden Splits:", len(overlap))

assert len(overlap) == 0, (
    "Patient:innen kommen sowohl im Training "
    "als auch im Testdatensatz vor."
)


# =========================================================
# 8) NaNs / Inf im Trainingsdatensatz bereinigen
# =========================================================

X_train_clean = X_train.copy()

X_train_clean.data = np.nan_to_num(
    X_train_clean.data,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_train_clean = (X_train_clean.tocsr())

print(
    "\nNaNs nach Bereinigung X_train:",
    np.isnan(
        X_train_clean.data
    ).sum()
)


# =========================================================
# 9) Cross-Validation für Modellauswahl
# =========================================================
# Alle Verfahren verwenden:
# - dieselben Trainingsdaten
# - dieselbe Group-CV
# - dieselbe Zielmetrik
#
# scoring = average_precision
# Average Precision (AP) als Precision-Recall-basierte
# Bewertungsmetrik für den unbalancierten Outcome.

cv_fast = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Einheitliche Anzahl zufälliger Parameterkombinationen
N_ITER = 3


# =========================================================
# 10) Logistische Regression mit L2 Regularisierung
# =========================================================

param_dist_logreg = {
    "C": loguniform(
        1e-2,
        1e2
    )
}

logreg = LogisticRegression(
    penalty="l2",
    solver="saga",
    class_weight="balanced",
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1,
    random_state=42
)

logreg_search = RandomizedSearchCV(
    estimator=logreg,
    param_distributions=param_dist_logreg,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv_fast,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    error_score="raise",
    return_train_score=True
)

logreg_search.fit(
    X_train_clean,
    y_train,
    groups=groups_train
)

print("\n=== Logistische Regression L2 ===")
print("Beste Parameter:")
print(logreg_search.best_params_)
print("Beste CV Average Precision:")
print(logreg_search.best_score_)


# =========================================================
# 11) Logistische Regression mit Elastic Net
# =========================================================
# Achtung: Dieser Durchlauf dauert mehrere Stunden

param_dist_elastic = {
    "C": loguniform(
        1e-2,
        1e2
    ),
    "l1_ratio": uniform(
        0.05,
        0.9
    )
}

elastic_net_model = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1,
    random_state=42
)

elastic_net_search = RandomizedSearchCV(
    estimator=elastic_net_model,
    param_distributions=param_dist_elastic,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv_fast,
    n_jobs=-1,
    verbose=3,
    random_state=42,
    error_score="raise",
    return_train_score=True
)

elastic_net_search.fit(
    X_train_clean,
    y_train,
    groups=groups_train
)

print("\n=== Logistische Regression Elastic Net ===")
print("Beste Parameter:")
print(elastic_net_search.best_params_)
print("Beste CV Average Precision:")
print(elastic_net_search.best_score_)


# =========================================================
# 12) Random Forest
# =========================================================

param_dist_rf = {
    "n_estimators": randint(150, 500),
    "max_depth": [5, 10, 15, 20],
    "min_samples_leaf": randint(5, 40),
    "min_samples_split": randint(5, 50),
    "max_features": ["sqrt", "log2"]
}

rf_model = RandomForestClassifier(
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist_rf,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv_fast,
    n_jobs=-1,
    verbose=10,
    random_state=42,
    error_score="raise",
    return_train_score=True
)

rf_search.fit(
    X_train_clean,
    y_train,
    groups=groups_train
)

print("\n=== Random Forest ===")
print("Beste Parameter:")
print(rf_search.best_params_)
print("Beste CV Average Precision:")
print(rf_search.best_score_)


# =========================================================
# 13) XGBoost
# =========================================================

n_negative = np.sum(y_train == 0)

n_positive = np.sum(y_train == 1)

scale_pos_weight = (n_negative / n_positive)

print("\nNegative Fälle:", n_negative)
print("Positive Fälle:", n_positive)
print("scale_pos_weight:", scale_pos_weight)

param_dist_xgb = {
    "n_estimators": randint(50, 200),
    "max_depth": randint(2, 4),
    "learning_rate": loguniform(0.05, 0.2),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.5, 0.4),
    "min_child_weight": randint(10, 50),
    "reg_lambda": loguniform(1e-1, 1e1)
}

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42,
    verbosity=1
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv_fast,
    n_jobs=-1,
    verbose=10,
    random_state=42,
    error_score="raise",
    return_train_score=True
)

xgb_search.fit(
    X_train_clean,
    y_train,
    groups=groups_train
)

print("\n=== XGBoost ===")
print("Beste Parameter:")
print(xgb_search.best_params_)
print("Beste CV Average Precision:")
print(xgb_search.best_score_)


# =========================================================
# 14) Modellvergleich ausschließlich anhand
#     der Cross-Validation im Trainingsdatensatz
# =========================================================

def get_best_cv_std(search):
    """
    Gibt die Standardabweichung der CV-Average Precision (AP)
    für die beste Parameterkombination zurück.
    """

    return search.cv_results_[
        "std_test_score"
    ][
        search.best_index_
    ]


model_selection_results = pd.DataFrame(
    [
        {
            "model": "Logistische Regression L2",
            "cv_ap_mean": logreg_search.best_score_,
            "cv_ap_std": get_best_cv_std(logreg_search),
            "best_params": logreg_search.best_params_              
        },
        {
            "model": "Logistische Regression Elastic Net",
            "cv_ap_mean": elastic_net_search.best_score_,
            "cv_ap_std": get_best_cv_std(elastic_net_search),
            "best_params":elastic_net_search.best_params_               
        },
        {
            "model": "Random Forest",
            "cv_ap_mean": rf_search.best_score_,
            "cv_ap_std": get_best_cv_std(rf_search),
            "best_params": rf_search.best_params_                
        },
        {
            "model": "XGBoost",
            "cv_ap_mean": xgb_search.best_score_,
            "cv_ap_std": get_best_cv_std(xgb_search),
            "best_params": xgb_search.best_params_               
        }
    ]
)

model_selection_results = (
    model_selection_results
    .sort_values(
        by="cv_ap_mean",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print("\n==========================================")
print("MODELLAUSWAHL")
print("nur anhand Cross-Validation im Training")
print("==========================================")
print(
    model_selection_results[
        [
            "model",
            "cv_ap_mean",
            "cv_ap_std"
        ]
    ]
)

print("\nBestes Modell:")
print(model_selection_results.iloc[0]["model"])

print("\nBeste Hyperparameter:")
print(model_selection_results.iloc[0]["best_params"])



