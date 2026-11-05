"""
TRANSFORMATION AUF AUFENTHALTSEBENE

Das Skript transformiert den Code-Level-Datensatz auf Aufenthaltsebene und integriert die zuvor 
erzeugten historischen Patientenmerkmale. Diagnosebezogene semantische Kennzahlen werden 
aggregiert, kategoriale Variablen binär kodiert und seltene Diagnose- und Behandlungscodes 
anhand festgelegter Mindesthäufigkeiten ausgeschlossen. Anschließend werden die finalen 
Baseline- und Full-Model-Datensätze erzeugt und als Sparse-Matrizen exportiert. Auch die
Baseline ohne ICD-10-Codes wird exportiert.
"""

import pandas as pd
import numpy as np
import ast
import gc
from pandas.api.types import CategoricalDtype
from collections import Counter
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.preprocessing import MultiLabelBinarizer
import random
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# PFAD ANPASSEN!
path = Path("Pfad_zu_zuvor_gespeicherten_Daten")
df = pd.read_csv(path / "mimic_with_semantics_categories_mortality_codelevel.csv")
stay_history = pd.read_csv(path / "historical_features_staylevel.csv")


# ============================================================
# 0) Check: Anzahl eindeutiger Aufenthalte vor Transformationen
# ============================================================

n_unique_hadm_df = df["hadm_id"].nunique()
n_rows_df = len(df)

n_unique_hadm_history = stay_history["hadm_id"].nunique()
n_rows_history = len(stay_history)

print("df:")
print("Zeilen:", n_rows_df)
print("Unique hadm_id:", n_unique_hadm_df)

print("\nstay_history:")
print("Zeilen:", n_rows_history)
print("Unique hadm_id:", n_unique_hadm_history)

print("\nSoll-Zeilenzahl finaler Datensatz:", n_unique_hadm_df)
print("stay_history bereits auf Aufenthaltsebene:",
      n_unique_hadm_history == n_rows_history)


# ============================================================
# 1) Schlanken Aufenthalt-Datensatz bauen 
# ============================================================
# Aufenthaltsebene: nicht-codebezogene Variablen
# Diese Variablen sollten pro hadm_id konstant sein.
# Es wird jeweils den ersten Wert je Aufenthalt übernommen.

stay_check = (
    df.sort_values(["hadm_id", "seq_num"], na_position="last")
      .groupby("hadm_id", as_index=False)
      .agg({
          "subject_id": "first",
          "admittime": "first",
          "dischtime": "first",
          "admission_type": "first",
          "admission_location": "first",
          "gender": "first",
          "hospital_expire_flag": "first",
          "age_at_admission": "first"
      })
)


# ============================================================
# 2) ICD-Codes pro Aufenthalt 
# ============================================================
# Jeder Aufenthalt bekommt zwei Listen aller eindeutigen Codes 
# (getrennt nach Diagnose- und Behandlungscodes) und eine Liste der Diagnosekapitel

# ---------------------------------------
# Diagnosen pro Aufenthalt
# ---------------------------------------

diag_codes_per_stay = (
    df[df["code_type"].str.lower().eq("diagnosis")]
    .dropna(subset=["icd_code"])
    .groupby(["subject_id", "hadm_id"])["icd_code"]
    .apply(lambda x: sorted(x.astype(str).unique()))
    .reset_index(name="diag_code_list")
)

# ---------------------------------------
# Behandlungen pro Aufenthalt
# ---------------------------------------
proc_codes_per_stay = (
    df[df["code_type"].str.lower().eq("procedure")]
    .dropna(subset=["icd_code"])
    .groupby(["subject_id", "hadm_id"])["icd_code"]
    .apply(lambda x: sorted(x.astype(str).unique()))
    .reset_index(name="proc_code_list")
)

# ---------------------------------------
# Diagnosekapitel pro Aufenthalt
# ---------------------------------------
diag_chapters_per_stay = (
    df[df["code_type"].str.lower().eq("diagnosis")]
    .dropna(subset=["icd10_chapter"])
    .groupby(["subject_id", "hadm_id"])["icd10_chapter"]
    .apply(lambda x: sorted(x.astype(str).unique()))
    .reset_index(name="icd10_chapter_list")
)

# ---------------------------------------
# Merge 
# ---------------------------------------
stay_level = stay_check.copy()

stay_level = stay_level.merge(diag_codes_per_stay, on=["subject_id", "hadm_id"], how="left")
stay_level = stay_level.merge(proc_codes_per_stay, on=["subject_id", "hadm_id"], how="left")
stay_level = stay_level.merge(diag_chapters_per_stay, on=["subject_id", "hadm_id"], how="left")

# ---------------------------------------
# Leere Listen statt NaN
# ---------------------------------------
for col in ["diag_code_list", "proc_code_list", "icd10_chapter_list"]:
    stay_level[col] = stay_level[col].apply(
        lambda x: x if isinstance(x, list) else []
    )

# ---------------------------------------
# Anzahl Codes bzw. Kapitel pro Aufenthalt 
# ---------------------------------------
stay_level["n_diag_codes"] = stay_level["diag_code_list"].apply(len)
stay_level["n_proc_codes"] = stay_level["proc_code_list"].apply(len)
stay_level["n_diag_chapters"] = stay_level["icd10_chapter_list"].apply(len)


# Check
print(stay_level[["n_diag_codes", "n_proc_codes", "n_diag_chapters"]].describe())

print("Beispiel Diagnosen:", stay_level["diag_code_list"].iloc[0][:5])
print("Beispiel Behandlungen:", stay_level["proc_code_list"].iloc[0][:5])
print("Beispiel Kapitel:", stay_level["icd10_chapter_list"].iloc[0][:5])

# Stimmen Counts mit Listen überein?
check = (
    (stay_level["n_diag_codes"] == stay_level["diag_code_list"].apply(len)) &
    (stay_level["n_proc_codes"] == stay_level["proc_code_list"].apply(len)) &
    (stay_level["n_diag_chapters"] == stay_level["icd10_chapter_list"].apply(len))
)

print(check.mean())  # sollte 1.0 sein


# ============================================================
# 3) Codelisten der Patientenhistorie
# ============================================================
def parse_code_collection(x):
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return sorted(x)

    if isinstance(x, set):
        return sorted(list(x))

    if isinstance(x, str):
        x = x.strip()

        if x in ["", "set()", "[]", "nan", "NaN", "None"]:
            return []

        try:
            parsed = ast.literal_eval(x)

            if isinstance(parsed, set):
                return sorted(list(parsed))
            if isinstance(parsed, list):
                return sorted(parsed)
            if isinstance(parsed, tuple):
                return sorted(list(parsed))

        except Exception:
            pass

        return [x]

    return []


# Neue historische Listen-Spalten aus CSV wieder in echte Listen umwandeln
history_list_cols = [
    "prev_diag_codes",
    "prev_proc_codes",
    "prev_icd10_chapters"
]

for col in history_list_cols:
    if col in stay_history.columns:
        stay_history[col] = stay_history[col].apply(parse_code_collection)


# Checks: Listen korrekt eingelesen?
for col in history_list_cols:
    if col in stay_history.columns:
        non_empty = stay_history[col].apply(len) > 0

        print(col)
        print("Anteil nicht-leer:", non_empty.mean())
        print("Anzahl nicht-leer:", non_empty.sum())
        print("Max Länge:", stay_history[col].apply(len).max())
        print(stay_history.loc[non_empty, col].head())
        print()


# Checks: Count-Variablen passen zu Listenlängen?
history_count_pairs = {
    "prev_diag_codes": "n_prev_unique_diag_codes",
    "prev_proc_codes": "n_prev_unique_proc_codes",
    "prev_icd10_chapters": "n_prev_unique_diag_chapters"
}

for list_col, count_col in history_count_pairs.items():
    if list_col in stay_history.columns and count_col in stay_history.columns:
        check = stay_history[count_col].fillna(0).astype(int).eq(
            stay_history[list_col].apply(len)
        )

        print(f"{count_col} passt zu {list_col}:")
        print(check.value_counts())
        print("Anteil korrekt:", check.mean())
        print()


# Nicht benötigte Spalten löschen
stay_history = stay_history.drop(
    columns=["admittime", "dischtime", "admission_type", "los_days", "prev_dischtime", 
             "prev_diag_codes", "prev_proc_codes", "is_emergency_admission"],
    errors="ignore"
)

# Historische Features an stay_level mergen
stay_level = stay_level.merge(
    stay_history,
    on=["subject_id", "hadm_id"],
    how="left",
    suffixes=("", "_hist")
)

print([c for c in stay_level.columns if c.endswith("_hist")])

print("Zeilen nach Merge:", len(stay_level))
print("Unique hadm_id:", stay_level["hadm_id"].nunique())
print("Aufenthaltsebene?", len(stay_level) == stay_level["hadm_id"].nunique())


# ============================================================
# 4) Prevalence- und IC-Variablen auf Aufenthaltsebene aggregieren
# ============================================================

prevalence_ic_agg = (
    df
    .groupby(["subject_id", "hadm_id"], as_index=False)
    .agg(
        # Diagnosecodes
        prevalence_diag_mean=("prevalence_diag", "mean"),
        prevalence_diag_min=("prevalence_diag", "min"),
        prevalence_diag_max=("prevalence_diag", "max"),
        prevalence_diag_std=("prevalence_diag", "std"),

        ic_diag_mean=("ic_diag", "mean"),
        ic_diag_min=("ic_diag", "min"),
        ic_diag_max=("ic_diag", "max"),
        ic_diag_std=("ic_diag", "std"),

        # Prozedurcodes
        prevalence_proc_mean=("prevalence_proc", "mean"),
        prevalence_proc_min=("prevalence_proc", "min"),
        prevalence_proc_max=("prevalence_proc", "max"),
        prevalence_proc_std=("prevalence_proc", "std"),

        ic_proc_mean=("ic_proc", "mean"),
        ic_proc_min=("ic_proc", "min"),
        ic_proc_max=("ic_proc", "max"),
        ic_proc_std=("ic_proc", "std"),
    )
)

# Standardabweichung bei nur einem Code ist NaN -> auf 0 setzen
std_cols = [col for col in prevalence_ic_agg.columns if col.endswith("_std")]
prevalence_ic_agg[std_cols] = prevalence_ic_agg[std_cols].fillna(0)


# ============================================================
# 5) Aggregation der Diagnosecode-Metriken
# ============================================================

# ---------------------------------------
# 5.1) Vorbereitung: nur Diagnosecodes für Hierarchie-Metriken
# ---------------------------------------

df_metrics = df.copy()

# Nur Diagnosecodes behalten
df_metrics = df_metrics[df_metrics["code_type"].str.lower().eq("diagnosis")].copy()


# Check
print(df_metrics["code_type"].value_counts(dropna=False))
print("Zeilen nach Filter auf Diagnosecodes:", len(df_metrics))
print("Eindeutige Aufenthalte mit Diagnosecodes:", df_metrics["hadm_id"].nunique())

# Jetzt sollte is_leaf keine NaNs mehr wegen Behandlungscodes enthalten
print("NaNs in is_leaf nach Diagnosefilter:", df_metrics["is_leaf"].isna().sum())

# Sicherheitscheck: Für alle vorhandenen Diagnosecodezeilen sollten Hierarchieinformationen vorliegen.
assert df_metrics["is_leaf"].notna().all(), (
    "Für einzelne Diagnosecodezeilen fehlt is_leaf."
)

# is_leaf sauber als Integer kodieren
df_metrics["is_leaf"] = df_metrics["is_leaf"].astype(int)

# Innere Knoten: Codes, die keine Blätter sind
df_metrics["is_internal_node"] = (df_metrics["is_leaf"] == 0).astype(int)


# Datentypen optimieren
# IDs kleiner speichern
df_metrics["subject_id"] = df_metrics["subject_id"].astype("int32")
df_metrics["hadm_id"] = df_metrics["hadm_id"].astype("int32")

# Kategorie-Variablen effizient speichern
cat_vars = [
    "rel_subtree_log_cat",
    "siblings_3cat",
    "depth_ord"
]

for col in cat_vars:
    df_metrics[col] = df_metrics[col].astype("category")


# ---------------------------------------
# 5.2) Hilfsvariablen für robustere Subtree-Aggregationen
# ---------------------------------------

# Überdurchschnittlich großer Subtree:
# > 1 bedeutet größer als der Durchschnitt auf derselben Ebene
df_metrics["subtree_above_avg"] = (
    df_metrics["relative_subtree_size"] > 1
).astype(int)

df_metrics["subtree_above_avg_log"] = (
    df_metrics["relative_subtree_size_log"] > np.log1p(1)
).astype(int)

# Subtree-Metriken nur für innere Knoten, weil Blätter die Werte dominieren
# Für Blätter werden die Werte auf NaN gesetzt, damit sie bei mean/median/std ignoriert werden.
df_metrics["rel_subtree_size_internal"] = np.where(
    df_metrics["is_leaf"] == 0,
    df_metrics["relative_subtree_size"],
    np.nan
)

df_metrics["rel_subtree_size_log_internal"] = np.where(
    df_metrics["is_leaf"] == 0,
    df_metrics["relative_subtree_size_log"],
    np.nan
)


# ---------------------------------------
# 5.3) Aggregation numerischer Metriken auf Aufenthaltsebene
# ---------------------------------------

stay_num_agg = (
    df_metrics
    .groupby(["subject_id", "hadm_id"], as_index=False)
    .agg(
        # ----------------------------
        # Tiefe
        # ----------------------------
        depth_mean=("depth", "mean"),
        depth_median=("depth", "median"),
        depth_std=("depth", "std"),

        # ----------------------------
        # Blattindikator
        # ----------------------------
        leaf_share=("is_leaf", "mean"),
        leaf_count=("is_leaf", "sum"),

        # Anteil innerer Knoten
        internal_node_share=("is_internal_node", "mean"),
        internal_node_count=("is_internal_node", "sum"),

        # ----------------------------
        # Anzahl direkter Geschwister
        # ----------------------------
        siblings_min=("n_direct_siblings", "min"),
        siblings_max=("n_direct_siblings", "max"),
        siblings_q25=("n_direct_siblings", lambda x: x.quantile(0.25)),
        siblings_q75=("n_direct_siblings", lambda x: x.quantile(0.75)),
        siblings_median=("n_direct_siblings", "median"),
        siblings_mean=("n_direct_siblings", "mean"),
        siblings_std=("n_direct_siblings", "std"),

        # ----------------------------
        # Relative Subtree-Größe: alle Codes
        # ----------------------------
        subtree_rel_max=("relative_subtree_size", "max"),
        subtree_rel_median=("relative_subtree_size", "median"),
        subtree_rel_mean=("relative_subtree_size", "mean"),
        subtree_rel_std=("relative_subtree_size", "std"),

        # Relative Subtree-Größe log: alle Codes
        subtree_rel_log_max=("relative_subtree_size_log", "max"),
        subtree_rel_log_median=("relative_subtree_size_log", "median"),
        subtree_rel_log_mean=("relative_subtree_size_log", "mean"),
        subtree_rel_log_std=("relative_subtree_size_log", "std"),

        # ----------------------------
        # Relative Subtree-Größe: nur innere Knoten
        # ----------------------------
        subtree_rel_internal_min=("rel_subtree_size_internal", "min"),
        subtree_rel_internal_max=("rel_subtree_size_internal", "max"),
        subtree_rel_internal_median=("rel_subtree_size_internal", "median"),
        subtree_rel_internal_mean=("rel_subtree_size_internal", "mean"),
        subtree_rel_internal_std=("rel_subtree_size_internal", "std"),

        # Relative Subtree-Größe log: nur innere Knoten
        subtree_rel_log_internal_min=("rel_subtree_size_log_internal", "min"),
        subtree_rel_log_internal_max=("rel_subtree_size_log_internal", "max"),
        subtree_rel_log_internal_median=("rel_subtree_size_log_internal", "median"),
        subtree_rel_log_internal_mean=("rel_subtree_size_log_internal", "mean"),
        subtree_rel_log_internal_std=("rel_subtree_size_log_internal", "std"),

        # ----------------------------
        # Anteile überdurchschnittlich großer Subtrees
        # ----------------------------
        subtree_above_avg_share=("subtree_above_avg", "mean"),
        subtree_above_avg_count=("subtree_above_avg", "sum"),
        
        subtree_above_avg_log_share=("subtree_above_avg_log", "mean"),
        subtree_above_avg_log_count=("subtree_above_avg_log", "sum"),
    )
)

# Standardabweichung bei nur einem Wert ist NaN.
# Das kann sinnvoll auf 0 gesetzt werden.
std_cols = [col for col in stay_num_agg.columns if col.endswith("_std")]
stay_num_agg[std_cols] = stay_num_agg[std_cols].fillna(0)

# Für Aufenthalte ohne innere Knoten bleiben internal-Min/Max/Mean/etc. NaN.
# Das ist methodisch besser als 0, weil 0 eine echte Ausprägung wäre.

print(stay_num_agg[std_cols].head(10))


# ---------------------------------------
# 5.4) Kategorieanteile berechnen
# ---------------------------------------
cat_vars = [
    "rel_subtree_log_cat",
    "siblings_3cat",
    "depth_ord"
]

# Ausgangspunkt ist die numerische Aggregation
stay_metric_agg = stay_num_agg.copy()

# Gesamtanzahl Diagnosecodes pro Aufenthalt einmal berechnen
stay_totals = (
    df_metrics
    .groupby(["subject_id", "hadm_id"], observed=True)
    .size()
    .rename("total_codes_for_cat_share")
    .reset_index()
)

for var in cat_vars:
    print(f"Verarbeite Kategorievariable: {var}")

    # Falls category dtype: nicht verwendete Kategorien entfernen
    if isinstance(df_metrics[var].dtype, CategoricalDtype):
        df_metrics[var] = df_metrics[var].cat.remove_unused_categories()

    # Nur benötigte Spalten verwenden
    tmp = df_metrics[["subject_id", "hadm_id", var]].dropna(subset=[var]).copy()

    # Zählen je Aufenthalt und Kategorie
    tmp = (
        tmp
        .groupby(["subject_id", "hadm_id", var], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )

    # Gesamtzahl Codes je Aufenthalt ergänzen
    tmp = tmp.merge(
        stay_totals,
        on=["subject_id", "hadm_id"],
        how="left"
    )

    # Anteil berechnen
    tmp["share"] = tmp["count"] / tmp["total_codes_for_cat_share"]

    # Von long nach wide
    tmp = tmp.pivot(
        index=["subject_id", "hadm_id"],
        columns=var,
        values="share"
    ).fillna(0)

    # Spaltennamen bereinigen
    tmp.columns = [f"{var}_share_{str(c)}" for c in tmp.columns]
    tmp = tmp.reset_index()

    # Direkt an stay_metric_agg mergen
    stay_metric_agg = stay_metric_agg.merge(
        tmp,
        on=["subject_id", "hadm_id"],
        how="left"
    )

    # Speicher freigeben
    del tmp
    gc.collect()

# Fehlende Kategorieanteile auf 0 setzen
share_cols = [col for col in stay_metric_agg.columns if "_share_" in col]
stay_metric_agg[share_cols] = stay_metric_agg[share_cols].fillna(0)


# ---------------------------------------
# 5.5) Merge der Prävalenz-/IC- und Diagnosemetriken
# ---------------------------------------

metric_df = prevalence_ic_agg.merge(
    stay_metric_agg,
    on=["subject_id", "hadm_id"],
    how="left"
)

# Kontrolle
print("Shape aggregierter Metrik-Datensatz:", metric_df.shape)
print("Eindeutige Aufenthalte:", metric_df["hadm_id"].nunique())
print("Duplikate pro hadm_id:", metric_df.duplicated("hadm_id").sum())


# ---------------------------------------
# 5.6) Analyse der aggregierten Metriken
# ---------------------------------------

id_cols = ["subject_id", "hadm_id"]
metric_cols = [c for c in metric_df.columns if c not in id_cols]

# Nur numerische Metriken untersuchen
num_cols = metric_df[metric_cols].select_dtypes(include=[np.number]).columns.tolist()

# ---------------------------------------
# Übersicht: Missing Values, Unique Values, Varianz
# ---------------------------------------

metric_summary = pd.DataFrame({
    "dtype": metric_df[num_cols].dtypes.astype(str),
    "n_missing": metric_df[num_cols].isna().sum(),
    "missing_share": metric_df[num_cols].isna().mean(),
    "n_unique": metric_df[num_cols].nunique(dropna=True),
    "mean": metric_df[num_cols].mean(),
    "std": metric_df[num_cols].std(),
    "min": metric_df[num_cols].min(),
    "q01": metric_df[num_cols].quantile(0.01),
    "q05": metric_df[num_cols].quantile(0.05),
    "q25": metric_df[num_cols].quantile(0.25),
    "median": metric_df[num_cols].median(),
    "q75": metric_df[num_cols].quantile(0.75),
    "q95": metric_df[num_cols].quantile(0.95),
    "q99": metric_df[num_cols].quantile(0.99),
    "max": metric_df[num_cols].max()
}).reset_index(names="variable")

metric_summary["zero_share"] = [
    (metric_df[col] == 0).mean() for col in metric_summary["variable"]
]

metric_summary["near_constant"] = (
    (metric_summary["n_unique"] <= 2) |
    (metric_summary["std"].fillna(0) == 0) |
    (metric_summary["zero_share"] > 0.95)
)

metric_summary = metric_summary.sort_values(
    ["near_constant", "missing_share", "zero_share"],
    ascending=[False, False, False]
)

metric_summary

# ---------------------------------------
# Vorschlagsliste: Kandidaten zum Entfernen
# ---------------------------------------

candidate_drop_vars = metric_summary.loc[
    (metric_summary["near_constant"]) |
    (metric_summary["n_unique"] <= 2) |
    (metric_summary["std"].fillna(0) == 0) |
    (metric_summary["zero_share"] > 0.95) |
    (metric_summary["missing_share"] > 0.50),
    "variable"
].tolist()

print(candidate_drop_vars)


# ---------------------------------------
# 5.7) Reduzierte Version des Feature-Sets erstellen
# ---------------------------------------

metric_df_reduced = metric_df.copy()

drop_cols = [

    # Prävalenz (gesamt): redundant zu IC
    "prevalence_diag_mean",
    "prevalence_diag_min",
    "prevalence_diag_max",
    "prevalence_diag_std",
    "prevalence_proc_mean",
    "prevalence_proc_min",
    "prevalence_proc_max",
    "prevalence_proc_std",

    # Subtree nicht-logarithmiert (gesamt): redundant zur log-Version
    "subtree_rel_max",
    "subtree_rel_median",
    "subtree_rel_mean",
    "subtree_rel_std",
    "subtree_rel_internal_min",
    "subtree_rel_internal_max",
    "subtree_rel_internal_median",
    "subtree_rel_internal_mean",
    "subtree_rel_internal_std",
    "subtree_above_avg_share",
    "subtree_above_avg_count",

    # Subtree logarithmiert: teilweise hoher Anteil an Nullen
    "subtree_rel_log_median",
    "rel_subtree_log_cat_share_zero",
    "rel_subtree_log_cat_share_nonzero",
    "subtree_rel_log_internal_min",
    "subtree_rel_log_internal_max",
    "subtree_rel_log_internal_median",
    "subtree_rel_log_internal_std",
    "subtree_above_avg_log_share",
    "subtree_above_avg_log_count",
    
    # Tiefe
    "depth_median",         # redundant zum Mittelwert, der feinere Unterschiede aufzeigt

    # Geschwisterzahl
    "siblings_min",       # hoher Anteil an Nullen; q25 ist besser

    # Innere Knoten: redundant zum Blattindikator
    "internal_node_share",
    "internal_node_count"
]

# Nur existierende Spalten entfernen
drop_cols_existing = [c for c in drop_cols if c in metric_df_reduced.columns]

metric_df_reduced = metric_df_reduced.drop(columns=drop_cols_existing)

# Kontrolle
print(f"Original shape: {metric_df.shape}")
print(f"Reduzierte shape: {metric_df_reduced.shape}")


# ---------------------------------------
# 5.8) Korrelationen prüfen
# ---------------------------------------

id_cols = ["subject_id", "hadm_id"]
metric_cols = [c for c in metric_df_reduced.columns if c not in id_cols]

# Nur numerische Metriken untersuchen
num_cols = metric_df_reduced[metric_cols].select_dtypes(include=[np.number]).columns.tolist()

corr = metric_df_reduced[num_cols].corr(method="spearman")

# Hoch korrelierte Variablenpaare extrahieren
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)

corr_pairs.columns = ["var1", "var2", "spearman_corr"]
corr_pairs["abs_corr"] = corr_pairs["spearman_corr"].abs()

high_corr_pairs = corr_pairs[
    corr_pairs["abs_corr"] >= 0.90
].sort_values("abs_corr", ascending=False)

# subtree_rel_log_mean, -max und -std sind untereinander stark korreliert, 
# was darauf hindeutet, dass sie ähnliche Information tragen und redundant sind.
# Es wird hier noch keine der drei Metriken final entfernt.


# ============================================================
# 6) Merge vom Basisdatensatz und den aggregierten Metriken
# ============================================================

df_stay_base_and_agg_metrics = stay_level.merge(
    metric_df_reduced,
    on=["subject_id", "hadm_id"],
    how="left"
)

# Merge prüfen
print(df_stay_base_and_agg_metrics.columns)

print("Shape finales DataFrame:")
print(df_stay_base_and_agg_metrics.shape)

print("\nAnzahl eindeutiger Aufenthalte:")
print(
    df_stay_base_and_agg_metrics[["subject_id", "hadm_id"]]
    .drop_duplicates()
    .shape[0]
)

# Prüfen auf doppelte Aufenthalte
duplicates = (
    df_stay_base_and_agg_metrics
    .duplicated(subset=["subject_id", "hadm_id"])
    .sum()
)
print("\nAnzahl doppelter Aufenthalte:")
print(duplicates)

# ---------------------------------------
# 6.1) Datensatz auf Aufenthaltsebene vor Encoding exportieren
# ---------------------------------------
# Enthält alle bereits berechneten Merkmale auf Aufenthaltsebene.
# Dient als gemeinsamer Ausgangspunkt für Haupt- und No-Leakage-Analyse.

assert df_stay_base_and_agg_metrics["hadm_id"].is_unique, (
    "Datensatz ist nicht eindeutig auf Aufenthaltsebene."
)

df_stay_base_and_agg_metrics.to_pickle(
    path / "full_model_staylevel_pre_encoding.pkl"
)

print(
    "Aufenthaltsdatensatz vor Encoding gespeichert:",
    df_stay_base_and_agg_metrics.shape
)

# Zeitvariablen wurden nur für die spätere No-Leakage-Transformation gespeichert.
# Im Hauptmodell werden sie nicht als Prädiktoren verwendet.
df_stay_base_and_agg_metrics = df_stay_base_and_agg_metrics.drop(
    columns=["admittime", "dischtime"],
    errors="ignore"
)


# ============================================================
# 7) One-Hot Encoding von kategorialen Variablen
# ============================================================

# Übersicht über die Kategorien
print("\nValue Counts admission_type:")
print(df_stay_base_and_agg_metrics["admission_type"].value_counts(dropna=False))

print("\nValue Counts admission_location:")
print(df_stay_base_and_agg_metrics["admission_location"].value_counts(dropna=False))

# One-Hot-Encoding
df_one_hot = pd.get_dummies(
    df_stay_base_and_agg_metrics,
    columns=["admission_type", "admission_location", "gender"],
    drop_first=False,   
    dummy_na=False ,
    dtype=int     
)

print("Shape nach One-Hot Encoding:")
print(df_one_hot.shape)



# ============================================================
# 8) Seltene Codes analysieren
# - Diagnosecodes: mindestens 10 Aufenthalte
# - Behandlungscodes: mindestens 3 Aufenthalte
# ============================================================

def analyze_code_filtering(
    df,
    code_list_col,
    death_col,
    min_count,
    label
):
    # --------------------------------------------------
    # Codes auf Aufenthaltsebene zählen
    # Jeder Code zählt je Aufenthalt maximal einmal
    # --------------------------------------------------
    code_counter = Counter(
        code
        for codes in df[code_list_col]
        if isinstance(codes, list)
        for code in set(codes)
    )

    frequent_codes = {
        code
        for code, count in code_counter.items()
        if count >= min_count
    }

    n_codes_before = len(code_counter)
    n_codes_after = len(frequent_codes)
    n_codes_removed = n_codes_before - n_codes_after

    share_codes_removed = (
        n_codes_removed / n_codes_before
        if n_codes_before > 0
        else 0
    )

    # --------------------------------------------------
    # Aufenthalte ohne Codes vor der Filterung
    # --------------------------------------------------
    originally_no_codes = df[code_list_col].apply(
        lambda x: not isinstance(x, list) or len(x) == 0
    )

    # --------------------------------------------------
    # Anzahl verbleibender Codes nach der Filterung
    # --------------------------------------------------
    remaining_counts = df[code_list_col].apply(
        lambda codes: len([
            code
            for code in codes
            if code in frequent_codes
        ])
        if isinstance(codes, list)
        else 0
    )

    # Alle Aufenthalte ohne verbleibende Codes
    no_codes_after_filtering = remaining_counts.eq(0)

    # Aufenthalte, die erst durch die Filterung codefrei werden
    artificially_no_codes = (
        (~originally_no_codes)
        & no_codes_after_filtering
    )

    # --------------------------------------------------
    # Mortalitätsangaben
    # --------------------------------------------------
    total_deaths = df[death_col].sum()

    deaths_originally_no_codes = df.loc[
        originally_no_codes,
        death_col
    ].sum()

    deaths_artificially_no_codes = df.loc[
        artificially_no_codes,
        death_col
    ].sum()

    share_deaths_artificially_no_codes = (
        deaths_artificially_no_codes / total_deaths
        if total_deaths > 0
        else 0
    )

    # --------------------------------------------------
    # Ausgabe
    # --------------------------------------------------
    print("=" * 70)
    print(f"{label}codes – Mindesthäufigkeit: {min_count} Aufenthalte")
    print("=" * 70)

    print("Codes:")
    print(f"  Eindeutige Codes vor Filterung: {n_codes_before}")
    print(f"  Behaltene Codes:                {n_codes_after}")
    print(f"  Entfernte seltene Codes:        {n_codes_removed}")
    print(f"  Anteil entfernter Codes:        {share_codes_removed:.2%}")
    print()

    print("Aufenthalte:")
    print(f"  Aufenthalte insgesamt: {len(df)}")
    print(
        f"  Ursprünglich ohne {label}codes: "
        f"{originally_no_codes.sum()} "
        f"({originally_no_codes.mean():.2%})"
    )
    print(
        f"  Nach Filterung ohne verbleibende {label}codes: "
        f"{no_codes_after_filtering.sum()} "
        f"({no_codes_after_filtering.mean():.2%})"
    )
    print(
        f"  Erst durch Filterung ohne {label}codes: "
        f"{artificially_no_codes.sum()} "
        f"({artificially_no_codes.mean():.2%})"
    )
    print()

    print("Mortalität:")
    print(f"  Todesfälle insgesamt: {int(total_deaths)}")
    print(
        f"  Todesfälle bei ursprünglich {label}codefreien Aufenthalten: "
        f"{int(deaths_originally_no_codes)}"
    )
    print(
        f"  Todesfälle bei erst durch Filterung {label}codefreien Aufenthalten: "
        f"{int(deaths_artificially_no_codes)}"
    )
    print(
        f"  Anteil dieser Todesfälle an allen Todesfällen: "
        f"{share_deaths_artificially_no_codes:.2%}"
    )
    print()

    return {
        "frequent_codes": frequent_codes,
        "code_counter": code_counter,
        "originally_no_codes": originally_no_codes,
        "no_codes_after_filtering": no_codes_after_filtering,
        "artificially_no_codes": artificially_no_codes,
        "remaining_counts": remaining_counts,
        "n_codes_before": n_codes_before,
        "n_codes_after": n_codes_after,
        "n_codes_removed": n_codes_removed,
        "share_codes_removed": share_codes_removed
    }


# --------------------------------------------------
# 8.1) Analyse für beide Codearten mit eigenen Grenzwerten
# --------------------------------------------------

filter_settings = {
    "Diagnose": {
        "code_list_col": "diag_code_list",
        "min_count": 10
    },
    "Behandlung": {
        "code_list_col": "proc_code_list",
        "min_count": 3
    }
}

filter_results = {}

for label, settings in filter_settings.items():
    filter_results[label] = analyze_code_filtering(
        df=df_one_hot,
        code_list_col=settings["code_list_col"],
        death_col="hospital_expire_flag",
        min_count=settings["min_count"],
        label=label
    )

# --------------------------------------------------
# 8.2) Aufenthalte ohne verbleibende Codes jeglicher Art
# --------------------------------------------------

diag_filter_info = filter_results["Diagnose"]
treat_filter_info = filter_results["Behandlung"]

# Anzahl verbleibender Diagnose- und Behandlungscodes
remaining_diag_counts = diag_filter_info["remaining_counts"]
remaining_proc_counts = treat_filter_info["remaining_counts"]

# Nach Filterung weder Diagnose- noch Behandlungscodes vorhanden
no_codes_any_type_after_filtering = (
    remaining_diag_counts.eq(0)
    & remaining_proc_counts.eq(0)
)

# Bereits vor der Filterung vollständig codefreie Aufenthalte
originally_no_codes_any_type = (
    diag_filter_info["originally_no_codes"]
    & treat_filter_info["originally_no_codes"]
)

# Erst durch die Filterung vollständig codefrei geworden
artificially_no_codes_any_type = (
    (~originally_no_codes_any_type)
    & no_codes_any_type_after_filtering
)

total_deaths = df_one_hot["hospital_expire_flag"].sum()

deaths_no_codes_any_type = df_one_hot.loc[
    no_codes_any_type_after_filtering,
    "hospital_expire_flag"
].sum()

deaths_artificially_no_codes_any_type = df_one_hot.loc[
    artificially_no_codes_any_type,
    "hospital_expire_flag"
].sum()

print("=" * 70)
print("AUFENTHALTE OHNE VERBLEIBENDE CODES JEGLICHER ART")
print("=" * 70)

print(f"Aufenthalte insgesamt: {len(df_one_hot)}")

print(
    "Bereits ursprünglich ohne Diagnose- und Behandlungscodes: "
    f"{originally_no_codes_any_type.sum()} "
    f"({originally_no_codes_any_type.mean():.2%})"
)

print(
    "Nach Filterung ohne Diagnose- und Behandlungscodes: "
    f"{no_codes_any_type_after_filtering.sum()} "
    f"({no_codes_any_type_after_filtering.mean():.2%})"
)

print(
    "Erst durch die Filterung vollständig codefrei geworden: "
    f"{artificially_no_codes_any_type.sum()} "
    f"({artificially_no_codes_any_type.mean():.2%})"
)

print()
print(
    "Todesfälle unter allen Aufenthalten ohne verbleibende Codes: "
    f"{int(deaths_no_codes_any_type)}"
)

print(
    "Todesfälle unter den erst durch Filterung vollständig "
    f"codefreien Aufenthalten: {int(deaths_artificially_no_codes_any_type)}"
)

print(
    "Anteil dieser Todesfälle an allen Todesfällen: "
    f"{deaths_artificially_no_codes_any_type / total_deaths:.2%}"
    if total_deaths > 0
    else "Anteil dieser Todesfälle an allen Todesfällen: nicht berechenbar"
)


# ============================================================
# 9) Seltene Codes entfernen
# - Diagnosecodes: < 10 Aufenthalte
# - Behandlungscodes: < 3 Aufenthalte
# - Ausschluss nur, wenn durch die Filterung gar keine ICD-10-Codes
#   mehr vorhanden sind
# ============================================================

df_reduced = df_one_hot.copy()

diag_col = "diag_code_list"
proc_col = "proc_code_list"
death_col = "hospital_expire_flag"

# Die in Abschnitt 8 bestimmten häufigen Codes übernehmen
frequent_diag_codes = diag_filter_info["frequent_codes"]
frequent_proc_codes = treat_filter_info["frequent_codes"]

# Ausgangszustand festhalten
n_before = len(df_reduced)
deaths_before = df_reduced[death_col].sum()
n_patients_before = df_reduced["subject_id"].nunique()

# --------------------------------------------------
# 9.1) Seltene Codes aus den Listen entfernen
# --------------------------------------------------
df_reduced[diag_col] = df_reduced[diag_col].apply(
    lambda codes: [
        code for code in codes
        if code in frequent_diag_codes
    ]
    if isinstance(codes, list)
    else []
)

df_reduced[proc_col] = df_reduced[proc_col].apply(
    lambda codes: [
        code for code in codes
        if code in frequent_proc_codes
    ]
    if isinstance(codes, list)
    else []
)

# --------------------------------------------------
# 9.2) Nur Aufenthalte löschen, die erst durch die Filterung
#      vollständig ohne Diagnose- und Behandlungscodes sind
# --------------------------------------------------
n_artificially_no_codes_any_type = (
    artificially_no_codes_any_type.sum()
)

deaths_artificially_no_codes_any_type = df_reduced.loc[
    artificially_no_codes_any_type,
    death_col
].sum()

df_reduced = df_reduced.loc[
    ~artificially_no_codes_any_type
].copy()

# --------------------------------------------------
# 9.3) Checks nach Filterung und Löschung
# --------------------------------------------------
n_after = len(df_reduced)
deaths_after = df_reduced[death_col].sum()
n_patients_after = df_reduced["subject_id"].nunique()

diag_counts_after = Counter(
    code
    for codes in df_reduced[diag_col]
    for code in set(codes)
)

proc_counts_after = Counter(
    code
    for codes in df_reduced[proc_col]
    for code in set(codes)
)

print()
print("=" * 70)
print("CHECKS NACH FILTERUNG UND LÖSCHUNG")
print("=" * 70)

print(f"Aufenthalte vorher: {n_before}")
print(f"Aufenthalte nachher: {n_after}")
print(f"Gelöschte Aufenthalte: {n_before - n_after}")
print(
    "Erwartet gelöschte Aufenthalte: "
    f"{n_artificially_no_codes_any_type}"
)

print()

print(f"Todesfälle vorher: {int(deaths_before)}")
print(f"Todesfälle nachher: {int(deaths_after)}")
print(f"Gelöschte Todesfälle: {int(deaths_before - deaths_after)}")
print(
    "Erwartet gelöschte Todesfälle: "
    f"{int(deaths_artificially_no_codes_any_type)}"
)

print()

print(f"Eindeutige Patienten vorher: {n_patients_before}")
print(f"Eindeutige Patienten nachher: {n_patients_after}")
print(f"Vollständig entfernte Patienten: {n_patients_before - n_patients_after}")

print()

print(f"Unique Diagnosecodes nachher: {len(diag_counts_after)}")
print(f"Erwartete Diagnosecodes: {len(frequent_diag_codes)}")

print(f"Unique Behandlungscodes nachher: {len(proc_counts_after)}")
print(f"Erwartete Behandlungscodes: {len(frequent_proc_codes)}")

# --------------------------------------------------
# 9.4) Sicherheitschecks
# --------------------------------------------------
assert n_before - n_after == n_artificially_no_codes_any_type

assert (deaths_before - deaths_after == deaths_artificially_no_codes_any_type)

assert set(diag_counts_after).issubset(frequent_diag_codes)
assert set(proc_counts_after).issubset(frequent_proc_codes)

# Nach der Löschung darf kein Aufenthalt vollständig codefrei sein
assert not (
    df_reduced[diag_col].apply(len).eq(0)
    & df_reduced[proc_col].apply(len).eq(0)
).any()

# --------------------------------------------------
# 9.5) Verwendete Codes speichern für den No-Leakage-Ansatz
# --------------------------------------------------

final_hadm_ids = df_reduced["hadm_id"].to_numpy()

np.save(
    path / "final_hadm_ids.npy",
    final_hadm_ids
)

with open(path / "frequent_diag_codes.pkl", "wb") as f:
    pickle.dump(frequent_diag_codes, f)

with open(path / "frequent_proc_codes.pkl", "wb") as f:
    pickle.dump(frequent_proc_codes, f)


# ============================================================
# 10) Deskriptive Statistik der Modellvariablen
# ============================================================

# IDs, Zielvariable und ICD-10-Codelisten ausschließen
exclude_cols = [
    "subject_id",
    "hadm_id",
    "hospital_expire_flag",
    "diag_code_list",
    "proc_code_list",
    "icd10_chapter_list",
    "prev_icd10_chapters"
]

# Nur numerische Variablen
numeric_cols = [
    col for col in df_reduced.select_dtypes(include=[np.number]).columns
    if col not in exclude_cols
]

# --------------------------------------------------
# 10.1) Deskriptive Statistik
# --------------------------------------------------

descriptive_stats = (
    df_reduced[numeric_cols]
    .describe(percentiles=[0.25, 0.50, 0.75])
    .T[
        ["min", "25%", "50%", "mean", "75%", "max"]
    ]
)

print("\nDeskriptive Statistik der Modellvariablen:")
print(descriptive_stats)

# Optional speichern
descriptive_stats.to_csv(
    path / "descriptive_statistics_model_variables.csv"
)


# --------------------------------------------------
# 10.2) Verteilungen nicht-binärer Variablen
# --------------------------------------------------

plot_cols = [
    col for col in numeric_cols
    if df_reduced[col].dropna().nunique() > 2
]

n_cols = 3
n_rows = int(np.ceil(len(plot_cols) / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(15, 3.5 * n_rows)
)

axes = np.array(axes).reshape(-1)

for ax, col in zip(axes, plot_cols):

    values = df_reduced[col].dropna()

    # Diskrete ganzzahlige Variable mit überschaubar vielen Ausprägungen
    is_discrete = (
        np.all(np.isclose(values, values.astype(int)))
        and values.nunique() <= 30
    )

    if is_discrete:
        counts = values.value_counts().sort_index()

        ax.bar(
            counts.index,
            counts.values
        )

    else:
        ax.hist(
            values,
            bins=30,
            edgecolor="black"
        )

    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Anzahl Aufenthalte")

# Nicht verwendete Teilplots entfernen
for ax in axes[len(plot_cols):]:
    ax.remove()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10.3) Verteilung der One-Hot-kodierten kategorialen Variablen
# --------------------------------------------------

one_hot_prefixes = [
    "admission_type_",
    "admission_location_",
    "gender_"
]

one_hot_cols = [
    col for col in df_reduced.columns
    if col.startswith(tuple(one_hot_prefixes))
]

one_hot_distribution = pd.DataFrame({
    "variable": one_hot_cols,
    "n": [df_reduced[col].sum() for col in one_hot_cols],
    "percent": [
        df_reduced[col].mean() * 100
        for col in one_hot_cols
    ]
})

print(one_hot_distribution)


# ============================================================
# 11) Dataframe für Baseline ohne ICD-10-Codes
# ============================================================

df_no_icd10 = df_reduced.copy()     # Datensatz, wo 35 Aufenthalte ohne Codes gelöscht wurden

selected_columns = [
    'hadm_id',
    'subject_id',
    'hospital_expire_flag',
    'age_at_admission',

    # admission_type
    'admission_type_AMBULATORY OBSERVATION',
    'admission_type_DIRECT EMER.',
    'admission_type_DIRECT OBSERVATION',
    'admission_type_ELECTIVE',
    'admission_type_EU OBSERVATION',
    'admission_type_EW EMER.',
    'admission_type_OBSERVATION ADMIT',
    'admission_type_SURGICAL SAME DAY ADMISSION',
    'admission_type_URGENT',

    # admission_location
    'admission_location_AMBULATORY SURGERY TRANSFER',
    'admission_location_CLINIC REFERRAL',
    'admission_location_EMERGENCY ROOM',
    'admission_location_INTERNAL TRANSFER TO OR FROM PSYCH',
    'admission_location_INFORMATION NOT AVAILABLE',
    'admission_location_PACU',
    'admission_location_PHYSICIAN REFERRAL',
    'admission_location_PROCEDURE SITE',
    'admission_location_TRANSFER FROM HOSPITAL',
    'admission_location_TRANSFER FROM SKILLED NURSING FACILITY',
    'admission_location_WALK-IN/SELF REFERRAL',

    # gender
    'gender_F',
    'gender_M',
    
    # Patientenhistorie
    'n_prev_admissions', 
    'n_prev_emergency_admissions', 
    'time_since_last_discharge_days',
    'had_admission_last_30d', 
    'max_past_los', 
    'avg_past_los'
]

# Basisdatensatz erstellen
df_no_icd10 = df_no_icd10[selected_columns].copy()

# Kontrolle
print(df_no_icd10.shape)
print(df_no_icd10.head())



# Datensatz exportieren
df_no_icd10.to_csv(path / "baseline_no_icd10_reduced.csv", index=False)

# ============================================================
# 12) Multi-Hot-Encoding
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

# --------------------------------------------------
# 12.1) Baseline-Matrix erzeugen, die nur die Basisvariablen enthält
# --------------------------------------------------

baseline_drop_cols = [
    "icd10_chapter_list",
    "n_diag_codes",
    "n_proc_codes",
    "n_diag_chapters",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters",
    "prev_icd10_chapters",
    "ic_diag_mean",
    "ic_diag_min",
    "ic_diag_max",
    "ic_diag_std",
    "ic_proc_mean",
    "ic_proc_min",
    "ic_proc_max",
    "ic_proc_std",
    "depth_mean",
    "depth_std",
    "leaf_share",
    "leaf_count",
    "siblings_max",
    "siblings_q25",
    "siblings_q75",
    "siblings_median",
    "siblings_mean",
    "siblings_std",
    "subtree_rel_log_max",
    "subtree_rel_log_mean",
    "subtree_rel_log_std",
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

df_baseline = df_reduced.drop(
    columns=baseline_drop_cols,
    errors="ignore"
)

baseline_list_columns = {
    "diag_code_list": "diag_code",
    "proc_code_list": "proc_code"
}

baseline_dataset = create_sparse_matrix(
    df=df_baseline,
    list_columns=baseline_list_columns
)

# Exportieren
save_sparse_dataset(
    dataset=baseline_dataset,
    output_path=path / "baseline_model"
)


# --------------------------------------------------
# 12.2) Matrix erzeugen, die alle Variablen enthält
# --------------------------------------------------

full_list_columns = {
    "diag_code_list": "diag_code",
    "proc_code_list": "proc_code",
    "icd10_chapter_list": "icd10_chapter",
    "prev_icd10_chapters": "prev_icd10_chapter"
}

full_dataset = create_sparse_matrix(
    df=df_reduced,
    list_columns=full_list_columns
)

# Exportieren
save_sparse_dataset(
    dataset=full_dataset,
    output_path=path / "full_model"
)



# ============================================================
# 13) Finale Checks
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

    # -----------------------------------------------------
    # 1) Grundchecks
    # -----------------------------------------------------

    assert X_sparse.shape[0] == df_reference.shape[0], (
        f"{dataset_name}: X_sparse und df_reference haben unterschiedlich viele Zeilen."
    )

    assert X_sparse.shape[1] == len(feature_names), (
        f"{dataset_name}: Anzahl Feature-Namen passt nicht zur Matrix."
    )

    print("Shape:", X_sparse.shape)
    print("Anzahl Features:", len(feature_names))
    print("List Columns:", list_columns)

    # -----------------------------------------------------
    # 2) Prüfen, ob Multi-Hot-Features nur 0/1 enthalten
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 3) Beispiel-Aufenthalte anzeigen
    # -----------------------------------------------------

    for example_idx in example_indices:

        if example_idx >= X_sparse.shape[0]:
            print(f"\nBeispielindex {example_idx} übersprungen.")
            continue

        row = X_sparse[example_idx]

        active_feature_indices = row.indices

        active_features = [
            feature_names[i]
            for i in active_feature_indices
        ]

        print("\n" + "-" * 80)
        print(f"Beispiel-Aufenthalt Index: {example_idx}")

        if "hadm_id" in df_reference.columns:
            print("HADM ID:")
            print(df_reference.iloc[example_idx]["hadm_id"])

        print("\nAktive Features:")
        print(active_features)

        # -------------------------------------------------
        # 4) Original-Codes vs aktive Multi-Hot-Features
        # -------------------------------------------------

        for col, prefix in list_columns.items():

            original_codes = df_reference.iloc[example_idx][col]

            if not isinstance(original_codes, list):
                original_codes = []

            expected_features = [
                f"{prefix}_{code}"
                for code in original_codes
            ]

            active_features_for_col = [
                f for f in active_features
                if f.startswith(f"{prefix}_")
            ]

            missing = set(expected_features) - set(active_features_for_col)

            unexpected = (
                set(active_features_for_col)
                - set(expected_features)
            )

            print("\n", col)
            print("Prefix:")
            print(prefix)

            print("Original Codes:")
            print(original_codes)

            print("Erwartete Multi-Hot Features:")
            print(expected_features)

            print("Aktive Multi-Hot Features:")
            print(active_features_for_col)

            print("Fehlend:", missing)
            print("Unerwartet:", unexpected)

            assert len(missing) == 0, (
                f"{dataset_name}: Fehlende Features "
                f"bei Index {example_idx}, Spalte {col}: {missing}"
            )

            assert len(unexpected) == 0, (
                f"{dataset_name}: Unerwartete Features "
                f"bei Index {example_idx}, Spalte {col}: {unexpected}"
            )

    # -----------------------------------------------------
    # 5) Zufällige Stichproben-Checks
    # -----------------------------------------------------

    rng = random.Random(random_state)

    n_available = df_reference.shape[0]
    n_checks = min(n_random_checks, n_available)

    random_indices = rng.sample(range(n_available), n_checks)

    for example_idx in random_indices:

        row = X_sparse[example_idx]

        active_features = [
            feature_names[i]
            for i in row.indices
        ]

        for col, prefix in list_columns.items():

            original_codes = df_reference.iloc[example_idx][col]

            if not isinstance(original_codes, list):
                original_codes = []

            expected_features = set(
                f"{prefix}_{code}"
                for code in original_codes
            )

            active_features_for_col = set(
                f for f in active_features
                if f.startswith(f"{prefix}_")
            )

            assert expected_features == active_features_for_col, (
                f"{dataset_name}: Mismatch "
                f"bei Index {example_idx}, Spalte {col}. "
                f"Fehlend: "
                f"{expected_features - active_features_for_col}. "
                f"Unerwartet: "
                f"{active_features_for_col - expected_features}."
            )

    print("\nAlle Stichproben-Checks bestanden.")
    print("=" * 80)


# Checks anwenden auf das gesamte Dataframe
check_sparse_dataset(
    dataset=full_dataset,
    df_reference=df_reduced,
    dataset_name="Full Model",
    example_indices=(0, 6000),
    n_random_checks=10
)

# Checks anwenden auf das Baseline-Dataset
check_sparse_dataset(
    dataset=baseline_dataset,
    df_reference=df_baseline,
    dataset_name="Baseline Model",
    example_indices=(0, 6000),
    n_random_checks=10
)


