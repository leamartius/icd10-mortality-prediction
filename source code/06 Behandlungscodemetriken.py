#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERZEUGUNG UND ANALYSE DER BEHANDLUNGSCODEMERKMALE

In dem Skript werden aus den ICD-10-PCS-Behandlungscodes semantische Merkmale auf 
Aufenthaltsebene erzeugt und exportiert, allerdings nur für Behandlungen mit Section 0.
Anschließend werden die erzeugten Merkmale deskriptiv sowie hinsichtlich ihrer Redundanz und 
ihres Zusammenhangs mit der Krankenhausmortalität untersucht. Zusätzlich werden verschiedene
PCS-Dimensionen hinsichtlich ihrer prognostischen Relevanz für die Mortalität verglichen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from xgboost import XGBClassifier

# PFAD ANPASSEN!
path = Path("Pfad_zu_zuvor_gespeicherten_Daten")
df = pd.read_csv(path / "mimic_with_semantics_categories_mortality_codelevel.csv")

# =========================================================
# 1) Nur Behandlungscodes auswählen
# =========================================================

df_proc = df[df["code_type"] == "procedure"].copy()

print("Anzahl Zeilen mit Behandlungscodes:", len(df_proc))

# =========================================================
# 2) Section-0-Datensatz für alle semantischen PCS-Metriken 
# =========================================================

# ----------------------------------------------
# 2.1) ICD-10-PCS Positionen extrahieren
# ----------------------------------------------
# ICD-10-PCS:
# Position 1 = Section
# Position 2 = Body System
# Position 3 = Root Operation
# Position 5 = Approach

df_proc["pcs_section"] = df_proc["icd_code"].str[0]
df_proc["pcs_body_system"] = df_proc["icd_code"].str[1]
df_proc["pcs_root_operation"] = df_proc["icd_code"].str[2]
df_proc["pcs_approach"] = df_proc["icd_code"].str[4]

# ----------------------------------------------
# 2.2) Alle PCS-Merkmale ausschließlich aus Section 0 (Medical and Surgical) berechnen
# ----------------------------------------------

df_proc_section0 = (
    df_proc.loc[
        df_proc["pcs_section"].eq("0")
    ]
    .copy()
)


# =========================================================
# 3) Invasivitätsgrad bestimmen
# =========================================================
# ICD-10-PCS Approach-Werte innerhalb Section 0 (Medical and Surgical):

# 0 = Open
# 3 = Percutaneous
# 4 = Percutaneous Endoscopic
# 7 = Via Natural or Artificial Opening
# 8 = Via Natural or Artificial Opening Endoscopic
# F = Via Natural or Artificial Opening with Percutaneous Endoscopic Assistance
# X = External

# Gruppierung:
# hoch      = Open
# mittel    = Percutaneous / Perc Endoscopic
# niedrig   = Natural Opening
# nicht invasiv = External

# ---------------------------------------------------------
# 3.1) Anteil von Section = 0 an allen Behandlungscodes
# ---------------------------------------------------------

# 1. Codezeilen
n_procedure_rows = len(df_proc)
n_section_0_rows = df_proc["pcs_section"].eq("0").sum()
share_section_0_rows = (n_section_0_rows / n_procedure_rows)

### Da 70.4 % der Behandlungscodes zur Section 0 gehören, ist es vertretbar, 
### die Metrik nur für diese Codes zu bestimmen

# 2. Eindeutige Aufenthalt-Code-Kombinationen
unique_stay_codes = (
    df_proc[
        ["hadm_id", "icd_code", "pcs_section"]
    ]
    .drop_duplicates(
        subset=["hadm_id", "icd_code"]
    )
)

n_unique_stay_codes = len(unique_stay_codes)
n_section_0_unique_stay_codes = (
    unique_stay_codes["pcs_section"].eq("0").sum()
)

share_section_0_unique_stay_codes = (
    n_section_0_unique_stay_codes /
    n_unique_stay_codes
)

# 3. Eindeutige Aufenthalte
n_procedure_stays = df_proc["hadm_id"].nunique()

n_section_0_stays = (
    df_proc.loc[
        df_proc["pcs_section"].eq("0"),
        "hadm_id"
    ]
    .nunique()
)

share_section_0_stays = (
    n_section_0_stays / n_procedure_stays
)

# 4. Eindeutige PCS-Codes
n_unique_procedure_codes = df_proc["icd_code"].nunique()

n_unique_section_0_codes = (
    df_proc.loc[
        df_proc["pcs_section"].eq("0"),
        "icd_code"
    ]
    .nunique()
)

share_unique_section_0_codes = (
    n_unique_section_0_codes /
    n_unique_procedure_codes
)

print("\n=== Section-0-Abdeckung ===")

print(
    f"Codezeilen mit Section 0: "
    f"{n_section_0_rows:,} von {n_procedure_rows:,} "
    f"({share_section_0_rows:.2%})"
)

print(
    f"Eindeutige Aufenthalt-Code-Kombinationen mit Section 0: "
    f"{n_section_0_unique_stay_codes:,} "
    f"von {n_unique_stay_codes:,} "
    f"({share_section_0_unique_stay_codes:.2%})"
)

print(
    f"Aufenthalte mit mindestens einer Section-0-Behandlung: "
    f"{n_section_0_stays:,} von {n_procedure_stays:,} "
    f"({share_section_0_stays:.2%})"
)

print(
    f"Eindeutige PCS-Codes mit Section 0: "
    f"{n_unique_section_0_codes:,} "
    f"von {n_unique_procedure_codes:,} "
    f"({share_unique_section_0_codes:.2%})"
)

# ---------------------------------------------------------
# 3.2) Kategorien mappen
# ---------------------------------------------------------

approach_mapping = {
    "0": "high",           # Open
    "3": "medium",         # Percutaneous
    "4": "medium",         # Percutaneous Endoscopic
    "7": "low",            # Natural/Artificial Opening
    "8": "low",            # Natural/Artificial Opening Endoscopic
    "F": "low",            # Natural Opening + Perc Endoscopic Assistance
    "X": "non_invasive"    # External
}

df_proc_section0["invasiveness_category"] = (
    df_proc_section0["pcs_approach"]
    .map(approach_mapping)
)

# ---------------------------------------------------------
# 3.3) Checks
# ---------------------------------------------------------

print("\nVerteilung Invasivitätsgrad:")
print(
    df_proc_section0["invasiveness_category"]
    .value_counts(dropna=False)
)

print("\nNicht gemappte Approach-Werte innerhalb Section 0:")
print(
    df_proc_section0.loc[
        df_proc_section0["invasiveness_category"].isna(),
        "pcs_approach"
    ]
    .value_counts(dropna=False)
)

# =========================================================
# 4) Invasivitätsmetriken auf Aufenthaltsebene
# =========================================================

# Häufigkeit je Kategorie
invasiveness_counts = (
    df_proc_section0
    .dropna(subset=["invasiveness_category"])
    .groupby(["hadm_id", "invasiveness_category"])
    .size()
    .unstack(fill_value=0)
)

# Sicherstellen, dass alle erwarteten Kategorien als Spalten existieren
expected_invasiveness_cols = [
    "high",
    "medium",
    "low",
    "non_invasive"
]

for col in expected_invasiveness_cols:
    if col not in invasiveness_counts.columns:
        invasiveness_counts[col] = 0

invasiveness_counts = invasiveness_counts[expected_invasiveness_cols]
invasiveness_counts.index.name = "hadm_id"

# Anzahl hoch invasiver Eingriffe pro Aufenthalt
n_high_invasive = invasiveness_counts["high"].rename("n_high_invasive")

# Relative Anteile
invasiveness_shares = (
    invasiveness_counts
    .div(invasiveness_counts.sum(axis=1), axis=0)
    .add_prefix("invasiveness_share_")
)

# Dominante Kategorie
dominant_invasiveness = (
    invasiveness_counts
    .idxmax(axis=1)
    .rename("dominant_invasiveness")
)


# =========================================================
# 5) Organsystem-Diversität
# =========================================================
# Anzahl unterschiedlicher Body Systems pro Aufenthalt

body_system_diversity = (
    df_proc_section0
    .dropna(subset=["pcs_body_system"])
    .groupby("hadm_id")["pcs_body_system"]
    .nunique()
    .rename("body_system_diversity")
)

body_system_diversity.index.name = "hadm_id"

print("\nAnzahl Aufenthalte mit Behandlungscodes:", df_proc["hadm_id"].nunique())
print(
    "\nAnzahl Aufenthalte mit mindestens einer Section-0-Behandlung:",
    df_proc_section0["hadm_id"].nunique()
)
print("Anzahl Zeilen body_system_diversity:", len(body_system_diversity))
print("Index eindeutig:", body_system_diversity.index.is_unique)
print(
    "Alle Section-0-Aufenthalte enthalten:",
    len(body_system_diversity)
    == df_proc_section0["hadm_id"].nunique()
)


# =========================================================
# 6) Interventionsdiversität
# =========================================================
# Anzahl unterschiedlicher Root Operations pro Aufenthalt

intervention_diversity = (
    df_proc_section0
    .dropna(subset=["pcs_root_operation"])
    .groupby("hadm_id")["pcs_root_operation"]
    .nunique()
    .rename("intervention_diversity")
)

intervention_diversity.index.name = "hadm_id"

print("\nAnzahl Zeilen intervention_diversity:", len(intervention_diversity))
print("Index eindeutig:", intervention_diversity.index.is_unique)


# =========================================================
# 7) Alles zusammenführen
# =========================================================

# Grundgerüst:
# Alle Aufenthalte mit mindestens einem Behandlungscode bleiben erhalten.
# Dadurch werden auch Aufenthalte berücksichtigt, die zwar
# Behandlungscodes haben, aber keine Section-0-Prozedur.

proc_metrics = (
    df_proc[["hadm_id"]]
    .drop_duplicates()
    .set_index("hadm_id")
)

# Section-0-basierte Metriken hinzufügen
proc_metrics = proc_metrics.join(
    [
        body_system_diversity,
        intervention_diversity,
        invasiveness_shares,
        dominant_invasiveness,
        n_high_invasive
    ],
    how="left"
)

proc_metrics = proc_metrics.reset_index()

# ---------------------------------------------------------
# 7.1) Kennzeichnen, ob mindestens eine Section-0-Prozedur vorliegt
# ---------------------------------------------------------

section0_hadm_ids = set(
    df_proc_section0["hadm_id"].unique()
)

proc_metrics["has_section_0_procedure"] = (
    proc_metrics["hadm_id"]
    .isin(section0_hadm_ids)
    .astype(int)
)


# ---------------------------------------------------------
# 7.2) Aufenthalte ohne Section-0-Prozedur
# ---------------------------------------------------------
# Die Section-0-basierten Metriken bleiben bewusst NaN.
#
# Begründung:
# Für diese Aufenthalte sind Body-System-Diversität,
# Interventionsdiversität und Invasivitätsmetriken
# nicht definiert.
#
# Ein Auffüllen mit 0 würde bei der deskriptiven Analyse
# so wirken, als läge tatsächlich eine Diversität von 0
# bzw. ein Anteil von 0 vor und könnte die Verteilungen
# künstlich verzerren.

section0_metric_cols = [
    "body_system_diversity",
    "intervention_diversity",
    "invasiveness_share_high",
    "invasiveness_share_medium",
    "invasiveness_share_low",
    "invasiveness_share_non_invasive",
    "n_high_invasive"
]

print(
    "\nAnzahl Aufenthalte ohne Section-0-Prozedur:",
    proc_metrics["has_section_0_procedure"].eq(0).sum()
)

print(
    "\nFehlende Werte in Section-0-Metriken:"
)

print(
    proc_metrics[
        section0_metric_cols
    ]
    .isna()
    .sum()
)

# ---------------------------------------------------------
# 7.3) Dominante Invasivitätskategorie
# ---------------------------------------------------------
# Wenn keine Section-0-Prozedur vorliegt, ist auch keine
# dominante Invasivitätskategorie definiert.
# Deshalb bleibt dominant_invasiveness ebenfalls NaN.

print(
    "\nFehlende Werte dominant_invasiveness:",
    proc_metrics["dominant_invasiveness"]
    .isna()
    .sum()
)

# ---------------------------------------------------------
# 7.4) Konsistenzcheck
# ---------------------------------------------------------

# Aufenthalte mit Section 0 sollten grundsätzlich
# berechnete Diversitätswerte besitzen.

print(
    "\nNaN bei body_system_diversity trotz Section 0:",
    proc_metrics.loc[
        proc_metrics["has_section_0_procedure"].eq(1),
        "body_system_diversity"
    ]
    .isna()
    .sum()
)

print(
    "NaN bei intervention_diversity trotz Section 0:",
    proc_metrics.loc[
        proc_metrics["has_section_0_procedure"].eq(1),
        "intervention_diversity"
    ]
    .isna()
    .sum()
)


# =========================================================
# 8) Checks
# =========================================================

share_cols = [
    "invasiveness_share_high",
    "invasiveness_share_medium",
    "invasiveness_share_low",
    "invasiveness_share_non_invasive"
]

print("\n=== Ergebnis ===")
print(proc_metrics.shape)

print("\nSpalten:")
print(proc_metrics.columns.tolist())

print("\nFehlende Werte:")
print(proc_metrics.isna().sum())

# Prüfen: Summieren sich die Shares zu 1?
proc_metrics["share_sum"] = (
    proc_metrics[share_cols]
    .sum(axis=1, min_count=1)
)

share_sum_is_one = np.isclose(
    proc_metrics["share_sum"],
    1,
    equal_nan=False
)

print(
    "Share-Summe ungefähr 1:",
    share_sum_is_one.sum()
)

print(
    "NaN Share-Summe:",
    proc_metrics["share_sum"].isna().sum()
)

print(
    "Andere Share-Summen:",
    (
        proc_metrics["share_sum"].notna()
        & ~share_sum_is_one
    ).sum()
)

# Prüfvariable wieder entfernen
proc_metrics = proc_metrics.drop(
    columns=["share_sum"]
)


# =========================================================
# 9) One-Hot-Encoding für dominant_invasiveness
# =========================================================

dominant_invasiveness_dummies = pd.get_dummies(
    proc_metrics["dominant_invasiveness"],
    prefix="dominant_invasiveness",
    dtype=int
)

# Zum DataFrame hinzufügen
proc_metrics = pd.concat(
    [
        proc_metrics,
        dominant_invasiveness_dummies
    ],
    axis=1
)

# Originalvariable entfernen
proc_metrics = proc_metrics.drop(
    columns=["dominant_invasiveness"]
)

# =========================================================
# 10) Checks
# =========================================================

print("Neue Spalten:")
print(dominant_invasiveness_dummies.columns.tolist())

print("\nShape:")
print(proc_metrics.shape)

print("\nBeispiel:")
print(
    proc_metrics[
        dominant_invasiveness_dummies.columns.tolist()
    ].head()
)


# =========================================================
# 11) dominant_invasiveness Variablen entfernen (redundant, siehe Analyse unten)
# =========================================================

# Vollständige Version für die nachgelagerte Analyse sichern
proc_metrics_analysis = proc_metrics.copy()

dominant_cols = [
    col
    for col in proc_metrics.columns
    if col.startswith("dominant_invasiveness_")
]

proc_metrics = proc_metrics.drop(columns=dominant_cols)

print("Entfernte Variablen:")
print(dominant_cols)

# =========================================================
# 12) Body System als Multi-Hot-Feature generieren (als Äquivalent zu Diagnosekapiteln)
# =========================================================

body_system_per_stay = (
    df_proc_section0
    .dropna(subset=["pcs_body_system"])
    .groupby("hadm_id")["pcs_body_system"]
    .apply(
        lambda x:
        sorted(x.astype(str).unique())
    )
    .reset_index(
        name="pcs_body_system_list"
    )
)

# Multi-Hot-Encoding
mlb_body = MultiLabelBinarizer()

body_system_encoded = pd.DataFrame(
    mlb_body.fit_transform(
        body_system_per_stay[
            "pcs_body_system_list"
        ]
    ),
    columns=[
        f"pcs_body_system_{x}"
        for x in mlb_body.classes_
    ],
    index=body_system_per_stay.index
)

# hadm_id wieder anhängen
body_system_encoded = pd.concat(
    [
        body_system_per_stay[["hadm_id"]],
        body_system_encoded
    ],
    axis=1
)

print(
    "Section-0 Body-System Features:",
    body_system_encoded.shape[1] - 1
)

print(
    "Aufenthalte mit Section-0 Body-System-Information:",
    len(body_system_encoded)
)


# =========================================================
# 13) Mit proc_metrics mergen
# =========================================================

proc_metrics = proc_metrics.merge(
    body_system_encoded,
    on="hadm_id",
    how="left"
)

# =========================================================
# 14) Fehlende Werte auffüllen
# =========================================================

body_cols = [
    col
    for col in proc_metrics.columns
    if col.startswith("pcs_body_system_")
]

proc_metrics[body_cols] = (
    proc_metrics[body_cols]
    .fillna(0)
    .astype(int)
)

# =========================================================
# 15) Finale Checks
# =========================================================

print("\nShape:", proc_metrics.shape)

print("\nAnzahl Body-System-Features:")
print(len(body_cols))

print(proc_metrics.columns)



# =========================================================
# 16) Export
# =========================================================

proc_metrics.to_csv(path / "procedure_metrics.csv", index=False)









##############################################################################
#### ANALYSE DER BEHANDLUNGSCODEMETRIKEN
##############################################################################

# Dieser Abschnitt ist von der Main-Pipeline getrennt.
# Für die Analyse wird proc_metrics_analysis verwendet.
# Dieser DataFrame enthält auch die später wegen Redundanz
# entfernten dominant_invasiveness-Variablen.

# ============================================================
# 1) Behandlungscodelisten pro Aufenthalt erzeugen
# ============================================================
# Vorbereitung einer manuellen Plausibilitätsprüfung:
# Die Behandlungscodes werden pro Aufenthalt
# zusammengefasst, damit sie später mit den berechneten
# Behandlungsmetriken verglichen werden können.

proc_codes_per_stay = (
    df_proc
    .dropna(subset=["icd_code"])
    .groupby(["subject_id", "hadm_id"])["icd_code"]
    .apply(lambda x: sorted(x.astype(str).unique()))
    .reset_index(name="proc_code_list")
)


# ============================================================
# 2) Eindeutigen Datensatz auf Aufenthaltsebene erzeugen
# ============================================================

stay_analysis = (
    df[
        [
            "subject_id",
            "hadm_id",
            "hospital_expire_flag"
        ]
    ]
    .drop_duplicates(subset=["hadm_id"])
    .copy()
)

print("\n=== Aufenthaltsdatensatz ===")
print("Anzahl Aufenthalte:", len(stay_analysis))
print(
    "hadm_id eindeutig:",
    stay_analysis["hadm_id"].is_unique
)


# ============================================================
# 3) Behandlungscodelisten und vollständige Behandlungsmetriken mergen
# ============================================================

stay_analysis = stay_analysis.merge(
    proc_codes_per_stay,
    on=["subject_id", "hadm_id"],
    how="left"
)

stay_analysis = stay_analysis.merge(
    proc_metrics_analysis,
    on="hadm_id",
    how="left"
)

print(
    "Nach den Merges weiterhin eine Zeile pro Aufenthalt:",
    stay_analysis["hadm_id"].is_unique
)


# ============================================================
# 4) Aufenthalte ohne Behandlungscodes kennzeichnen
# ============================================================

stay_analysis["has_procedure_code"] = (
    stay_analysis["proc_code_list"]
    .apply(
        lambda x:
        isinstance(x, list) and len(x) > 0
    )
    .astype(int)
)


# ============================================================
# 5) Fehlende Werte behandeln
# ============================================================
# Aufenthalte ohne Section-0-Prozedur:
# - Section-0-basierte Metriken bleiben NaN
# - has_section_0_procedure = 0
#
# Dadurch werden nicht definierte Section-0-Metriken
# bei deskriptiven Statistiken und Korrelationen ausgeschlossen.

share_cols = [
    "invasiveness_share_high",
    "invasiveness_share_medium",
    "invasiveness_share_low",
    "invasiveness_share_non_invasive"
]

dominant_cols = [
    col
    for col in proc_metrics_analysis.columns
    if col.startswith("dominant_invasiveness_")
]

section0_metric_cols = [
    "body_system_diversity",
    "intervention_diversity",
    "n_high_invasive"
] + share_cols

# ------------------------------------------------------------
# 5.1) has_section_0_procedure
# ------------------------------------------------------------
# Aufenthalte ohne jeglichen Prozedurcode haben ebenfalls
# keine Section-0-Prozedur.

stay_analysis["has_section_0_procedure"] = (
    stay_analysis["has_section_0_procedure"]
    .fillna(0)
    .astype(int)
)

# ------------------------------------------------------------
# 5.2) Section-0-Metriken
# ------------------------------------------------------------
# Diese bleiben NaN, wenn keine Section-0-Prozedur vorhanden ist.
# Dadurch gehen solche Aufenthalte nicht als künstliche Nullwerte
# in Mittelwerte, Verteilungen und Korrelationen ein.

print(
    "\nFehlende Werte in Section-0-Metriken:"
)

print(
    stay_analysis[
        section0_metric_cols
    ]
    .isna()
    .sum()
)

# ------------------------------------------------------------
# 5.3) Dominante Invasivität
# ------------------------------------------------------------
# Auch diese ist ohne Section-0-Prozedur nicht definiert.
# pd.get_dummies() hat NaN zuvor allerdings als lauter 0 kodiert.
# Für die Analyse werden diese Fälle daher wieder auf NaN gesetzt.

stay_analysis.loc[
    stay_analysis["has_section_0_procedure"].eq(0),
    dominant_cols
] = np.nan


# ============================================================
# 6) Konsistenzprüfungen
# ============================================================

print("\n=== Konsistenzprüfungen ===")

print(
    "Anzahl Aufenthalte mit Behandlungscodes:",
    stay_analysis["has_procedure_code"].sum()
)

print("\nVerteilung has_section_0_procedure:")

print(
    stay_analysis.loc[
        stay_analysis["has_procedure_code"].eq(1),
        "has_section_0_procedure"
    ]
    .value_counts()
    .sort_index()
)

print("\nNaN-Werte in den Section-0-Metriken (bei Aufenthalten ohne Section 0 erwartbar):")

print(
    stay_analysis[
        section0_metric_cols
        + ["has_section_0_procedure"]
        + dominant_cols
    ]
    .isna()
    .sum()
)

# Kontrolle der Invasivitätsanteile
stay_analysis["share_sum_check"] = (
    stay_analysis[share_cols]
    .sum(axis=1, min_count=1)
)

shares_correct = (
    stay_analysis["share_sum_check"].isna()
    |
    np.isclose(
        stay_analysis["share_sum_check"],
        1
    )
)

print(
    "\nAlle Invasivitätsanteile korrekt:",
    shares_correct.all()
)

print(
    "Aufenthalte mit Anteilssumme 1:",
    np.isclose(
        stay_analysis["share_sum_check"],
        1,
        equal_nan=False
    ).sum()
)

print(
    "Aufenthalte ohne definierte Invasivitätsanteile:",
    stay_analysis["share_sum_check"]
    .isna()
    .sum()
)

# Prüfvariable wieder entfernen
stay_analysis = stay_analysis.drop(
    columns=["share_sum_check"]
)


# ============================================================
# 7) Datensatz für Aufenthalte mit Behandlungscodes
# ============================================================
# Für die deskriptive Analyse der eigentlichen
# Behandlungsmetriken werden nur Aufenthalte verwendet,
# in denen mindestens eine Section-0-Prozedur dokumentiert wurde.

procedure_stay_analysis = stay_analysis.loc[
    stay_analysis["has_section_0_procedure"].eq(1)
].copy()

print(
    "\nAnzahl analysierter Aufenthalte mit mindestens "
    "einer Section-0-Prozedur:",
    len(procedure_stay_analysis)
)


# ============================================================
# 8) Metriken für die Analysen festlegen
# ============================================================

stat_cols = [
    "body_system_diversity",
    "intervention_diversity",
    "n_high_invasive",
    "invasiveness_share_high",
    "invasiveness_share_medium",
    "invasiveness_share_low",
    "invasiveness_share_non_invasive",
    "has_section_0_procedure",
    "dominant_invasiveness_high",
    "dominant_invasiveness_low",
    "dominant_invasiveness_medium",
    "dominant_invasiveness_no_section_0_procedure",
    "dominant_invasiveness_non_invasive"
]

# Nur tatsächlich vorhandene Spalten verwenden
stat_cols = [
    col
    for col in stat_cols
    if col in procedure_stay_analysis.columns
]


# ============================================================
# 9) Deskriptive Statistiken
# ============================================================

description = (procedure_stay_analysis[stat_cols].describe())

print("\n=== Deskriptive Statistiken ===")
print(description)

# Häufigkeit der dominanten Kategorien
print("\n=== Dominante Invasivitätskategorien ===")

print(
    procedure_stay_analysis[dominant_cols]
    .sum()
)

print("\nAnteile:")

print(
    procedure_stay_analysis[dominant_cols]
    .mean()
    .mul(100)
    .round(2)
)


# Anteil der Aufenthalte mit mindestens einer Section-0-Behandlung
print("\n=== Aufenthalte mit Section-0-Behandlung ===")

print(
    procedure_stay_analysis[
        "has_section_0_procedure"
    ]
    .value_counts()
    .sort_index()
)

print(
    procedure_stay_analysis[
        "has_section_0_procedure"
    ]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# Verteilung der Diversitäts- und Count-Metriken
for col in [
    "body_system_diversity",
    "intervention_diversity",
    "n_high_invasive"
]:
    print(f"\n=== {col} ===")

    print(
        procedure_stay_analysis[col]
        .value_counts()
        .sort_index()
        .head(30)
    )


# ============================================================
# 10) Aufenthalte mit mindestens einem hoch invasiven Eingriff
# ============================================================

high_invasive_stays = (
    stay_analysis.loc[
        stay_analysis["has_section_0_procedure"].eq(1),
        "n_high_invasive"
    ] > 0
)

print(
    "\n=== Aufenthalte mit mindestens einem "
    "hoch invasiven Eingriff – alle Aufenthalte ==="
)

print(
    high_invasive_stays
    .value_counts()
    .rename(
        index={
            True: "Mindestens ein hoch invasiver Eingriff",
            False: "Kein hoch invasiver Eingriff"
        }
    )
)

print("\nAnteile in Prozent:")

print(
    high_invasive_stays
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .rename(
        index={
            True: "Mindestens ein hoch invasiver Eingriff",
            False: "Kein hoch invasiver Eingriff"
        }
    )
)


# ============================================================
# 11) Korrelation der Behandlungsmetriken untereinander
# ============================================================
# Spearman wird verwendet, da mehrere Variablen diskret,
# schief verteilt und durch viele Nullwerte gekennzeichnet sind.

# has_section_0_procedure ist innerhalb von procedure_stay_analysis
# konstant (= 1) und wird daher aus der Korrelationsanalyse ausgeschlossen.
corr_cols = [
    col
    for col in stat_cols
    if col != "has_section_0_procedure"
]

corr_matrix = (
    procedure_stay_analysis[corr_cols]
    .corr(method="spearman")
)

# Heatmap
plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={
        "label": "Spearman-Korrelation"
    }
)

plt.title(
    "Korrelationen der Behandlungsmetriken"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# body_system_diversity und intervention_diversity korrelieren relativ stark (0.73)
# *_share und dominant_*-Metriken korrelieren stark und sind redundant 
# -> dominant-Metriken entfernen, weil share-Metriken mehr Information beinhalten

# ============================================================
# 12) Zusammenhang mit Krankenhausmortalität
# ============================================================
# Die Auswertung erfolgt auf Aufenthaltsebene.
# Jeder Aufenthalt geht genau einmal in die Analyse ein.

for col in stat_cols:

    print(f"\n=== {col} ===")

    print(
        stay_analysis
        .groupby("hospital_expire_flag")[col]
        .describe()[["mean", "50%", "std"]]
    )


# ============================================================
# 13) Punktbiseriale Korrelation mit Mortalität
# ============================================================

correlation_results = []

for col in stat_cols:

    analysis_data = (
        stay_analysis[
            [
                "hospital_expire_flag",
                col
            ]
        ]
        .dropna()
    )

    # Konstante Variablen können nicht sinnvoll korreliert werden
    if analysis_data[col].nunique() < 2:
        corr = np.nan
        pval = np.nan

    else:
        corr, pval = pointbiserialr(
            analysis_data[
                "hospital_expire_flag"
            ],
            analysis_data[col]
        )

    correlation_results.append(
        {
            "metric": col,
            "correlation": corr,
            "p_value": pval
        }
    )

correlation_df = pd.DataFrame(
    correlation_results
)

correlation_df = correlation_df.sort_values(
    by="correlation",
    ascending=True
)

print(
    "\n=== Punktbiseriale Korrelationen "
    "mit Krankenhausmortalität ==="
)

print(correlation_df)


# ============================================================
# 14) Plot der Korrelationen mit Mortalität
# ============================================================

plot_df = correlation_df.dropna(
    subset=["correlation"]
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.barh(
    plot_df["metric"],
    plot_df["correlation"]
)

# Vertikale Nulllinie
ax.axvline(
    x=0,
    linewidth=1,
    color="black"
)

max_abs_corr = (
    plot_df["correlation"]
    .abs()
    .max()
)

if max_abs_corr == 0:
    max_abs_corr = 0.01

ax.set_xlim(
    -max_abs_corr * 1.35,
    max_abs_corr * 1.35
)

# Werte an den Balken anzeigen
for bar, value in zip(
    bars,
    plot_df["correlation"]
):

    y_position = (
        bar.get_y()
        + bar.get_height() / 2
    )

    if value >= 0:
        ax.text(
            value + max_abs_corr * 0.03,
            y_position,
            f"{value:.3f}",
            va="center",
            ha="left"
        )

    else:
        ax.text(
            value - max_abs_corr * 0.03,
            y_position,
            f"{value:.3f}",
            va="center",
            ha="right"
        )

ax.set_xlabel(
    "Punktbiseriale Korrelation "
    "mit Krankenhausmortalität"
)

ax.set_ylabel(
    "Behandlungsmetrik"
)

ax.set_title(
    "Zusammenhang der Behandlungsmetriken "
    "mit der Krankenhausmortalität"
)

plt.tight_layout()
plt.show()



# ============================================================
# 15) Vergleich der prognostischen Stärke der PCS-Dimensionen
# ============================================================
#
# Berücksichtigte Repräsentationen:
#
# Alle Sections:
# - Section
# - Section × Body System
# - Section × Root Operation
# - Section × Body System × Root Operation
#
# Nur innerhalb Section 0:
# - Body System
# - Root Operation
# - Body System × Root Operation
#
# Begründung:
# Die Bedeutung von Body System und Root Operation hängt von
# den vorangehenden PCS-Dimensionen ab. Werden diese Dimensionen
# ohne Section betrachtet, erfolgt die Analyse deshalb nur
# innerhalb von Section 0.
#
# Alle Feature-Sets werden dennoch auf demselben vollständigen
# Aufenthaltsdatensatz evaluiert. Aufenthalte ohne entsprechende
# PCS-Information erhalten für das jeweilige Feature-Set
# ausschließlich Nullwerte.
# ============================================================

# ------------------------------------------------------------
# 15.1) Gemeinsamer Mortalitätsdatensatz
# ------------------------------------------------------------

mortality_stay = (
    stay_analysis[
        [
            "hadm_id",
            "subject_id",
            "hospital_expire_flag"
        ]
    ]
    .copy()
)

print(
    "\nhadm_id eindeutig in mortality_stay:",
    mortality_stay["hadm_id"].is_unique
)

print(
    "Anzahl Aufenthalte:",
    len(mortality_stay)
)

print(
    mortality_stay[
        "hospital_expire_flag"
    ]
    .value_counts(dropna=False)
)

# ------------------------------------------------------------
# 15.2) Hilfsfunktion:
# Multi-Hot-Matrix aus kategorialen PCS-Ausprägungen erzeugen
# ------------------------------------------------------------

def create_multihot_matrix(
    source_df,
    value_col,
    list_col,
    feature_prefix
):

    values_per_stay = (
        source_df
        .dropna(subset=[value_col])
        .groupby("hadm_id")[value_col]
        .apply(
            lambda x:
            sorted(x.astype(str).unique())
        )
        .reset_index(name=list_col)
    )

    stay_df = (
        mortality_stay
        .merge(
            values_per_stay,
            on="hadm_id",
            how="left"
        )
    )

    stay_df[list_col] = (
        stay_df[list_col]
        .apply(
            lambda x:
            x if isinstance(x, list) else []
        )
    )

    mlb = MultiLabelBinarizer(
        sparse_output=True
    )

    X = mlb.fit_transform(
        stay_df[list_col]
    )

    feature_names = [
        f"{feature_prefix}_{x}"
        for x in mlb.classes_
    ]

    return X, feature_names


# ============================================================
# 16) Einzelne PCS-Dimensionen
# ============================================================

# ------------------------------------------------------------
# 16.1) Section
# Alle PCS-Sections
# ------------------------------------------------------------

X_section, section_feature_names = (
    create_multihot_matrix(
        source_df=df_proc,
        value_col="pcs_section",
        list_col="pcs_section_list",
        feature_prefix="pcs_section"
    )
)

print(
    "\nPCS Section:",
    X_section.shape
)

# ------------------------------------------------------------
# 16.2) Body System
# Nur Section 0
# ------------------------------------------------------------

X_body, body_feature_names = (
    create_multihot_matrix(
        source_df=df_proc_section0,
        value_col="pcs_body_system",
        list_col="pcs_body_system_list",
        feature_prefix="pcs_body_system"
    )
)

print(
    "PCS Body System (Section 0):",
    X_body.shape
)

# ------------------------------------------------------------
# 16.3) Root Operation
# Nur Section 0
# ------------------------------------------------------------

X_root, root_feature_names = (
    create_multihot_matrix(
        source_df=df_proc_section0,
        value_col="pcs_root_operation",
        list_col="pcs_root_operation_list",
        feature_prefix="pcs_root_operation"
    )
)

print(
    "PCS Root Operation (Section 0):",
    X_root.shape
)

# ------------------------------------------------------------
# 16.4) Zielvariable und Gruppen
# ------------------------------------------------------------

y = (
    mortality_stay[
        "hospital_expire_flag"
    ]
    .astype(int)
    .values
)

groups = (
    mortality_stay[
        "subject_id"
    ]
    .values
)


# ============================================================
# 17) Tatsächlich vorkommende Dimensionskombinationen erzeugen
# ============================================================

# ------------------------------------------------------------
# 17.1) Section × Body System
# Alle Sections
# ------------------------------------------------------------

df_proc["section_body_system"] = (
    df_proc["pcs_section"]
    + df_proc["pcs_body_system"]
)

X_section_body, section_body_feature_names = (
    create_multihot_matrix(
        source_df=df_proc,
        value_col="section_body_system",
        list_col="section_body_system_list",
        feature_prefix="pcs_section_body"
    )
)

print(
    "\nPCS Section × Body System:",
    X_section_body.shape
)

# ------------------------------------------------------------
# 17.2) Section × Root Operation
# Alle Sections
# ------------------------------------------------------------

df_proc["section_root_operation"] = (
    df_proc["pcs_section"]
    + df_proc["pcs_root_operation"]
)

X_section_root, section_root_feature_names = (
    create_multihot_matrix(
        source_df=df_proc,
        value_col="section_root_operation",
        list_col="section_root_operation_list",
        feature_prefix="pcs_section_root"
    )
)

print(
    "PCS Section × Root Operation:",
    X_section_root.shape
)

# ------------------------------------------------------------
# 17.3) Body System × Root Operation
# Nur Section 0
# ------------------------------------------------------------

df_proc_section0[
    "body_system_root_operation"
] = (
    df_proc_section0["pcs_body_system"]
    + df_proc_section0["pcs_root_operation"]
)

X_body_root, body_root_feature_names = (
    create_multihot_matrix(
        source_df=df_proc_section0,
        value_col="body_system_root_operation",
        list_col="body_system_root_operation_list",
        feature_prefix="pcs_body_root"
    )
)

print(
    "PCS Body System × Root Operation (Section 0):",
    X_body_root.shape
)

# ------------------------------------------------------------
# 17.4) Section × Body System × Root Operation
# Alle Sections
# ------------------------------------------------------------

df_proc[
    "section_body_root_operation"
] = (
    df_proc["pcs_section"]
    + df_proc["pcs_body_system"]
    + df_proc["pcs_root_operation"]
)

X_section_body_root, section_body_root_feature_names = (
    create_multihot_matrix(
        source_df=df_proc,
        value_col="section_body_root_operation",
        list_col="section_body_root_operation_list",
        feature_prefix="pcs_section_body_root"
    )
)

print(
    "PCS Section × Body System × Root Operation:",
    X_section_body_root.shape
)


# ============================================================
# 18) Prognostische Stärke mittels XGBoost vergleichen
# ============================================================

def evaluate_xgb_feature_set(
    X,
    y,
    groups,
    label
):

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision"
    }

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    return {
        "feature_set": label,
        "n_features": X.shape[1],
        "roc_auc_mean":
            scores["test_roc_auc"].mean(),
        "roc_auc_std":
            scores["test_roc_auc"].std(),
        "average_precision_mean":
            scores[
                "test_average_precision"
            ].mean(),
        "average_precision_std":
            scores[
                "test_average_precision"
            ].std()
    }


# ============================================================
# 19) Sieben strukturell sinnvolle Feature-Sets evaluieren
# ============================================================

# 1. Section – alle Sections
result_section = (
    evaluate_xgb_feature_set(
        X_section,
        y,
        groups,
        "PCS Section"
    )
)

# 2. Body System – nur Section 0
result_body = (
    evaluate_xgb_feature_set(
        X_body,
        y,
        groups,
        "PCS Body System (Section 0)"
    )
)

# 3. Root Operation – nur Section 0
result_root = (
    evaluate_xgb_feature_set(
        X_root,
        y,
        groups,
        "PCS Root Operation (Section 0)"
    )
)

# 4. Section × Body System – alle Sections
result_section_body = (
    evaluate_xgb_feature_set(
        X_section_body,
        y,
        groups,
        "PCS Section x Body System"
    )
)

# 5. Section × Root Operation – alle Sections
result_section_root = (
    evaluate_xgb_feature_set(
        X_section_root,
        y,
        groups,
        "PCS Section x Root Operation"
    )
)

# 6. Body System × Root Operation – nur Section 0
result_body_root = (
    evaluate_xgb_feature_set(
        X_body_root,
        y,
        groups,
        (
            "PCS Body System x Root Operation "
            "(Section 0)"
        )
    )
)

# 7. Section × Body System × Root Operation – alle Sections
result_section_body_root = (
    evaluate_xgb_feature_set(
        X_section_body_root,
        y,
        groups,
        (
            "PCS Section x Body System "
            "x Root Operation"
        )
    )
)


# ============================================================
# 20) Gesamter Performance-Vergleich
# ============================================================

performance_comparison = pd.DataFrame(
    [
        result_section,
        result_body,
        result_root,
        result_section_body,
        result_section_root,
        result_body_root,
        result_section_body_root
    ]
)

performance_comparison = (
    performance_comparison
    .sort_values(
        by="average_precision_mean",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    "\n=== Prognostische Stärke der PCS-Repräsentationen ==="
)

print(performance_comparison)

