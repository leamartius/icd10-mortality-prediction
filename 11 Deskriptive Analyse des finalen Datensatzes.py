#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALYSE DES FINALEN DATENSATZES

Das Skript beschreibt die finale Analysepopulation und untersucht ausgewählte
demographische, administrative und aufenthaltsbezogene Merkmale. Zusätzlich werden
die für die Modellierung verwendeten Diagnose- und Behandlungscodes analysiert
und die Auswirkungen des zuvor festgelegten Seltenheitsfilters ausgewertet.
Das Skript dient primär der nachträglichen Extraktion von Kennzahlen für die deskriptive 
Analyse in der Arbeit und ist nicht Bestandteil der eigentlichen Datenaufbereitungs- 
und Modellierungspipeline.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# PFAD ANPASSEN!
path = Path("Pfad_zu_zuvor_gespeicherten_Daten")
df_icd10_codelevel = pd.read_csv(path / "icd10_clean_codelevel.csv")

# Finale Analysepopulation aus Hauptanalyse
final_hadm_ids = np.load(path / "final_hadm_ids.npy")

# Tatsächlich für die Modellierung verwendete Codes
with open(
    path / "frequent_diag_codes.pkl",
    "rb"
) as f:
    frequent_diag_codes = pickle.load(f)

with open(
    path / "frequent_proc_codes.pkl",
    "rb"
) as f:
    frequent_proc_codes = pickle.load(f)


# ============================================================
# 1) Code-Level-Datensatz auf finale Analysepopulation begrenzen
# ============================================================

n_rows_before = len(df_icd10_codelevel)
n_stays_before = df_icd10_codelevel["hadm_id"].nunique()

df_icd10_codelevel = df_icd10_codelevel[
    df_icd10_codelevel["hadm_id"].isin(final_hadm_ids)
].copy()

n_rows_after = len(df_icd10_codelevel)
n_stays_after = df_icd10_codelevel["hadm_id"].nunique()

print("=" * 70)
print("FILTERUNG AUF FINALE ANALYSEPOPULATION")
print("=" * 70)

print("Codezeilen vorher:", n_rows_before)
print("Codezeilen nachher:", n_rows_after)

print("Aufenthalte vorher:", n_stays_before)
print("Aufenthalte nachher:", n_stays_after)

print(
    "Erwartete finale Aufenthalte:",
    len(final_hadm_ids)
)

missing_hadm_ids = (
    set(final_hadm_ids)
    - set(df_icd10_codelevel["hadm_id"].unique())
)

print(
    "Finale hadm_id, die im Code-Level-Datensatz fehlen:",
    len(missing_hadm_ids)
)

assert len(missing_hadm_ids) == 0, (
    "Nicht alle finalen Aufenthalte sind im Code-Level-Datensatz enthalten."
)

assert (
    df_icd10_codelevel["hadm_id"].nunique()
    == len(final_hadm_ids)
), (
    "Anzahl finaler Aufenthalte stimmt nicht mit final_hadm_ids überein."
)


# ============================================================
# 2) Zeitspalten in Datetime umwandeln
# ============================================================

time_cols = [
    "admittime",
    "dischtime",
    "deathtime",
    "chartdate"
]

for col in time_cols:
    if col in df_icd10_codelevel.columns:
        df_icd10_codelevel[col] = pd.to_datetime(
            df_icd10_codelevel[col],
            errors="coerce"
        )


# ============================================================
# 3) Aufenthaltsebene erzeugen
# ============================================================
# Der Datensatz liegt auf Codeebene vor.
# Jeder Aufenthalt kann daher mehrfach vorkommen.

df_icd10_stays = (
    df_icd10_codelevel
    .sort_values(["subject_id", "hadm_id"])
    .drop_duplicates(subset=["hadm_id"])
    .copy()
)

assert (
    df_icd10_stays["hadm_id"].nunique()
    == len(final_hadm_ids)
)

print("\n" + "=" * 70)
print("FINALE ICD-10-ANALYSEKOHORTE")
print("=" * 70)

print(
    "Eindeutige Patienten:",
    df_icd10_stays["subject_id"].nunique()
)

print(
    "Eindeutige Aufenthalte:",
    df_icd10_stays["hadm_id"].nunique()
)


# Anzahl Aufenthalte pro Patient
stays_per_patient = (
    df_icd10_stays
    .groupby("subject_id")["hadm_id"]
    .nunique()
)

print("\n" + "=" * 70)
print("ANZAHL AUFENTHALTE PRO PATIENT")
print("=" * 70)

print(
    stays_per_patient.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print("\nAnteil Patienten mit genau einem Aufenthalt:")
print(
    f"{(stays_per_patient == 1).mean() * 100:.2f} %"
)

print("\nAnteil Patienten mit mehr als einem Aufenthalt:")
print(
    f"{(stays_per_patient > 1).mean() * 100:.2f} %"
)


# ------------------------------------------------------------
# Verteilung als Säulendiagramm
# ------------------------------------------------------------

stays_per_patient_counts = (
    stays_per_patient
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 5))

plt.bar(
    stays_per_patient_counts.index,
    stays_per_patient_counts.values
)

plt.title("Anzahl Aufenthalte pro Patient")
plt.xlabel("Anzahl Aufenthalte")
plt.ylabel("Anzahl Patienten")

plt.tight_layout()
plt.show()


# ============================================================
# 4) Ungefähren realen Aufnahmezeitraum bestimmen
# ============================================================

if (
    "anchor_year_group" in df_icd10_stays.columns
    and "anchor_year" in df_icd10_stays.columns
):

    anchor_bounds = (
        df_icd10_stays["anchor_year_group"]
        .astype("string")
        .str.extract(
            r"(?P<anchor_group_start>\d{4})\s*-\s*"
            r"(?P<anchor_group_end>\d{4})"
        )
    )

    df_icd10_stays["anchor_group_start"] = pd.to_numeric(
        anchor_bounds["anchor_group_start"],
        errors="coerce"
    )

    df_icd10_stays["anchor_group_end"] = pd.to_numeric(
        anchor_bounds["anchor_group_end"],
        errors="coerce"
    )

    df_icd10_stays["year_offset"] = (
        df_icd10_stays["admittime"].dt.year
        - df_icd10_stays["anchor_year"]
    )

    df_icd10_stays["estimated_year_min"] = (
        df_icd10_stays["anchor_group_start"]
        + df_icd10_stays["year_offset"]
    )

    df_icd10_stays["estimated_year_max"] = (
        df_icd10_stays["anchor_group_end"]
        + df_icd10_stays["year_offset"]
    )

    print(
        "\nAufenthalte ohne schätzbaren realen Zeitraum:",
        df_icd10_stays[
            ["estimated_year_min", "estimated_year_max"]
        ].isna().any(axis=1).sum()
    )

    estimated_start = (
        df_icd10_stays["estimated_year_min"].min()
    )

    estimated_end = (
        df_icd10_stays["estimated_year_max"].max()
    )

    if (
        pd.notna(estimated_start)
        and pd.notna(estimated_end)
    ):
        print(
            "Ungefährer realer Aufnahmezeitraum: "
            f"{int(estimated_start)} bis {int(estimated_end)}"
        )


# ============================================================
# 5) Klassenverteilung der Zielvariable
# ============================================================

counts = (
    df_icd10_stays["hospital_expire_flag"]
    .value_counts(dropna=False)
    .sort_index()
)

props = (
    df_icd10_stays["hospital_expire_flag"]
    .value_counts(
        dropna=False,
        normalize=True
    )
    .sort_index()
    * 100
)

print("\n" + "=" * 70)
print("MORTALITÄT")
print("=" * 70)

print("Absolute Häufigkeiten:")
print(counts)

print("\nRelative Häufigkeiten (%):")
print(props.round(2))


# ============================================================
# 6) Geschlecht auf Patientenebene
# ============================================================

gender_data = (
    df_icd10_stays
    .drop_duplicates(subset=["subject_id"])
)

gender_counts = (
    gender_data["gender"]
    .value_counts(dropna=False)
)

gender_props = (
    gender_data["gender"]
    .value_counts(
        dropna=False,
        normalize=True
    )
    * 100
)

print("\n" + "=" * 70)
print("GESCHLECHT AUF PATIENTENEBENE")
print("=" * 70)

print("Absolute Häufigkeiten:")
print(gender_counts)

print("\nRelative Häufigkeiten (%):")
print(gender_props.round(2))


# ============================================================
# 7) Alter bei Aufnahme
# ============================================================

print("\n" + "=" * 70)
print("ALTER BEI AUFNAHME")
print("=" * 70)

print(
    df_icd10_stays["age_at_admission"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75
        ]
    )
)

plt.figure(figsize=(8, 5))

plt.hist(
    df_icd10_stays[
        "age_at_admission"
    ].dropna(),
    bins=30
)

plt.title("Alter bei Aufnahme")
plt.xlabel("Alter bei Aufnahme")
plt.ylabel("Anzahl Aufenthalte")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Altersgruppen
# ------------------------------------------------------------

age_bins = [
    0,
    18,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    np.inf
]

age_labels = [
    "0–17",
    "18–29",
    "30–39",
    "40–49",
    "50–59",
    "60–69",
    "70–79",
    "80–89",
    "90+"
]

df_icd10_stays["age_group"] = pd.cut(
    df_icd10_stays["age_at_admission"],
    bins=age_bins,
    labels=age_labels,
    right=False
)

print("\nAltersgruppen:")
print(
    df_icd10_stays["age_group"]
    .value_counts(dropna=False)
    .sort_index()
)

age_group_plot = (
    df_icd10_stays["age_group"]
    .value_counts()
    .sort_index()
)

ax = age_group_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title(
    "Altersverteilung der finalen Analysepopulation"
)

ax.set_xlabel(
    "Alter bei Aufnahme"
)

ax.set_ylabel(
    "Anzahl Aufenthalte"
)

ax.tick_params(
    axis="x",
    rotation=45
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.0f",
        padding=3
    )

plt.tight_layout()
plt.show()


# ============================================================
# 8) Art und Ort der Aufnahme
# ============================================================

def analyze_categorical(
    df,
    col,
    top_n=20
):
    counts = (
        df[col]
        .value_counts(dropna=False)
    )

    props = (
        df[col]
        .value_counts(
            dropna=False,
            normalize=True
        )
        * 100
    )

    print("\n" + "=" * 70)
    print(col)
    print("=" * 70)

    print("Absolute Häufigkeiten:")
    print(counts.head(top_n))

    print("\nRelative Häufigkeiten (%):")
    print(
        props.head(top_n).round(2)
    )

    print(
        "\nAnzahl eindeutiger Kategorien:",
        df[col].nunique(dropna=True)
    )

    print(
        "Missing:",
        df[col].isna().sum()
    )


for col in [
    "admission_type",
    "admission_location"
]:

    if col in df_icd10_stays.columns:

        analyze_categorical(
            df_icd10_stays,
            col
        )

        plt.figure(
            figsize=(10, 5)
        )

        (
            df_icd10_stays[col]
            .value_counts(dropna=False)
            .plot(kind="bar")
        )

        plt.title(
            f"Ausprägungen von {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Anzahl Aufenthalte")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# 9) Length of Stay
# ============================================================

adm = df_icd10_stays.copy()

adm["los_days"] = (
    adm["dischtime"]
    - adm["admittime"]
).dt.total_seconds() / (24 * 3600)

adm["valid_time_order"] = (
    adm["admittime"]
    < adm["dischtime"]
)

print("\n" + "=" * 70)
print("LENGTH OF STAY")
print("=" * 70)

print(
    "Aufenthalte mit admittime >= dischtime:",
    (~adm["valid_time_order"]).sum()
)

print("\nLOS deskriptiv:")

print(
    adm["los_days"].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.995,
            0.999
        ]
    )
)

threshold = (
    adm["los_days"]
    .quantile(0.995)
)

extreme_long_los = (
    adm[
        adm["los_days"]
        > threshold
    ]
)

print(
    "\n99,5%-Quantil:",
    round(threshold, 2)
)

print(
    "Aufenthalte oberhalb:",
    len(extreme_long_los)
)

very_long_los = (
    adm[
        adm["los_days"] > 180
    ]
)

print(
    "Aufenthalte mit LOS > 180 Tage:",
    len(very_long_los)
)

problem_cases = adm[
    (~adm["valid_time_order"])
    | (
        adm["los_days"]
        > threshold
    )
]

print("\nBeispiele auffälliger Aufenthalte:")

print(
    problem_cases[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "los_days"
        ]
    ].head(20)
)


# ============================================================
# 10) hospital_expire_flag und deathtime prüfen
# ============================================================

adm = df_icd10_stays.copy()

case1 = adm[
    (adm["hospital_expire_flag"] == 1)
    & adm["deathtime"].isna()
]

case2 = adm[
    (adm["hospital_expire_flag"] == 0)
    & adm["deathtime"].notna()
]

case3 = adm[
    adm["deathtime"].notna()
    & (
        (
            adm["deathtime"]
            < adm["admittime"]
        )
        |
        (
            adm["deathtime"]
            > adm["dischtime"]
        )
    )
].copy()

print("\n" + "=" * 70)
print("INKONSISTENZEN TODESZEITPUNKT")
print("=" * 70)

print(
    "hospital_expire_flag = 1, aber deathtime fehlt:",
    len(case1)
)

print(
    "hospital_expire_flag = 0, aber deathtime vorhanden:",
    len(case2)
)

print(
    "deathtime außerhalb des Aufenthalts:",
    len(case3)
)


if len(case1) > 0:
    print("\nBeispiele Fall 1:")
    print(
        case1[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "hospital_expire_flag"
            ]
        ].head(10)
    )


if len(case2) > 0:
    print("\nBeispiele Fall 2:")
    print(
        case2[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "hospital_expire_flag"
            ]
        ].head(10)
    )


if len(case3) > 0:

    case3[
        "death_before_admit"
    ] = (
        case3["deathtime"]
        < case3["admittime"]
    ).astype(int)

    case3[
        "death_after_discharge"
    ] = (
        case3["deathtime"]
        > case3["dischtime"]
    ).astype(int)

    case3[
        "delta_from_discharge_hours"
    ] = (
        (
            case3["deathtime"]
            - case3["dischtime"]
        )
        .dt.total_seconds()
        / 3600
    )

    print("\nFall 3 nach hospital_expire_flag:")
    print(
        case3[
            "hospital_expire_flag"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nAnzahl vor Aufnahme / nach Entlassung:"
    )

    print(
        case3[
            [
                "death_before_admit",
                "death_after_discharge"
            ]
        ].sum()
    )

    print(
        "\nAbweichung zur Entlassung in Stunden:"
    )

    print(
        case3[
            "delta_from_discharge_hours"
        ].describe()
    )


# ============================================================
# 11) Todesfälle innerhalb 24 und 48 Stunden
# ============================================================

adm = df_icd10_stays.copy()

adm[
    "time_to_death_hours"
] = (
    (
        adm["deathtime"]
        - adm["admittime"]
    )
    .dt.total_seconds()
    / 3600
)

adm[
    "death_within_48h"
] = (
    (adm["hospital_expire_flag"] == 1)
    & adm[
        "time_to_death_hours"
    ].between(
        0,
        48,
        inclusive="both"
    )
)

adm[
    "death_within_24h"
] = (
    (adm["hospital_expire_flag"] == 1)
    & adm[
        "time_to_death_hours"
    ].between(
        0,
        24,
        inclusive="both"
    )
)

deaths = adm[
    adm["hospital_expire_flag"] == 1
]

n_deaths_total = len(deaths)

n_deaths_48h = int(
    deaths["death_within_48h"].sum()
)

n_deaths_24h = int(
    deaths["death_within_24h"].sum()
)

share_48h_all = (
    n_deaths_48h
    / len(adm)
    * 100
)

share_48h_deaths = (
    n_deaths_48h
    / n_deaths_total
    * 100
    if n_deaths_total > 0
    else np.nan
)

share_24h_all = (
    n_deaths_24h
    / len(adm)
    * 100
)

share_24h_deaths = (
    n_deaths_24h
    / n_deaths_total
    * 100
    if n_deaths_total > 0
    else np.nan
)

print("\n" + "=" * 70)
print("FRÜHE TODESFÄLLE")
print("=" * 70)

print(
    "Gesamte Todesfälle:",
    n_deaths_total
)

print(
    "Davon innerhalb 48h:",
    n_deaths_48h
)

print(
    f"Anteil 48h an allen Aufenthalten: "
    f"{share_48h_all:.2f} %"
)

print(
    f"Anteil 48h an allen Todesfällen: "
    f"{share_48h_deaths:.2f} %"
)

print(
    "\nDavon innerhalb 24h:",
    n_deaths_24h
)

print(
    f"Anteil 24h an allen Aufenthalten: "
    f"{share_24h_all:.2f} %"
)

print(
    f"Anteil 24h an allen Todesfällen: "
    f"{share_24h_deaths:.2f} %"
)


# ============================================================
# 12) Diagnose- und Behandlungscodes trennen
# ============================================================

diagnoses_icd10 = (
    df_icd10_codelevel[
        df_icd10_codelevel[
            "code_type"
        ]
        .astype("string")
        .str.lower()
        .eq("diagnosis")
    ]
    .copy()
)

procedures_icd10 = (
    df_icd10_codelevel[
        df_icd10_codelevel[
            "code_type"
        ]
        .astype("string")
        .str.lower()
        .eq("procedure")
    ]
    .copy()
)


# ============================================================
# 13) Auf tatsächlich verwendete Codes begrenzen
# ============================================================

# Codes für sicheren Vergleich normalisieren
diagnoses_icd10[
    "icd_code_clean"
] = (
    diagnoses_icd10[
        "icd_code"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)

procedures_icd10[
    "icd_code_clean"
] = (
    procedures_icd10[
        "icd_code"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)

frequent_diag_codes_clean = {
    str(code).strip().upper()
    for code in frequent_diag_codes
}

frequent_proc_codes_clean = {
    str(code).strip().upper()
    for code in frequent_proc_codes
}

diagnoses_icd10_model = (
    diagnoses_icd10[
        diagnoses_icd10[
            "icd_code_clean"
        ].isin(
            frequent_diag_codes_clean
        )
    ]
    .copy()
)

procedures_icd10_model = (
    procedures_icd10[
        procedures_icd10[
            "icd_code_clean"
        ].isin(
            frequent_proc_codes_clean
        )
    ]
    .copy()
)

print("\n" + "=" * 70)
print("CODEFILTERUNG")
print("=" * 70)

print("\nDiagnosecodes:")

print(
    "Eindeutige Codes vor Seltenheitsfilter:",
    diagnoses_icd10[
        "icd_code_clean"
    ].nunique()
)

print(
    "Eindeutige Codes nach Seltenheitsfilter:",
    diagnoses_icd10_model[
        "icd_code_clean"
    ].nunique()
)

print("\nBehandlungscodes:")

print(
    "Eindeutige Codes vor Seltenheitsfilter:",
    procedures_icd10[
        "icd_code_clean"
    ].nunique()
)

print(
    "Eindeutige Codes nach Seltenheitsfilter:",
    procedures_icd10_model[
        "icd_code_clean"
    ].nunique()
)



# ============================================================
# Vergleich Diagnosecodes pro Aufenthalt:
# vor vs. nach Seltenheitsfilter
# ============================================================

diag_per_hadm_before = (
    diagnoses_icd10
    .groupby("hadm_id")["icd_code_clean"]
    .nunique()
    .reindex(final_hadm_ids, fill_value=0)
)

diag_per_hadm_after = (
    diagnoses_icd10_model
    .groupby("hadm_id")["icd_code_clean"]
    .nunique()
    .reindex(final_hadm_ids, fill_value=0)
)

comparison_diag = pd.DataFrame({
    "vor_filter": diag_per_hadm_before,
    "nach_filter": diag_per_hadm_after
})

comparison_diag["entfernte_codes"] = (
    comparison_diag["vor_filter"]
    - comparison_diag["nach_filter"]
)

print("\n" + "=" * 70)
print("AUSWIRKUNG DES SELTENHEITSFILTERS AUF DIAGNOSECODES")
print("=" * 70)

print("\nVor Seltenheitsfilter:")
print(diag_per_hadm_before.describe())

print("\nNach Seltenheitsfilter:")
print(diag_per_hadm_after.describe())

print("\nEntfernte Diagnosecodes pro Aufenthalt:")
print(comparison_diag["entfernte_codes"].describe())

print(
    "\nAufenthalte, bei denen mindestens ein Diagnosecode entfernt wurde:",
    (comparison_diag["entfernte_codes"] > 0).sum()
)

print(
    "Anteil:",
    f"{(comparison_diag['entfernte_codes'] > 0).mean() * 100:.2f} %"
)

print(
    "\nInsgesamt entfernte Code-Aufenthalt-Kombinationen:",
    comparison_diag["entfernte_codes"].sum()
)



# ============================================================
# 14) Allgemeine Codeanalyse
# ============================================================

def analyze_icd_table(
    df,
    name="ICD-10"
):
    d = df.copy()

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Codezeilen:",
        len(d)
    )

    print(
        "Missing icd_code:",
        d["icd_code"].isna().sum()
    )

    if "icd_code_clean" not in d.columns:
        d[
            "icd_code_clean"
        ] = (
            d["icd_code"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    code_freq = (
        d.groupby(
            "icd_code_clean"
        )["hadm_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        "Eindeutige Codes:",
        d[
            "icd_code_clean"
        ].nunique()
    )

    rare_1 = (
        code_freq == 1
    ).sum()

    rare_lt_5 = (
        code_freq < 5
    ).sum()

    rare_lt_10 = (
        code_freq < 10
    ).sum()

    print("\nLong-Tail-Metriken:")

    print(
        "Codes in genau einem Aufenthalt:",
        rare_1
    )

    print(
        "Codes in weniger als 5 Aufenthalten:",
        rare_lt_5
    )

    print(
        "Codes in weniger als 10 Aufenthalten:",
        rare_lt_10
    )

    total_code_stays = (
        code_freq.sum()
    )

    if total_code_stays > 0:

        for n in [
            10,
            50,
            100,
            500
        ]:

            coverage = (
                code_freq
                .head(n)
                .sum()
                / total_code_stays
                * 100
            )

            print(
                f"Top {n} Codes decken "
                f"{coverage:.2f} % "
                "aller Codevorkommen auf Aufenthaltsebene ab."
            )

    return (
        d,
        code_freq
    )


diag_checked, diag_freq = (
    analyze_icd_table(
        diagnoses_icd10_model,
        "ICD-10-Diagnosecodes – finales Feature-Set"
    )
)

proc_checked, proc_freq = (
    analyze_icd_table(
        procedures_icd10_model,
        "ICD-10-Behandlungscodes – finales Feature-Set"
    )
)


# ============================================================
# 15) Anzahl eindeutiger Diagnosecodes pro Aufenthalt
# ============================================================

diag_per_hadm = (
    diagnoses_icd10_model
    .groupby(
        "hadm_id"
    )[
        "icd_code_clean"
    ]
    .nunique()
)

# Aufenthalte ohne verbleibende Diagnosecodes als 0 ergänzen
diag_per_hadm = (
    diag_per_hadm
    .reindex(
        final_hadm_ids,
        fill_value=0
    )
)

print("\n" + "=" * 70)
print("DIAGNOSECODES PRO AUFENTHALT")
print("=" * 70)

print(
    diag_per_hadm.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)

diag_counts = (
    diag_per_hadm
    .value_counts()
    .sort_index()
)

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    diag_counts.index,
    diag_counts.values
)

plt.title(
    "Anzahl eindeutiger Diagnosecodes pro Aufenthalt"
)

plt.xlabel(
    "Anzahl eindeutiger Diagnosecodes"
)

plt.ylabel(
    "Anzahl Aufenthalte"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16) Anzahl eindeutiger Behandlungscodes pro Aufenthalt
# ============================================================

proc_per_hadm = (
    procedures_icd10_model
    .groupby(
        "hadm_id"
    )[
        "icd_code_clean"
    ]
    .nunique()
)

# Aufenthalte ohne Behandlungscodes als 0 ergänzen
proc_per_hadm = (
    proc_per_hadm
    .reindex(
        final_hadm_ids,
        fill_value=0
    )
)

print("\n" + "=" * 70)
print("BEHANDLUNGSCODES PRO AUFENTHALT")
print("=" * 70)

print(
    proc_per_hadm.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)

proc_counts = (
    proc_per_hadm
    .value_counts()
    .sort_index()
)

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    proc_counts.index,
    proc_counts.values
)

plt.title(
    "Anzahl eindeutiger Behandlungscodes pro Aufenthalt"
)

plt.xlabel(
    "Anzahl eindeutiger Behandlungscodes"
)

plt.ylabel(
    "Anzahl Aufenthalte"
)

plt.tight_layout()
plt.show()


# ============================================================
# 17) Aufenthalte mit mindestens einem verbleibenden Code
# ============================================================

n_stays_with_diag = (diag_per_hadm > 0).sum()
n_stays_with_proc = (proc_per_hadm > 0).sum()

share_stays_with_diag = (
    n_stays_with_diag / len(final_hadm_ids) * 100
)

share_stays_with_proc = (
    n_stays_with_proc / len(final_hadm_ids) * 100
)

print("\n" + "=" * 70)
print("AUFENTHALTE MIT DIAGNOSE- UND BEHANDLUNGSCODES")
print("=" * 70)

print(
    "Aufenthalte mit mindestens einem Diagnosecode:",
    n_stays_with_diag,
    f"({share_stays_with_diag:.2f} %)"
)

print(
    "Aufenthalte mit mindestens einem Behandlungscode:",
    n_stays_with_proc,
    f"({share_stays_with_proc:.2f} %)"
)

print(
    "Aufenthalte ohne verbleibenden Diagnosecode:",
    (diag_per_hadm == 0).sum()
)

print(
    "Aufenthalte ohne verbleibenden Behandlungscode:",
    (proc_per_hadm == 0).sum()
)

# ============================================================
# 18) Längenverteilung der ICD-10-Diagnosecodes
# ============================================================

diag_codes_length = (
    diagnoses_icd10_model[
        diagnoses_icd10_model[
            "icd_code_clean"
        ].notna()
    ]
    .copy()
)

# Punkte entfernen, falls vorhanden
diag_codes_length[
    "icd_code_clean_no_dot"
] = (
    diag_codes_length[
        "icd_code_clean"
    ]
    .astype("string")
    .str.replace(
        ".",
        "",
        regex=False
    )
    .str.strip()
    .str.upper()
)

diag_codes_length[
    "code_len"
] = (
    diag_codes_length[
        "icd_code_clean_no_dot"
    ]
    .str.len()
)

code_length_counts = (
    diag_codes_length[
        "code_len"
    ]
    .value_counts()
    .sort_index()
)

code_length_share = (
    diag_codes_length[
        "code_len"
    ]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print("\n" + "=" * 70)
print("LÄNGENVERTEILUNG DER DIAGNOSECODES")
print("=" * 70)

print(
    "Absolute Häufigkeiten aller Codevorkommen:"
)

print(
    code_length_counts
)

print(
    "\nRelative Häufigkeiten aller Codevorkommen (%):"
)

print(
    code_length_share.round(2)
)


# ------------------------------------------------------------
# Längenverteilung eindeutiger Diagnosecodes
# ------------------------------------------------------------

unique_diag_codes = (
    diag_codes_length[
        [
            "icd_code_clean",
            "code_len"
        ]
    ]
    .drop_duplicates(
        subset=[
            "icd_code_clean"
        ]
    )
    .copy()
)

unique_code_length_counts = (
    unique_diag_codes[
        "code_len"
    ]
    .value_counts()
    .sort_index()
)

unique_code_length_share = (
    unique_diag_codes[
        "code_len"
    ]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print(
    "\nAbsolute Häufigkeiten – eindeutige Diagnosecodes:"
)

print(
    unique_code_length_counts
)

print(
    "\nRelative Häufigkeiten – eindeutige Diagnosecodes (%):"
)

print(
    unique_code_length_share.round(2)
)



