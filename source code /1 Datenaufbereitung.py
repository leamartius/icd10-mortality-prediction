#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATENAUFBEREITUNG

Das Skript bereitet die MIMIC-IV-Daten für die weitere Analyse auf. Dazu werden zuerst
Patienten-, Aufenthalts-, Diagnose- und Behandlungsdaten zusammengeführt. 
Zuerst wird der vollständige und danach der ICD-10-Datensatz deskriptiv analysiert 
und auf Dateninkonsistenzen geprüft.
Ausgewählte Variablen werden bereinigt und transformiert. 
Der bereinigte Datensatz wird anschließend exportiert.
Am Ende folgt eine Untersuchung zur Festlegung der Mindesthäufigkeit von Diagnose- und 
Behandlungscodes: Es werden verschiedene Häufigkeitsschwellen untersucht und deren 
Auswirkungen auf die Anzahl verbleibender Codes, betroffener Aufenthalte und Todesfälle 
analysiert. Auf dieser Grundlage werden in einem anderen Skript Diagnosecodes mit 
weniger als zehn und Behandlungscodes mit weniger als drei Aufenthalten ausgeschlossen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PFADE ANPASSEN!
path = Path("Pfad_zu_MIMIC-IV_hosp-Tabellen")
output_path = Path("Pfad_zum_Output")
output_path.mkdir(parents=True, exist_ok=True)

patients = pd.read_csv(path / "patients.csv")
admissions = pd.read_csv(path / "admissions.csv")
diagnoses_icd = pd.read_csv(path / "diagnoses_icd.csv")
procedures_icd = pd.read_csv(path / "procedures_icd.csv")
d_icd_diagnoses = pd.read_csv(path / "d_icd_diagnoses.csv")
d_icd_procedures = pd.read_csv(path / "d_icd_procedures.csv")


# ===========================================================
# BESCHREIBUNG DES VOLLSTÄNDIGEN DATENSATZES
# EINSCHLIESSLICH ICD-9 UND ICD-10
# ===========================================================

# -----------------------------------------------------------
# 1. Allgemeiner Check: Ist jede subject_id und hadm_id eindeutig?
# -----------------------------------------------------------

# subject_id in patients (hosp + ICU)
n_all_patients = len(patients)
n_unique_subjects = patients["subject_id"].nunique()
duplicate_subjects = patients[patients["subject_id"].duplicated(keep=False)].sort_values("subject_id")

print("PATIENTS")
print(f"Zeilen insgesamt: {n_all_patients}")
print(f"Eindeutige subject_id: {n_unique_subjects}")
print(f"Doppelte subject_id: {n_all_patients - n_unique_subjects}")

if not duplicate_subjects.empty:
    print("\nBeispiele für doppelte subject_id:")
    print(duplicate_subjects.head(20))
else:
    print("Jede subject_id kommt genau einmal vor.")

# hadm_id in admissions
n_adm = len(admissions)
n_unique_hadm = admissions["hadm_id"].nunique()
duplicate_hadm = admissions[admissions["hadm_id"].duplicated(keep=False)].sort_values("hadm_id")

print("\nADMISSIONS")
print(f"Zeilen insgesamt: {n_adm}")
print(f"Eindeutige hadm_id: {n_unique_hadm}")
print(f"Doppelte hadm_id: {n_adm - n_unique_hadm}")

if not duplicate_hadm.empty:
    print("\nBeispiele für doppelte hadm_id:")
    print(duplicate_hadm.head(20))
else:
    print("Jede hadm_id kommt genau einmal vor.")
    

# -----------------------------------------------------------
# 2. Anzahl eindeutiger Patienten und Aufenthalte
# -----------------------------------------------------------

# subject_id in admissions (nur stationäre Patienten)
n_inpatients = admissions["subject_id"].nunique()
print(f"Anzahl eindeutiger stationärer Patienten: {n_inpatients}")
print(f"Anzahl eindeutiger Aufenthalte: {n_unique_hadm}")


# -----------------------------------------------------------
# 3. Anzahl der ICD-9- und ICD-10-Codes
# -----------------------------------------------------------

# Diagnose- und Behandlungscodes gemeinsam
# Für die gemeinsame Auswertung nur relevante Variablen übernehmen
diag_versions = diagnoses_icd[
    ["hadm_id", "icd_code", "icd_version"]
].copy()
diag_versions["code_type"] = "diagnosis"

proc_versions = procedures_icd[
    ["hadm_id", "icd_code", "icd_version"]
].copy()
proc_versions["code_type"] = "procedure"

all_codes_raw = pd.concat(
    [diag_versions, proc_versions],
    ignore_index=True
)

# Nur gültige ICD-9- und ICD-10-Codezeilen berücksichtigen
all_codes_raw = all_codes_raw[
    all_codes_raw["icd_version"].isin([9, 10])
    & all_codes_raw["icd_code"].notna()
].copy()

# Codes für konsistente Vergleiche bereinigen
all_codes_raw["icd_code"] = (
    all_codes_raw["icd_code"]
    .astype("string")
    .str.strip()
    .str.upper()
)

# Gesamtzahl der Codezeilen pro ICD-Version
code_rows_by_version = (
    all_codes_raw["icd_version"]
    .value_counts()
    .reindex([9, 10], fill_value=0)
)

# Anzahl eindeutiger Codes pro ICD-Version
#
# code_type wird berücksichtigt, damit ein gleich geschriebener Diagnose-
# und Behandlungscode nicht fälschlich als derselbe Code behandelt wird.
unique_codes_by_version = (
    all_codes_raw
    .drop_duplicates(
        subset=["icd_version", "code_type", "icd_code"]
    )
    ["icd_version"]
    .value_counts()
    .reindex([9, 10], fill_value=0)
)

print("\n=== ICD-Codeverteilung ===")
print(
    f"ICD-9-Codezeilen insgesamt: "
    f"{code_rows_by_version.loc[9]:,}"
)
print(
    f"ICD-10-Codezeilen insgesamt: "
    f"{code_rows_by_version.loc[10]:,}"
)
print(
    f"Eindeutige ICD-9-Codes: "
    f"{unique_codes_by_version.loc[9]:,}"
)
print(
    f"Eindeutige ICD-10-Codes: "
    f"{unique_codes_by_version.loc[10]:,}"
)


# Getrennt nach Diagnose- und Behandlungscode
code_summary_by_type = (
    all_codes_raw
    .groupby(["code_type", "icd_version"])
    .agg(
        n_code_rows=("icd_code", "size"),
        n_unique_codes=("icd_code", "nunique")
    )
    .reindex(
        pd.MultiIndex.from_product(
            [["diagnosis", "procedure"], [9, 10]],
            names=["code_type", "icd_version"]
        ),
        fill_value=0
    )
)

print("\n=== ICD-Codes nach Codetyp ===")
print(code_summary_by_type)


# Vorhandene Häufigkeitstabelle für das Diagramm umformen
unique_codes_plot = (
    code_summary_by_type["n_unique_codes"]
    .unstack("code_type")
    .reindex(index=[9, 10], fill_value=0)
    .reindex(columns=["diagnosis", "procedure"], fill_value=0)
)

unique_codes_plot = unique_codes_plot.rename(
    index={
        9: "ICD-9",
        10: "ICD-10"
    },
    columns={
        "diagnosis": "Diagnosecodes",
        "procedure": "Behandlungscodes"
    }
)

# Gestapeltes Säulendiagramm
ax = unique_codes_plot.plot(
    kind="bar",
    stacked=True,
    figsize=(7, 5)
)

ax.set_title("Anzahl eindeutiger ICD-Codes nach Version und Code-Typ")
ax.set_xlabel("ICD-Version")
ax.set_ylabel("Anzahl eindeutiger Codes")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Code-Typ")

# Werte in den Säulensegmenten anzeigen
for container in ax.containers:
    ax.bar_label(
        container,
        label_type="center",
        fmt="%.0f"
    )

# Gesamtsumme oberhalb der Säulen
totals = unique_codes_plot.sum(axis=1)

for i, total in enumerate(totals):
    ax.text(
        i,
        total,
        f"{total:,.0f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 4. Aufenthalte nach vorhandener ICD-Version klassifizieren
# -----------------------------------------------------------

# Pro Aufenthalt prüfen, ob mindestens ein ICD-9- bzw. ICD-10-Code vorliegt
stay_versions = (
    all_codes_raw
    .assign(
        has_icd9=all_codes_raw["icd_version"].eq(9),
        has_icd10=all_codes_raw["icd_version"].eq(10)
    )
    .groupby("hadm_id", as_index=False)
    .agg(
        has_icd9=("has_icd9", "any"),
        has_icd10=("has_icd10", "any")
    )
)

# Alle Aufenthalte aus admissions als Ausgangspunkt verwenden,
# damit auch Aufenthalte ohne jeglichen ICD-Code sichtbar bleiben
stay_versions = (
    admissions[["hadm_id"]]
    .drop_duplicates()
    .merge(
        stay_versions,
        on="hadm_id",
        how="left"
    )
)

stay_versions[["has_icd9", "has_icd10"]] = (
    stay_versions[["has_icd9", "has_icd10"]]
    .fillna(False)
    .astype(bool)
)

conditions = [
    stay_versions["has_icd9"] & ~stay_versions["has_icd10"],
    ~stay_versions["has_icd9"] & stay_versions["has_icd10"],
    stay_versions["has_icd9"] & stay_versions["has_icd10"]
]

categories = [
    "nur ICD-9",
    "nur ICD-10",
    "ICD-9 und ICD-10"
]

stay_versions["icd_version_group"] = np.select(
    conditions,
    categories,
    default="keine ICD-Codes"
)

stay_group_counts = (
    stay_versions["icd_version_group"]
    .value_counts()
    .reindex(
        [
            "nur ICD-9",
            "nur ICD-10",
            "ICD-9 und ICD-10",
            "keine ICD-Codes"
        ],
        fill_value=0
    )
)

stay_group_shares = (
    stay_group_counts / len(stay_versions) * 100
)

print("\n=== Aufenthalte nach vorhandener ICD-Version ===")

for group in stay_group_counts.index:
    print(
        f"{group}: {stay_group_counts.loc[group]:,} "
        f"({stay_group_shares.loc[group]:.2f} %)"
    )

# Konsistenzprüfung
assert stay_group_counts.sum() == n_unique_hadm




### =============================
### DATAFRAMES AUF CODEEBENE
### =============================

# -------------------------------
# 1. Basis-Tabellen zusammenführen
# -------------------------------

# Merge admissions + patients
base_df = admissions.merge(
    patients,
    on="subject_id",
    how="left"
)

# Relevante Spalten auswählen 
base_df = base_df[[
    "subject_id", "hadm_id",
    "admittime", "dischtime", "deathtime",
    "admission_type", "admission_location",
    "gender", "anchor_age", "anchor_year", "hospital_expire_flag"
]]

# -------------------------------
# 2. Diagnosen vorbereiten
# -------------------------------

diagnoses = diagnoses_icd.merge(
    d_icd_diagnoses,
    on=["icd_code", "icd_version"],
    how="left"
)

diagnoses["code_type"] = "diagnosis"

# -------------------------------
# 3. Behandlungen vorbereiten
# -------------------------------

procedures = procedures_icd.merge(
    d_icd_procedures,
    on=["icd_code", "icd_version"],
    how="left"
)

procedures["code_type"] = "procedure"

# -------------------------------
# 4. Diagnosen + Behandlungen kombinieren
# -------------------------------

codes_df = pd.concat([diagnoses, procedures], ignore_index=True)

# Einheitliche Spaltennamen
codes_df = codes_df.rename(columns={
    "long_title": "code_description"
})

# -------------------------------
# 5. Codes mit Basisdaten mergen
# -------------------------------

full_df = base_df.merge(
    codes_df,
    on=["subject_id", "hadm_id"],
    how="left"
)

# Datensatz 1: Alle ICD-Codes
df_all_icd = full_df.copy()

# Datensatz 2: Nur ICD-10
df_icd10 = full_df[full_df["icd_version"] == 10].copy()

# -------------------------------
# 6. Zeitspalten in Datetime-Objekte umwandeln
# -------------------------------

dfs = [base_df, diagnoses, procedures, codes_df, full_df, df_all_icd, df_icd10]

time_cols = ["admittime", "dischtime", "deathtime", "chartdate"]

for df in dfs:
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

# -------------------------------
# 7. Merge prüfen
# -------------------------------

print("admissions:", admissions.shape)
print("patients:", patients.shape)
print("base_df:", base_df.shape)

print("diagnoses_icd:", diagnoses_icd.shape)
print("d_icd_diagnoses:", d_icd_diagnoses.shape)
print("diagnoses:", diagnoses.shape)

print("procedures_icd:", procedures_icd.shape)
print("d_icd_procedures:", d_icd_procedures.shape)
print("procedures:", procedures.shape)

print("codes_df:", codes_df.shape)
print("full_df:", full_df.shape)
print("df_icd10:", df_icd10.shape)

base_dup = base_df.duplicated(subset=["subject_id", "hadm_id"]).sum()
print("Doppelte subject_id-hadm_id in base_df:", base_dup)

missing_diag_titles = diagnoses["long_title"].isna().sum()
print("Diagnosezeilen ohne long_title:", missing_diag_titles)

missing_proc_titles = procedures["long_title"].isna().sum()
print("Behandlungszeilen ohne long_title:", missing_proc_titles)

print("diagnoses_icd rows:", len(diagnoses_icd))
print("diagnoses rows:", len(diagnoses))

print("procedures_icd rows:", len(procedures_icd))
print("procedures rows:", len(procedures))

print(
    d_icd_diagnoses.duplicated(subset=["icd_code", "icd_version"]).sum(),
    "Duplikate in d_icd_diagnoses"
)

print(
    d_icd_procedures.duplicated(subset=["icd_code", "icd_version"]).sum(),
    "Duplikate in d_icd_procedures"
)

print("Anzahl eindeutiger Aufenthalte in base_df:", base_df["hadm_id"].nunique())
print("Anzahl eindeutiger Aufenthalte in full_df:", full_df["hadm_id"].nunique())

print("Verteilung der ICD-Versionen:")
print(full_df["icd_version"].value_counts(dropna=False))

print("Check: Nur ICD-10 im gefilterten Dataframe?")
print(df_icd10["icd_version"].value_counts(dropna=False))

# Check anhand eines Beispiels
example_hadm = full_df["hadm_id"].dropna().iloc[0]

with pd.option_context('display.max_columns', None, 'display.width', None):
    print(full_df[full_df["hadm_id"] == example_hadm].head(10))


# -------------------------------
# 8. Aufenthalte ohne Codes prüfen
# -------------------------------

no_code_hadm = (
    full_df.groupby("hadm_id")["icd_code"]
    .apply(lambda x: x.isna().all())
)
no_code_hadm = no_code_hadm[no_code_hadm].index

print("Anzahl Aufenthalte ohne Code:", len(no_code_hadm))
# Anzahl Aufenthalte ohne Code: 452

no_code_cases = base_df[base_df["hadm_id"].isin(no_code_hadm)].copy()

with pd.option_context('display.max_columns', None, 'display.width', None):
    print(no_code_cases.head(10))

# Anzahl Todesfälle ohne Codes -> 4
cases_no_code_and_death = (
    base_df[base_df["hadm_id"].isin(no_code_hadm)]["hospital_expire_flag"].sum()
)

print("Stationäre Todesfälle ohne ICD-Codes:", cases_no_code_and_death)

# Durchschnittliche LOS dieser Fälle -> 1.86
no_code_cases["los_days"] = (
    no_code_cases["dischtime"] - no_code_cases["admittime"]
).dt.total_seconds() / (24 * 60 * 60)

avg_los = no_code_cases["los_days"].mean()

print("Durchschnittliche Length of Stay (in Tagen):", avg_los)


"""
# -------------------------------
# 9. Exportieren (als Zwischenergebnisse)
# -------------------------------

df_all_icd.to_csv(output_path / "mimic_all_icd.csv", index=False)
df_icd10.to_csv(output_path / "mimic_icd10.csv", index=False)

print("Export abgeschlossen.")
"""




### =============================
### DESKRIPTIVE ANALYSE VOM ICD-10-DATAFRAME 
### =============================

# -----------------------------------------------------------
# 1. ICD-10-Datensatz auf Aufenthaltsebene
# -----------------------------------------------------------

# df_icd10 liegt auf Codeebene vor:
# Ein Aufenthalt kommt aufgrund mehrerer Codes mehrfach vor.
df_icd10_stays = (
    df_icd10
    .sort_values(["subject_id", "hadm_id"])
    .drop_duplicates(subset=["hadm_id"])
    .copy()
)

print("=== ICD-10-Analysekohorte ===")
print(
    "Eindeutige Patienten:",
    df_icd10_stays["subject_id"].nunique()
)
print(
    "Eindeutige Aufenthalte:",
    df_icd10_stays["hadm_id"].nunique()
)

# -----------------------------------------------------------
# Verteilung der Aufenthalte pro Patient
# -----------------------------------------------------------

stays_per_patient = (
    df_icd10_stays
    .groupby("subject_id")["hadm_id"]
    .nunique()
)

print("\n=== Anzahl Aufenthalte pro Patient ===")

print(
    stays_per_patient.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.99
        ]
    )
)

# -----------------------------------------------------------
# 2. Ungefähren realen Aufnahmezeitraum bestimmen
# -----------------------------------------------------------

df_icd10_stays = df_icd10_stays.merge(
    patients[
        [
            "subject_id",
            "anchor_year_group"
        ]
    ],
    on="subject_id",
    how="left",
    validate="many_to_one"
)

# Anfang und Ende der Anchor-Year-Gruppe extrahieren
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

# Zeitliche Verschiebung des Aufenthalts relativ zum Anchor-Jahr
df_icd10_stays["year_offset"] = (
    df_icd10_stays["admittime"].dt.year
    - df_icd10_stays["anchor_year"]
)

# Ungefährer realer Aufnahmezeitraum je Aufenthalt
df_icd10_stays["estimated_year_min"] = (
    df_icd10_stays["anchor_group_start"]
    + df_icd10_stays["year_offset"]
)

df_icd10_stays["estimated_year_max"] = (
    df_icd10_stays["anchor_group_end"]
    + df_icd10_stays["year_offset"]
)

# Fehlende Schätzwerte prüfen
print(
    "Aufenthalte ohne schätzbaren realen Zeitraum:",
    df_icd10_stays[
        ["estimated_year_min", "estimated_year_max"]
    ].isna().any(axis=1).sum()
)

# Ungefährer Gesamtzeitraum der ICD-10-Kohorte
estimated_start = df_icd10_stays["estimated_year_min"].min()
estimated_end = df_icd10_stays["estimated_year_max"].max()

print(
    "Ungefährer realer Aufnahmezeitraum der ICD-10-Kohorte: "
    f"{int(estimated_start)} bis {int(estimated_end)}"
)


# -----------------------------------------------------------
# 3. Klassenverteilung der Zielvariable hospital_expire_flag
# -----------------------------------------------------------

counts = (
    df_icd10_stays["hospital_expire_flag"]
    .value_counts(dropna=False)
    .sort_index()
)

props = (
    df_icd10_stays["hospital_expire_flag"]
    .value_counts(dropna=False, normalize=True)
    .sort_index()
    * 100
)

print("Klassenverteilung hospital_expire_flag (absolut):")
print(counts)

print("\nKlassenverteilung hospital_expire_flag (%):")
print(props.round(2))

# Mortalitätsrate: 2.48 %


# -----------------------------------------------------------
# 4. Verteilung der Geschlechter auf Patientenebene
# -----------------------------------------------------------

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
    .value_counts(dropna=False, normalize=True)
    * 100
)

print("Gender – absolute Häufigkeiten:")
print(gender_counts)

print("\nGender – Prozent:")
print(gender_props.round(2))


# -----------------------------------------------------------
# 5. Alter bei Aufnahme berechnen und beschreiben
# -----------------------------------------------------------

df_icd10_stays["admission_year"] = (
    df_icd10_stays["admittime"].dt.year
)

df_icd10_stays["age_at_admission"] = (
    df_icd10_stays["anchor_age"]
    + (
        df_icd10_stays["admission_year"]
        - df_icd10_stays["anchor_year"]
    )
)

print("\nAlter bei Aufnahme – deskriptive Statistik:")
print(
    df_icd10_stays["age_at_admission"]
    .describe()
)

plt.figure(figsize=(8, 5))
plt.hist(
    df_icd10_stays["age_at_admission"].dropna(),
    bins=30
)
plt.title("Alter bei Aufnahme")
plt.xlabel("Alter bei Aufnahme")
plt.ylabel("Anzahl Aufenthalte")
plt.tight_layout()
plt.show()


# Altersgruppen
age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 90, np.inf]

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

# Fehlende Werte für das Diagramm ausschließen
age_group_plot = (
    df_icd10_stays["age_group"]
    .value_counts()
    .sort_index()
)

ax = age_group_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Altersverteilung der ICD-10-Analysekohorte")
ax.set_xlabel("Alter bei Aufnahme")
ax.set_ylabel("Anzahl Aufenthalte")
ax.tick_params(axis="x", rotation=45)

# Absolute Häufigkeiten oberhalb der Balken anzeigen
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.0f",
        padding=3
    )

plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 6. Art und Ort der Aufnahme analysieren
# -----------------------------------------------------------

def analyze_categorical(df, col, top_n=20):
    counts = df[col].value_counts(dropna=False)

    props = (
        df[col]
        .value_counts(dropna=False, normalize=True)
        * 100
    )

    print(f"\n=== {col} ===")
    print("Absolute Häufigkeiten:")
    print(counts.head(top_n))

    print("\nProzent:")
    print(props.head(top_n).round(2))

    print(
        f"\nAnzahl einzigartiger Kategorien: "
        f"{df[col].nunique(dropna=True)}"
    )

    print(
        f"Missing: {df[col].isna().sum()}"
    )


analyze_categorical(
    df_icd10_stays,
    "admission_type"
)

analyze_categorical(
    df_icd10_stays,
    "admission_location"
)

# Diagramme
for col in ["admission_type", "admission_location"]:
    plt.figure(figsize=(10, 5))

    (
        df_icd10_stays[col]
        .value_counts(dropna=False)
        .plot(kind="bar")
    )

    plt.title(f"Ausprägungen von {col}")
    plt.xlabel(col)
    plt.ylabel("Anzahl Aufenthalte")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 7. Length of Stay prüfen
# -----------------------------------------------------------

adm = df_icd10_stays.copy()

# LOS in Tagen
adm["los_days"] = (
    adm["dischtime"] - adm["admittime"]
).dt.total_seconds() / (24 * 3600)

# Gültige zeitliche Reihenfolge: Entlassung muss strikt nach der Aufnahme liegen
adm["valid_time_order"] = (adm["admittime"] < adm["dischtime"])

print("Anzahl Aufenthalte mit admittime >= dischtime:")
print((~adm["valid_time_order"]).sum())
# 81 Aufenthalte mit admittime >= dischtime (negativer LOS)

# Verteilung von LOS
print("\nLOS deskriptiv:")
print(
    adm["los_days"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]
    )
)

# Extrem lange Aufenthalte definieren, z. B. > 99.5%-Quantil
threshold = (adm["los_days"].quantile(0.995))
    
extreme_long_los = adm[adm["los_days"] > threshold]
    
print(f"\nSchwelle für extrem lange LOS (99.5%-Quantil): {threshold:.2f} Tage")
print(f"Anzahl Aufenthalte darüber: {len(extreme_long_los)}")
# 99.5%-Quantil: 50.37 Tage

# Alternativ harte Schwelle, z. B. > 180 Tage
very_long_los = adm[ adm["los_days"] > 180]

print(f"\nAnzahl Aufenthalte > 180 Tage: {len(very_long_los)}")
# Anzahl Aufenthalte > 180 Tage: 28

# Problemfälle ansehen
problem_cases = adm[
    (~adm["valid_time_order"])
    | (adm["los_days"] > threshold)
]

print("\nBeispiele problematischer Aufenthalte:")
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


# -------------------------------
# 8. hospital_expire_flag und deathtime auf Inkonsistenzen prüfen
# -------------------------------

adm = df_icd10_stays.copy()

# Fall 1: Tod im Krankenhaus-Flag = 1, aber deathtime fehlt
case1 = adm[(adm["hospital_expire_flag"] == 1) & (adm["deathtime"].isna())]

# Fall 2: Tod im Krankenhaus-Flag = 0, aber deathtime vorhanden
case2 = adm[(adm["hospital_expire_flag"] == 0) & (adm["deathtime"].notna())]

# Fall 3: deathtime vorhanden, aber nicht innerhalb des Aufenthalts
case3 = adm[
    adm["deathtime"].notna() &
    (
        (adm["deathtime"] < adm["admittime"]) |
        (adm["deathtime"] > adm["dischtime"])
    )
].copy()

print("Inkonsistenzen:")
print(f"1) hospital_expire_flag = 1, aber deathtime fehlt: {len(case1)}")
print(f"2) hospital_expire_flag = 0, aber deathtime vorhanden: {len(case2)}")
print(f"3) deathtime liegt nicht im Aufenthaltszeitraum: {len(case3)}")
# 1. flag = 1, aber deathtime = 0: 7 Aufenthalte
# 2. flag = 0, aber deathtime = 1: 0 Aufenthalte
# 3. deathtime außerhalb vom Aufenthaltszeitraum: 467 Aufenthalte

print("\nBeispiele Fall 1:")
print(case1[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]].head(10))

print("\nBeispiele Fall 2:")
print(case2[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]].head(10))

print("\nBeispiele Fall 3:")
print(case3[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]].head(10))

# Fälle prüfen, wo deathtime nicht im Aufenthaltszeitraum liegt (case 3)
case3["death_before_admit"] = (case3["deathtime"] < case3["admittime"]).astype(int)
case3["death_after_discharge"] = (case3["deathtime"] > case3["dischtime"]).astype(int)
case3["delta_from_discharge_hours"] = (
    (case3["deathtime"] - case3["dischtime"]).dt.total_seconds() / 3600
)

print(case3["hospital_expire_flag"].value_counts(dropna=False))
print(case3[["death_before_admit", "death_after_discharge"]].sum())
print(case3["delta_from_discharge_hours"].describe())
print(case3[[
    "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
    "hospital_expire_flag", "death_before_admit", "death_after_discharge",
    "delta_from_discharge_hours"
]].head(30))
# in betrachteten Fällen ist die Abweichung immer < 24h


# -----------------------------------------------------------
# 8. Todesfälle innerhalb der ersten 48 und 24 Stunden
# -----------------------------------------------------------

adm = df_icd10_stays.copy()

# Zeit bis Tod in Stunden 
adm["time_to_death_hours"] = (
    (adm["deathtime"] - adm["admittime"]).dt.total_seconds() / 3600
)

# Definition: Tod innerhalb 48h
adm["death_within_48h"] = (
    (adm["hospital_expire_flag"] == 1)
    & adm["time_to_death_hours"]
        .between(0, 48, inclusive="both")
)

deaths = adm[adm["hospital_expire_flag"] == 1]

n_deaths_total = len(deaths)
n_deaths_48h = deaths["death_within_48h"].sum()

share_48h = n_deaths_48h / len(adm) * 100

share_48h_deaths = n_deaths_48h / n_deaths_total * 100

# Todesfälle innerhalb der ersten 24h
adm["death_within_24h"] = (
    (adm["hospital_expire_flag"] == 1)
    & adm["time_to_death_hours"]
        .between(0, 24, inclusive="both")
)

n_deaths_24h = adm["death_within_24h"].sum()

share_24h = n_deaths_24h / len(adm) * 100

share_24h_deaths = n_deaths_24h / n_deaths_total * 100

print(f"Gesamte Todesfälle: {n_deaths_total}")
print(f"Davon innerhalb 48h: {n_deaths_48h}")
print(f"Anteil an allen Aufenthalten: {share_48h:.2f} %")
print(f"Anteil an allen Todesfällen: {share_48h_deaths:.2f} %")

print(f"\nTodesfälle innerhalb 24h: {n_deaths_24h}")
print(f"Anteil an allen Aufenthalten: {share_24h:.2f} %")
print(f"Anteil an allen Todesfällen: {share_24h_deaths:.2f} %")


# -------------------------------
# 10. Codes pro Aufenthalt
# -------------------------------

def analyze_icd_table(df, name="diagnoses_icd"):
    d = df.copy()
    
    # Grundchecks
    print(f"\n===== {name} =====")
    print("Zeilen:", len(d))
    print("Missing icd_code:", d["icd_code"].isna().sum())
    
    # Normalisieren
    d["icd_code_clean"] = d["icd_code"].astype(str).str.strip().str.upper()
       
    # Häufigkeiten
    code_freq = (
        d.groupby("icd_code_clean")["hadm_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    print("\nAnzahl eindeutiger Codes:", d["icd_code_clean"].nunique())
    
    # Long Tail
    rare_1 = (code_freq == 1).sum()
    rare_lt_5 = (code_freq < 5).sum()
    rare_lt_10 = (code_freq < 10).sum()
    
    print("\nLong-Tail-Metriken:")
    print(f"Codes in genau einem Aufenthalt: {rare_1}")
    print(f"Codes in weniger als 5 Aufenthalten: {rare_lt_5}")
    print(f"Codes in weniger als 10 Aufenthalten: {rare_lt_10}")
    
    # Kumulative Abdeckung der Top-N-Codes
    total_code_stays = code_freq.sum()
    
    for n in [10, 50, 100, 500]:
        coverage = code_freq.head(n).sum() / total_code_stays * 100
        print(
            f"Top {n} Codes decken {coverage:.2f} % "
            "aller eindeutigen Codevorkommen auf Aufenthaltsebene ab"
        )
    
    return d, code_freq


diagnoses_icd10 = df_icd10[
    df_icd10["code_type"] == "diagnosis"
].copy()

procedures_icd10 = df_icd10[
    df_icd10["code_type"] == "procedure"
].copy()

diag_checked, diag_freq = analyze_icd_table(
    diagnoses_icd10,
    "ICD-10-Diagnosecodes"
)

proc_checked, proc_freq = analyze_icd_table(
    procedures_icd10,
    "ICD-10-Behandlungscodes"
)

# Anzahl eindeutiger Diagnosecodes pro Aufenthalt
diag_per_hadm = (
    diagnoses_icd10
    .assign(
        icd_code_clean=(
            diagnoses_icd10["icd_code"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
    )
    .groupby("hadm_id")["icd_code_clean"]
    .nunique()
)

print("Verteilung der Diagnosecodes pro Aufenthalt")
print(diag_per_hadm.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

diag_counts = (
    diag_per_hadm
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 5))
plt.bar(
    diag_counts.index,
    diag_counts.values
)

plt.title("Anzahl eindeutiger Diagnosecodes pro Aufenthalt")
plt.xlabel("Anzahl eindeutiger Diagnosecodes")
plt.ylabel("Anzahl Aufenthalte")
plt.xticks(range(int(diag_counts.index.min()), int(diag_counts.index.max()) + 1, 2))
plt.tight_layout()
plt.show()

# Anzahl eindeutiger Behandlungscodes pro Aufenthalt
proc_per_hadm = (
    procedures_icd10
    .assign(
        icd_code_clean=(
            procedures_icd10["icd_code"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
    )
    .groupby("hadm_id")["icd_code_clean"]
    .nunique()
)

print("Verteilung der Behandlungscodes pro Aufenthalt")
print(proc_per_hadm.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

proc_counts = (
    proc_per_hadm
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 5))
plt.bar(
    proc_counts.index,
    proc_counts.values
)

plt.title("Anzahl eindeutiger Behandlungscodes pro Aufenthalt")
plt.xlabel("Anzahl eindeutiger Behandlungscodes")
plt.ylabel("Anzahl Aufenthalte")
plt.xticks(range(int(proc_counts.index.min()), int(proc_counts.index.max()) + 1, 2))
plt.tight_layout()
plt.show()


# -------------------------------
# 11. Längenverteilung der ICD-10-Diagnosecodes
# -------------------------------

diag_codes_length = (
    diagnoses_icd10[
        diagnoses_icd10["icd_code"].notna()
    ]
    .copy()
)

diag_codes_length["icd_code_clean"] = (
    diag_codes_length["icd_code"]
    .astype("string")
    .str.replace(".", "", regex=False)
    .str.strip()
    .str.upper()
)

diag_codes_length["code_len"] = (
    diag_codes_length["icd_code_clean"]
    .str.len()
)

code_length_counts = (
    diag_codes_length["code_len"]
    .value_counts()
    .sort_index()
)

print("Absolute Häufigkeiten der Codelängen:")
print(code_length_counts)

code_length_share = (
    diag_codes_length["code_len"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nRelative Häufigkeiten der Codelängen (%):")
print(code_length_share.round(2))


# Längenverteilung der eindeutigen ICD-10-Diagnosecodes
unique_diag_codes = (
    diag_codes_length[
        ["icd_code_clean", "code_len"]
    ]
    .drop_duplicates(subset=["icd_code_clean"])
    .copy()
)

unique_code_length_counts = (
    unique_diag_codes["code_len"]
    .value_counts()
    .sort_index()
)

unique_code_length_share = (
    unique_diag_codes["code_len"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nAbsolute Häufigkeiten der Codelängen – eindeutige Diagnosecodes:")
print(unique_code_length_counts)

print("\nRelative Häufigkeiten der Codelängen – eindeutige Diagnosecodes (%):")
print(unique_code_length_share.round(2))




### =============================
### TRANSFORMATION VOM ICD-10-DATAFRAME 
### =============================

df_icd10_clean = df_icd10.copy()

# Vorhandensein von exakten Duplikaten prüfen -> 0
print("Exakte Duplikate:", df_icd10_clean.duplicated().sum())

# -------------------------------
# 1. Berechnung vom Alter zum Zeitpunkt der Aufnahme 
#    - neue Spalte age_at_admission
#    - anchor_age und anchor_year löschen)
# -------------------------------

df_icd10_clean["admission_year"] = df_icd10_clean["admittime"].dt.year
df_icd10_clean["age_at_admission"] = (
    df_icd10_clean["anchor_age"] + (df_icd10_clean["admission_year"] - df_icd10_clean["anchor_year"])
)

print(
    df_icd10_clean[
        [
            "subject_id",
            "hadm_id",
            "anchor_age",
            "anchor_year",
            "admission_year",
            "age_at_admission"
        ]
    ]
    .drop_duplicates(subset=["hadm_id"])
    .head(10)
)
df_icd10_clean = df_icd10_clean.drop(columns=["anchor_age", "anchor_year", "admission_year"])

# -------------------------------
# 2. Fehlenden Wert in admission_location behandeln
# -------------------------------

df_icd10_clean["admission_location"] = (
    df_icd10_clean["admission_location"]
    .fillna("INFORMATION NOT AVAILABLE")
)

# Check
# Verteilung nur auf Ebene eindeutiger Aufenthalte
admission_location_staylevel = (
    df_icd10_clean[["hadm_id", "admission_location"]]
    .drop_duplicates(subset=["hadm_id"])
)

print("Verteilung von admission_location auf Aufenthaltsebene:")
print(
    admission_location_staylevel["admission_location"]
    .value_counts(dropna=False)
)
# INFORMATION NOT AVAILABLE - vorher: 385
# INFORMATION NOT AVAILABLE - jetzt: 386
# kein NaN-Wert mehr

# -------------------------------
# 3. Kategoriale Variablen bereinigen
# -------------------------------

cat_cols = [
    "gender",
    "admission_type",
    "admission_location",
    "code_type",
    "icd_code",
    "code_description"
]

for col in cat_cols:
    if col in df_icd10_clean.columns:
        df_icd10_clean[col] = (
            df_icd10_clean[col]
            .astype("string")
            .str.strip()
        )

# -------------------------------
# 4. Exportieren
# -------------------------------
df_icd10_clean.to_csv(output_path / "icd10_clean_codelevel.csv", index=False)

print("Export abgeschlossen.")










# ========================================================================================
# ANALYSE DER CODEHÄUFIGKEITEN UND AUSWIRKUNGEN VERSCHIEDENER MINDESTHÄUFIGKEITSSCHWELLEN
# ========================================================================================

import matplotlib.pyplot as plt

CODE_COL = "icd_code"  
TYPE_COL = "code_type"            
TARGET_COL = "hospital_expire_flag"  
HADM_COL = "hadm_id"

# Nur gültige Codes betrachten
df_filtered = df_icd10_clean.copy()
df_filtered = df_filtered[df_filtered[CODE_COL].notna()].copy()

print(df_filtered[TYPE_COL].value_counts(dropna=False))
print(df_filtered[TARGET_COL].value_counts(dropna=False))

print("Anzahl Zeilen mit ICD-Code:", len(df_filtered))
print("Anzahl eindeutiger ICD-Codes:", df_filtered[CODE_COL].nunique())

df_diag = df_filtered.loc[df_filtered[TYPE_COL] == "diagnosis"].copy()
df_proc = df_filtered.loc[df_filtered[TYPE_COL] == "procedure"].copy()


# Hilfsfunktionen
def build_code_frequency_table(
    df_sub,
    code_col=CODE_COL,
    hadm_col=HADM_COL
):
    # Anzahl eindeutiger Aufenthalte pro Code
    freq = (
        df_sub.loc[df_sub[code_col].notna()]
        .groupby(code_col)[hadm_col]
        .nunique()
        .reset_index(name="n")
    )

    # Anteil an der Summe der Code-Aufenthalts-Häufigkeiten
    freq["share"] = freq["n"] / freq["n"].sum()

    return freq

def summarize_rare_code_impact(df_sub, rare_codes, code_col=CODE_COL,
                               target_col=TARGET_COL, hadm_col=HADM_COL):
    temp = df_sub.copy()
    
    affected_rows = temp[temp[code_col].isin(rare_codes)].copy()
    
    total_rows = len(temp)
    total_admissions = temp[hadm_col].nunique()
    total_deaths = temp.loc[temp[target_col] == 1, hadm_col].nunique()
    
    affected_admissions = affected_rows[hadm_col].nunique()
    affected_deaths = affected_rows.loc[affected_rows[target_col] == 1, hadm_col].nunique()
    
    return {
        "n_rare_codes": len(rare_codes),
        "rows_affected": len(affected_rows),
        "share_rows_affected": len(affected_rows) / total_rows if total_rows else np.nan,
        "admissions_affected": affected_admissions,
        "share_admissions_affected": affected_admissions / total_admissions if total_admissions else np.nan,
        "deaths_affected": affected_deaths,
        "share_deaths_affected": affected_deaths / total_deaths if total_deaths else np.nan,
    }


def summarize_thresholds(df_sub, code_col=CODE_COL, target_col=TARGET_COL, hadm_col=HADM_COL,
                         thresholds=(2, 3, 5, 10, 20, 30, 50, 100)):
    freq = build_code_frequency_table(df_sub, code_col)
    total_codes = freq[code_col].nunique()
    
    results = []
    
    for t in thresholds:
        rare_codes = set(freq.loc[freq["n"] < t, code_col])
        
        impact = summarize_rare_code_impact(
            df_sub=df_sub,
            rare_codes=rare_codes,
            code_col=code_col,
            target_col=target_col,
            hadm_col=hadm_col
        )
        
        impact["min_count"] = t
        impact["share_rare_codes"] = len(rare_codes) / total_codes if total_codes else np.nan
        
        results.append(impact)
    
    return pd.DataFrame(results)



def summarize_quantiles(df_sub, code_col=CODE_COL, target_col=TARGET_COL, hadm_col=HADM_COL,
                        quantiles=(0.05, 0.10, 0.25, 0.50, 0.60, 0.75)):
    freq = build_code_frequency_table(df_sub, code_col)
    total_codes = freq[code_col].nunique()
    
    results = []
    
    for q in quantiles:
        threshold = freq["n"].quantile(q)
        rare_codes = set(freq.loc[freq["n"] <= threshold, code_col])
        
        impact = summarize_rare_code_impact(
            df_sub=df_sub,
            rare_codes=rare_codes,
            code_col=code_col,
            target_col=target_col,
            hadm_col=hadm_col
        )
        
        impact["quantile"] = q
        impact["freq_threshold"] = threshold
        impact["share_rare_codes"] = len(rare_codes) / total_codes if total_codes else np.nan
        
        results.append(impact)
    
    return pd.DataFrame(results)    


def plot_code_frequency_distribution(df, code_col="icd_code", title_prefix=""):
    temp = df[df[code_col].notna()].copy()
    freq = build_code_frequency_table(temp, code_col)

    plt.figure(figsize=(8, 5))
    plt.hist(freq["n"], bins=50)
    plt.xlabel("Anzahl Aufenthalte pro Code")
    plt.ylabel("Anzahl Codes")
    plt.title(f"{title_prefix} Verteilung der Code-Häufigkeiten")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(freq["n"], bins=50, log=True)
    plt.xlabel("Anzahl Aufenthalte pro Code")
    plt.ylabel("Anzahl Codes (log)")
    plt.title(f"{title_prefix} Verteilung der Code-Häufigkeiten (log y)")
    plt.show()

    ranked = freq.sort_values("n", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    plt.figure(figsize=(8, 5))
    plt.plot(ranked["rank"], ranked["n"])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rang des Codes (log)")
    plt.ylabel("Häufigkeit (log)")
    plt.title(f"{title_prefix} Rank-Frequency-Plot")
    plt.show()

def explain_quantile(df, code_col="icd_code", q=0.60):
    freq = build_code_frequency_table(df, code_col)
    threshold = freq["n"].quantile(q)

    n_codes_leq = (freq["n"] <= threshold).sum()
    total_codes = len(freq)

    print(f"Quantil: {q}")
    print(f"Schwellenwert der Häufigkeit: {threshold}")
    print(f"Codes mit n <= {threshold}: {n_codes_leq} von {total_codes}")
    print(f"Anteil dieser Codes: {n_codes_leq / total_codes:.4f}")


# --------------
# Diagnosecodes
# --------------
diag_freq = build_code_frequency_table(df_diag, CODE_COL)

print("Diagnosecodes:")
print("Anzahl Zeilen mit Code:", df_diag[CODE_COL].notna().sum())
print("Anzahl eindeutiger Codes:", diag_freq[CODE_COL].nunique())
print("Verteilung der Anzahl eindeutiger Aufenthalte je Diagnosecode:")
print(diag_freq["n"].describe())

# Plots zur Verteilung
plot_code_frequency_distribution(df_diag, CODE_COL, title_prefix="Diagnosecodes")

# Schwellen zur Mindestfallzahl und Quantile
diag_min_count_summary = summarize_thresholds(
    df_diag,
    thresholds=(2, 3, 5, 10, 20, 30, 50)
)

diag_quantile_summary = summarize_quantiles(
    df_diag,
    quantiles=(0.05, 0.10, 0.25, 0.50, 0.60, 0.75)
)

print("Diagnosen: Mindestfallzahl")
print(diag_min_count_summary)

print("\nDiagnosen: Quantile")
print(diag_quantile_summary)

explain_quantile(df_diag, CODE_COL, q=0.60)


# --------------
# Behandlungscodes
# --------------

proc_freq = build_code_frequency_table(df_proc, CODE_COL)

print("Behandlungscodes:")
print("Anzahl Zeilen mit Code:", df_proc[CODE_COL].notna().sum())
print("Anzahl eindeutiger Codes:", proc_freq[CODE_COL].nunique())
print(proc_freq["n"].describe())

# Plots zur Verteilung
plot_code_frequency_distribution(df_proc, CODE_COL, title_prefix="Behandlungscodes")

# Schwellen zur Mindestfallzahl und Quantile
proc_min_count_summary = summarize_thresholds(
    df_proc,
    thresholds=(2, 3, 5, 10, 20, 30, 50)
)

proc_quantile_summary = summarize_quantiles(
    df_proc,
    quantiles=(0.05, 0.10, 0.25, 0.50, 0.60, 0.75)
)

print("Behandlungen: Mindestfallzahl")
print(proc_min_count_summary)

print("\nBehandlungen: Quantile")
print(proc_quantile_summary)





