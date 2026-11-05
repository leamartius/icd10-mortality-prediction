#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALIDIERUNG DER ICD-10-CM-CODEABDECKUNG FÜR MIMIC-IV

Das Skript untersucht die Abdeckung der in MIMIC-IV vorkommenden ICD-10-Diagnosecodes 
durch die ICD-10-CM-Tabular-XML und die offiziellen Codelisten verschiedener Jahrgänge. 
Hierzu werden nicht abgedeckte Codes identifiziert und geprüft, ob frühere Jahresdateien 
zusätzliche Codes erfassen. Die Analyse dient der Auswahl der für die anschließende 
Rekonstruktion der ICD-10-CM-Hierarchie verwendeten Quelldateien.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re

# ============================================================
# 0) Pfade
# ============================================================

# PFADE ANPASSEN!
BASE_PATH = Path("Pfad_zu_ICD-10-CM_Dateien")
DATA_PATH = Path("Pfad_zu_zuvor_gespeicherten_Daten")

XML_PATH = BASE_PATH / "icd10cm-tabular-2023.xml"
CODES_2023_PATH = BASE_PATH / "icd10cm-codes-2023.txt"
MIMIC_CODES_PATH = DATA_PATH / "icd10_clean_codelevel.csv"

order_files = {
    2015: BASE_PATH / "icd10cm_order_2015.txt",
    2016: BASE_PATH / "icd10cm_order_2016.txt",
}

year_files = {
    #2015: BASE_PATH / "icd10cm_codes_2015.txt",
    #2016: BASE_PATH / "icd10cm_codes_2016.txt",
    2017: BASE_PATH / "icd10cm_codes_2017.txt",
    2018: BASE_PATH / "icd10cm_codes_2018.txt",
    2019: BASE_PATH / "icd10cm_codes_2019.txt",
    2020: BASE_PATH / "icd10cm_codes_2020.txt",
    2021: BASE_PATH / "icd10cm_codes_2021.txt",
    2022: BASE_PATH / "icd10cm_codes_2022.txt",
    2023: BASE_PATH / "icd10cm-codes-2023.txt",
}


# ============================================================
# 1) Hilfsfunktionen
# ============================================================

def get_text(elem, tag):
    child = elem.find(tag)
    if child is not None and child.text is not None:
        return child.text.strip()
    return None


def normalize_code(code):
    if pd.isna(code):
        return None
    code = str(code).strip().upper().replace(".", "")
    return code if code else None


def is_valid_icd10cm_length(code):
    if code is None:
        return False
    return len(code) in {3, 4, 5, 6, 7}


def load_xml_codes(xml_path):
    """
    Extrahiert alle Codes, die explizit als <diag><name>...</name></diag>
    in der XML vorkommen.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_codes = []

    for diag_elem in root.findall(".//diag"):
        raw_code = get_text(diag_elem, "name")
        code = normalize_code(raw_code)

        if code is not None and is_valid_icd10cm_length(code):
            xml_codes.append(code)

    xml_df = pd.DataFrame({"icd_code_clean": pd.Series(xml_codes).drop_duplicates()})
    return xml_df


def load_icd10cm_codefile(txt_path, year):
    """
    Liest eine code-Datei im Format:
    A000 Cholera due to Vibrio cholerae ...
    """
    rows = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = re.match(r"^(\S+)\s+(.*)$", line)
            if m:
                raw_code = m.group(1)
                description = m.group(2)
                clean_code = normalize_code(raw_code)

                if is_valid_icd10cm_length(clean_code):
                    rows.append({
                        "icd_code_clean": clean_code,
                        "description": description,
                        "code_year": year
                    })

    return pd.DataFrame(rows)


def process_order_file(file_path):
    """
    Liest eine order-Datei ein und bringt sie ins code-Datei-Format.
    """
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            m = re.match(r"^\s*(\d+)\s+([A-Z0-9]+)\s+([01])\s+(.*)$", line)
            if m:
                rows.append({
                    "icd_code": m.group(2).strip().upper(),
                    "is_billable": int(m.group(3)),
                    "description": m.group(4).strip()
                })

    df = pd.DataFrame(rows)
    df_code_format = df[["icd_code", "description"]].drop_duplicates().copy()
    return df, df_code_format


def summarize_match(mimic_df, reference_set, label):
    matched_mask = mimic_df["icd_code_clean"].isin(reference_set)

    n_total = mimic_df.shape[0]
    n_matched = matched_mask.sum()
    n_unmatched = n_total - n_matched
    match_rate = n_matched / n_total if n_total else np.nan

    print(f"\n--- Match gegen {label} ---")
    print("Gesamtzahl eindeutiger MIMIC-ICD-10-Diagnosecodes:", n_total)
    print("Gematcht:", int(n_matched))
    print("Nicht gematcht:", int(n_unmatched))
    print("Match-Quote:", round(match_rate, 4))

    unmatched_df = mimic_df.loc[~matched_mask].copy()
    return unmatched_df


# ============================================================
# 2) XML laden und auf Code-Ebene prüfen
# ============================================================

xml_df = load_xml_codes(XML_PATH)
xml_code_set = set(xml_df["icd_code_clean"].dropna())

print("Anzahl expliziter Codes in XML:", len(xml_code_set))
print("Codelängen in XML:")
print(xml_df["icd_code_clean"].str.len().value_counts().sort_index())


# ============================================================
# 3) Offizielle 2023-Code-Datei laden
# ============================================================

codes_2023_df = load_icd10cm_codefile(CODES_2023_PATH, year=2023)
codes_2023_set = set(codes_2023_df["icd_code_clean"].dropna())

print("\nAnzahl Codes in offizieller Datei 2023:", len(codes_2023_set))
print("Codelängen in offizieller Datei 2023:")
print(codes_2023_df["icd_code_clean"].str.len().value_counts().sort_index())


# ============================================================
# 4) MIMIC ICD-10 Diagnosecodes laden
# ============================================================

df_icd10_clean = pd.read_csv(MIMIC_CODES_PATH)

df_diag = df_icd10_clean[
    df_icd10_clean["code_type"] == "diagnosis"
].copy()

df_diag["icd_code_clean"] = df_diag["icd_code"].map(normalize_code)

mimic_unique_codes = (
    df_diag[["icd_code", "icd_code_clean", "code_description"]]
    .drop_duplicates()
    .copy()
)

print("\nAnzahl eindeutiger MIMIC-ICD-10-Diagnosecodes:", mimic_unique_codes.shape[0])


# ============================================================
# 5) MIMIC gegen XML und gegen 2023-Code-Datei matchen
# ============================================================

unmatched_vs_xml = summarize_match(
    mimic_df=mimic_unique_codes,
    reference_set=xml_code_set,
    label="XML"
)

print("\nBeispiele nicht gematcht gegen XML:")
print(unmatched_vs_xml[["icd_code", "code_description"]].head(20))

unmatched_vs_2023 = summarize_match(
    mimic_df=mimic_unique_codes,
    reference_set=codes_2023_set,
    label="offizielle Code-Datei 2023"
)

print("\nBeispiele nicht gematcht gegen offizielle Code-Datei 2023:")
print(unmatched_vs_2023[["icd_code", "code_description"]].head(20))


# Aufenthalte mit nicht gematchten Codes
unmatched_code_set = set(unmatched_vs_2023["icd_code_clean"].dropna())

total_admissions = df_diag["hadm_id"].nunique()

admissions_with_unmatched = df_diag[
    df_diag["icd_code_clean"].isin(unmatched_code_set)
]["hadm_id"].nunique()

share_unmatched = admissions_with_unmatched / total_admissions if total_admissions else np.nan

print("\nGesamtanzahl Aufenthalte:", total_admissions)
print("Aufenthalte mit nicht gematchten Codes:", admissions_with_unmatched)
print("Anteil:", round(share_unmatched, 4))


# ============================================================
# 6) 2015/2016 order-Dateien in code-Dateien umwandeln
# ============================================================

converted_order_dfs = {}

for year, file_path in order_files.items():
    if not file_path.exists():
        print(f"Datei fehlt: {file_path}")
        continue

    df_full, df_code = process_order_file(file_path)

    print(f"\n--- Jahr {year}: order-Datei in Code-Format umwandeln ---")
    print("Alle Zeilen:", df_full.shape[0])
    print("Unique Codes:", df_code.shape[0])

    # --------------------------------------------------------
    # Optionaler Export
    # --------------------------------------------------------
    
    output_file = BASE_PATH / f"icd10cm_codes_{year}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for row in df_code.itertuples(index=False):
            f.write(f"{row.icd_code} {row.description}\n")

    print(f"Exportiert: {output_file}")
    

    # --------------------------------------------------------
    # Direkt im Skript weiterverwenden
    # --------------------------------------------------------

    df_code_prepared = df_code.rename(
        columns={"icd_code": "icd_code_clean"}
    ).copy()

    df_code_prepared["icd_code_clean"] = (
        df_code_prepared["icd_code_clean"]
        .map(normalize_code)
    )

    df_code_prepared["code_year"] = year

    df_code_prepared = df_code_prepared[
        [
            "icd_code_clean",
            "description",
            "code_year"
        ]
    ].copy()

    converted_order_dfs[year] = df_code_prepared


# ============================================================
# 7) Code-Dateien mehrerer Jahre laden
# ============================================================

# Direkt aus den order-Dateien erzeugte DataFrames für 2015/2016
annual_dfs = list(converted_order_dfs.values())

# Code-Dateien ab 2017 laden
# (Wenn die Dateien für 2015 und 2016 bereits im richtigen Format exportiert wurden,
# können sie auch hier geladen werden, nachdem sie oben in year_files wieder
# aktiviert wurden.)
for year, file_path in year_files.items():
    if file_path.exists():
        df_year = load_icd10cm_codefile(
            file_path,
            year
        )

        annual_dfs.append(df_year)

        print(
            f"\nJahr {year}: "
            f"{df_year.shape[0]} Codes geladen"
        )
    else:
        print(f"Datei fehlt: {file_path}")

all_years_codes_df = pd.concat(annual_dfs, ignore_index=True).drop_duplicates()

code_years_df = (
    all_years_codes_df
    .groupby("icd_code_clean")
    .agg(
        first_year=("code_year", "min"),
        last_year=("code_year", "max"),
        n_years=("code_year", "nunique"),
        years_present=("code_year", lambda x: sorted(set(x))),
        description_any=("description", "first")
    )
    .reset_index()
)

print("\nAnzahl eindeutiger Codes über alle Jahresdateien:", code_years_df.shape[0])


# ============================================================
# 8) Nicht gegen 2023 gematchte MIMIC-Codes gegen Jahresdateien prüfen
# ============================================================

unmatched_check = unmatched_vs_2023.merge(
    code_years_df,
    on="icd_code_clean",
    how="left"
)

unmatched_check["found_in_any_year"] = unmatched_check["first_year"].notna()

print("\nNicht gegen 2023 gematchte Codes mit Jahresinfo:")
print(unmatched_check[
    [
        "icd_code",
        "code_description",
        "found_in_any_year",
        "first_year",
        "last_year",
        "years_present"
    ]
].head(20))

n_total_unmatched = unmatched_check.shape[0]
n_found_in_any_year = unmatched_check["found_in_any_year"].sum()

print("\nAnzahl der gegen 2023 nicht gematchten Codes:", n_total_unmatched)
print("Davon in mindestens einem anderen Jahresfile gefunden:", int(n_found_in_any_year))
print("Auch über alle Jahresdateien hinweg nicht gefunden:", int(n_total_unmatched - n_found_in_any_year))


# ---------------------------
# 9) Prüfen, ob 2015 allein alle fehelenden Codes abdeckt
# ---------------------------

# Codes, die weder im 2023-Code-File noch in der XML vorkommen
unmatched_vs_xml_and_2023 = mimic_unique_codes.loc[
    (~mimic_unique_codes["icd_code_clean"].isin(codes_2023_set)) &
    (~mimic_unique_codes["icd_code_clean"].isin(xml_code_set))
].copy()

print("Anzahl Codes, die weder durch XML noch durch 2023 abgedeckt sind:",
      unmatched_vs_xml_and_2023.shape[0])

# Jahresinfos anhängen
unmatched_check = unmatched_vs_xml_and_2023.merge(
    code_years_df,
    on="icd_code_clean",
    how="left"
)

unmatched_check["found_in_any_year"] = unmatched_check["first_year"].notna()

# Menge der Codes, die in mindestens einem Jahresfile gefunden werden
codes_found_any_year = set(
    unmatched_check.loc[
        unmatched_check["found_in_any_year"],
        "icd_code_clean"
    ].dropna()
)

# Menge der Codes, die in 2015 gefunden werden
codes_found_2015 = set(
    unmatched_check.loc[
        unmatched_check["years_present"].apply(
            lambda years: isinstance(years, list) and 2015 in years
        ),
        "icd_code_clean"
    ].dropna()
)

# Vergleich
print("Anzahl Codes gefunden in irgendeinem Jahresfile:", len(codes_found_any_year))
print("Anzahl Codes gefunden in 2015:", len(codes_found_2015))

# Codes, die 2015 nicht abdeckt
missing_in_2015 = codes_found_any_year - codes_found_2015

missing_in_2015_df = unmatched_check[
    unmatched_check["icd_code_clean"].isin(missing_in_2015)
][["icd_code_clean", "years_present"]].copy()

print("Codes, die 2015 NICHT abdeckt:")
print(missing_in_2015_df.head())


# Vorkommen vom fehlenden Code Q51.20 in MIMIC prüfen
target_code = "Q5120"

q5120_rows = df_diag[
    df_diag["icd_code_clean"] == target_code
].copy()

n_admissions = q5120_rows["hadm_id"].nunique()
n_patients = q5120_rows["subject_id"].nunique()

print("\n--- Vorkommen von Q51.20 in MIMIC-IV ---")
print("Anzahl eindeutiger Aufenthalte:", n_admissions)
print("Anzahl eindeutiger Patienten:", n_patients)


