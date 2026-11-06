#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAPPING DER CCSR-KATEGORIEN

In dem Skript werden die in MIMIC-IV enthaltenen ICD-10-Diagnose- und Behandlungscodes 
den entsprechenden CCSR-Kategorien zugeordnet und die Qualität der Zuordnung geprüft. 
Anschließend werden die Kategorien auf Aufenthaltsebene als Multi-Hot-Merkmale kodiert, 
zu einem gemeinsamen Feature-Set zusammengeführt und für die weitere Modellierung gespeichert.
"""

import pandas as pd
from pathlib import Path

# PFADE ANPASSEN!
path_mimic = Path("Pfad_zu_zuvor_gespeicherten_Daten")
path_ccsr = Path("Pfad_zu_CCSR-Dateien")

mimic_df = pd.read_csv(path_mimic / "icd10_clean_codelevel.csv", dtype=str)
ccsr_diag = pd.read_csv(path_ccsr / "DXCCSR_v2023-1/DXCCSR_v2023-1.csv", dtype=str)
ccsr_proc = pd.read_csv(path_ccsr / "PRCCSR_v2023-1/PRCCSR_v2023-1.csv", dtype=str)


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def clean_string_dataframe(df):
    """
    Entfernt Leerzeichen und zusätzliche einfache Anführungszeichen
    aus Spaltennamen und Textwerten.
    """
    df = df.copy()

    df.columns = (df.columns.str.strip().str.strip("'"))

    for col in df.columns:
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.strip("'")
            )

    return df


def normalize_icd_code(series):
    return (
        series
        .astype("string")
        .str.upper()
        .str.strip()
        .str.replace(".", "", regex=False)
    )


def prepare_ccsr_mapping(
    mapping_df,
    code_column,
    category_column,
    description_column
):
    """
    Bereitet eine CCSR-Mapping-Tabelle für das spätere Mapping vor.
    """
    mapping = mapping_df[
        [
            code_column,
            category_column,
            description_column
        ]
    ].copy()

    mapping = mapping.rename(
        columns={
            code_column: "icd_code_clean",
            category_column: "ccsr_category",
            description_column: "ccsr_category_description"
        }
    )

    mapping["icd_code_clean"] = normalize_icd_code(
        mapping["icd_code_clean"]
    )

    # Vollständig leere Mapping-Zeilen entfernen
    mapping = mapping.dropna(
        subset=[
            "icd_code_clean",
            "ccsr_category"
        ]
    )

    # Sicherheitsprüfung auf doppelte Codes
    duplicate_mask = mapping["icd_code_clean"].duplicated(keep=False)

    n_duplicate_rows = duplicate_mask.sum()

    if n_duplicate_rows > 0:
        duplicate_codes = (
            mapping.loc[
                duplicate_mask,
                "icd_code_clean"
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Die Mapping-Tabelle enthält Codes mit mehreren "
            f"Mapping-Zeilen: {duplicate_codes[:10]}"
        )

    return mapping


def create_ccsr_features(
    mimic_df,
    mapping_df,
    code_type,
    feature_prefix,
    label
):
    """
    Führt für eine Codeart das vollständige CCSR-Mapping durch:
    1. Filterung nach code_type
    2. Mapping auf CCSR
    3. Prüfung der Mapping-Qualität
    4. Entfernung mehrfach vorkommender Kategorien je Aufenthalt
    5. Erstellung der Multi-Hot-Matrix
    """

    # -------------------------------------------------------------------------
    # Codeart auswählen
    # -------------------------------------------------------------------------
    code_df = mimic_df.loc[
        mimic_df["code_type"].eq(code_type)
    ].copy()

    if code_df.empty:
        raise ValueError(
            f"Für code_type='{code_type}' wurden keine Zeilen gefunden."
        )

    # -------------------------------------------------------------------------
    # Mapping
    # -------------------------------------------------------------------------
    mapped_df = code_df.merge(
        mapping_df,
        on="icd_code_clean",
        how="left",
        validate="many_to_one"
    )

    # Sicherstellen, dass der Merge keine Zeilen vervielfacht hat
    if len(mapped_df) != len(code_df):
        raise ValueError(
            f"Beim Mapping von {label} hat sich die Zeilenzahl verändert."
        )

    # -------------------------------------------------------------------------
    # Mapping-Qualität überprüfen
    # -------------------------------------------------------------------------
    n_total = len(mapped_df)
    n_mapped = mapped_df["ccsr_category"].notna().sum()
    n_unmapped = mapped_df["ccsr_category"].isna().sum()

    unique_code_check = (
        mapped_df[
            [
                "icd_code_clean",
                "ccsr_category"
            ]
        ]
        .drop_duplicates()
    )

    unique_mapping_rate = (
        unique_code_check["ccsr_category"]
        .notna()
        .mean()
    )

    unmapped_codes = (
        mapped_df.loc[
            mapped_df["ccsr_category"].isna()
        ]
        .groupby(
            "icd_code_clean",
            dropna=False
        )
        .agg(
            n_unique_stays=("hadm_id", "nunique"),
            n_rows=("hadm_id", "size")
        )
        .sort_values(
            [
                "n_unique_stays",
                "n_rows"
            ],
            ascending=False
        )
        .reset_index()
    )

    summary = {
        "label": label,
        "n_rows": n_total,
        "n_mapped_rows": n_mapped,
        "n_unmapped_rows": n_unmapped,
        "n_unique_codes": mapped_df["icd_code_clean"].nunique(),
        "n_unique_unmapped_codes": len(unmapped_codes),
        "unique_code_mapping_rate": unique_mapping_rate
    }

    print("\n" + "=" * 70)
    print(f"CCSR-MAPPING: {label.upper()}")
    print("=" * 70)

    print(f"Codezeilen insgesamt: {n_total:,}")
    print(f"Gemappte Codezeilen: {n_mapped:,}")
    print(f"Nicht gemappte Codezeilen: {n_unmapped:,}")

    print(
        f"Eindeutige ICD-Codes insgesamt: "
        f"{summary['n_unique_codes']:,}"
    )

    print(
        f"Nicht gemappte eindeutige ICD-Codes: "
        f"{summary['n_unique_unmapped_codes']:,}"
    )

    print(
        f"Mapping-Rate der eindeutigen ICD-Codes: "
        f"{unique_mapping_rate:.2%}"
    )

    if not unmapped_codes.empty:
        print("\nNicht gemappte Codes:")
        print(unmapped_codes.head(35))
    else:
        print("\nAlle Codes wurden gemappt.")

    # -------------------------------------------------------------------------
    # CCSR-Kategorien je Aufenthalt
    # -------------------------------------------------------------------------
    hadm_categories = (
        mapped_df.loc[
            mapped_df["ccsr_category"].notna(),
            [
                "subject_id",
                "hadm_id",
                "ccsr_category",
                "ccsr_category_description"
            ]
        ]
        .drop_duplicates(
            subset=[
                "subject_id",
                "hadm_id",
                "ccsr_category"
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Multi-Hot-Encoding
    # -------------------------------------------------------------------------
    multihot_df = pd.crosstab(
        index=[
            hadm_categories["subject_id"],
            hadm_categories["hadm_id"]
        ],
        columns=hadm_categories["ccsr_category"]
    )

    multihot_df = multihot_df.astype("uint8")
    
    multihot_df.columns = [
        f"{feature_prefix}{category}"
        for category in multihot_df.columns
    ]
    
    multihot_df = multihot_df.reset_index()

    if multihot_df["hadm_id"].duplicated().any():
        raise ValueError(
            f"Das Multi-Hot-DataFrame für {label} enthält "
            "doppelte hadm_id."
        )

    print(
        f"\nAnzahl Aufenthalte mit mindestens einer "
        f"gemappten Kategorie: {len(multihot_df):,}"
    )

    print(
        f"Anzahl erzeugter CCSR-Features: "
        f"{sum(col.startswith(feature_prefix) for col in multihot_df.columns)}"
    )

    return (
        mapped_df,
        unmapped_codes,
        hadm_categories,
        multihot_df,
        summary
    )


# =============================================================================
# DATEN BEREINIGEN
# =============================================================================

ccsr_diag = clean_string_dataframe(ccsr_diag)
ccsr_proc = clean_string_dataframe(ccsr_proc)

mimic_df["icd_code_clean"] = normalize_icd_code(
    mimic_df["icd_code"]
)

mimic_df["code_type"] = (
    mimic_df["code_type"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("MIMIC-Codearten:")
print(mimic_df["code_type"].value_counts(dropna=False))

print("\nCCSR-Diagnosecodes-Datei:")
print(ccsr_diag.shape)
print(ccsr_diag.columns.tolist())

print("\nCCSR-Behandlungscodes-Datei:")
print(ccsr_proc.shape)
print(ccsr_proc.columns.tolist())


# =============================================================================
# MAPPING-TABELLEN VORBEREITEN
# =============================================================================
# Diagnosen:
# - Es wird die Spalte "Default CCSR CATEGORY IP" verwendet. 
# - Dadurch erhält jeder ICD-10-Code genau eine Standardkategorie 
#   und es entsteht keine komplizierte Mehrfachzuordnung.
# - Außerdem handelt es sich bei dem MIMIC-Datensatz um stationäre Patienten (IP).

ccsr_diag_mapping = prepare_ccsr_mapping(
    mapping_df=ccsr_diag,
    code_column="ICD-10-CM CODE",
    category_column="Default CCSR CATEGORY IP",
    description_column="Default CCSR CATEGORY DESCRIPTION IP"
)

# Behandlungen
ccsr_proc_mapping = prepare_ccsr_mapping(
    mapping_df=ccsr_proc,
    code_column="ICD-10-PCS",
    category_column="PRCCSR",
    description_column="PRCCSR DESCRIPTION"
)

print("\nDiagnose-Mapping-Tabelle:")
print(ccsr_diag_mapping.shape)
print(ccsr_diag_mapping.head())

print("\nBehandlung-Mapping-Tabelle:")
print(ccsr_proc_mapping.shape)
print(ccsr_proc_mapping.head())

# =============================================================================
# DIAGNOSEN MAPPEN
# =============================================================================
(
    diagnoses_ccsr,
    unmapped_diag_ccsr_codes,
    hadm_diag_ccsr_categories,
    diag_ccsr_multihot,
    diag_mapping_summary
) = create_ccsr_features(
    mimic_df=mimic_df,
    mapping_df=ccsr_diag_mapping,
    code_type="diagnosis",
    feature_prefix="diag_ccsr_",
    label="Diagnosen"
)

# =============================================================================
# BEHANDLUNGEN MAPPEN
# =============================================================================
(
    procedures_ccsr,
    unmapped_proc_ccsr_codes,
    hadm_proc_ccsr_categories,
    proc_ccsr_multihot,
    proc_mapping_summary
) = create_ccsr_features(
    mimic_df=mimic_df,
    mapping_df=ccsr_proc_mapping,
    code_type="procedure",
    feature_prefix="proc_ccsr_",
    label="Behandlungen"
)

# =============================================================================
# DIAGNOSE- UND BEHANDLUNGSFEATURES VERBINDEN
# =============================================================================

ccsr_multihot_all = diag_ccsr_multihot.merge(
    proc_ccsr_multihot,
    on=["subject_id", "hadm_id"],
    how="outer",
    validate="one_to_one"
)

ccsr_feature_columns = [
    col
    for col in ccsr_multihot_all.columns
    if col.startswith(
        (
            "diag_ccsr_",
            "proc_ccsr_"
        )
    )
]

ccsr_multihot_all[ccsr_feature_columns] = (
    ccsr_multihot_all[ccsr_feature_columns]
    .fillna(0)
    .astype("uint8")
)

if ccsr_multihot_all["hadm_id"].duplicated().any():
    raise ValueError(
        "Das gemeinsame CCSR-DataFrame enthält doppelte hadm_id."
    )

n_diag_features = sum(
    col.startswith("diag_ccsr_")
    for col in ccsr_multihot_all.columns
)

n_proc_features = sum(
    col.startswith("proc_ccsr_")
    for col in ccsr_multihot_all.columns
)

print("\n" + "=" * 70)
print("GEMEINSAMES CCSR-FEATURE-DATAFRAME")
print("=" * 70)

print(f"Form: {ccsr_multihot_all.shape}")
print(f"Anzahl Diagnose-CCSR-Features: {n_diag_features}")
print(f"Anzahl Behandlung-CCSR-Features: {n_proc_features}")
print(f"Anzahl CCSR-Features insgesamt: {len(ccsr_feature_columns)}")

print(
    "Doppelte hadm_id:",
    ccsr_multihot_all["hadm_id"].duplicated().sum()
)


# =============================================================================
# PRÜFUNG: AUFENTHALTE OHNE GEMAPPTE CCSR-KATEGORIE
# =============================================================================

all_stays = (
    mimic_df[
        ["subject_id", "hadm_id"]
    ]
    .drop_duplicates()
)

mapped_stays = (
    ccsr_multihot_all[
        ["subject_id", "hadm_id"]
    ]
    .drop_duplicates()
)

stays_without_ccsr = (
    all_stays
    .merge(
        mapped_stays,
        on=["subject_id", "hadm_id"],
        how="left",
        indicator=True
    )
    .loc[lambda x: x["_merge"].eq("left_only")]
    .drop(columns="_merge")
    .reset_index(drop=True)
)

print(
    "\nAnzahl Aufenthalte ohne irgendeine gemappte CCSR-Kategorie:",
    len(stays_without_ccsr)
)


# =============================================================================
# SPEICHERN
# =============================================================================

ccsr_multihot_all.to_pickle(path_mimic / "ccsr_multihot.pkl")



"""
Ergebnisse
    
- Diagnosecodes: 
    - 31 eindeutige Codes konnten nicht gemappt werden.
    - Mapping-Rate: 99.84 %
    - Alle davon kommen nur bei jeweils einem Aufenthalt vor, daher können sie ignoriert werden.
    - 474 CCSR-Features (Multi-Hot)
- Behandlungscodes:
    - Alle Codes konnten gemappt werden.
    - 305 CCSR-Features (Multi-Hot)
- Es gibt keine Aufenthalte ohne irgendeine gemappte CCSR-Kategorie
"""

