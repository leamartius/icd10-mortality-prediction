#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERZEUGUNG TAXONOMIEBASIERTER DIAGNOSECODEMERKMALE

In dem Skript wird aus den ICD-10-CM-Tabular- und Code-Dateien eine vollständige 
hierarchische Diagnosetaxonomie rekonstruiert. Daraus werden strukturelle Merkmale auf Codeebene
berechnet. Die erzeugten Metriken werden mit den in MIMIC-IV vorkommenden Diagnosecodes verknüpft, 
deskriptiv und hinsichtlich ihrer Beziehung zur Krankenhausmortalität untersucht sowie um 
Prävalenz-, Information-Content- und kategoriale Merkmale ergänzt. Abschließend werden die 
Merkmale mit dem vollständigen Code-Level-Datensatz zusammengeführt und exportiert.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 0) Pfade
# ============================================================

# PFADE ANPASSEN!
BASE_PATH = Path("Pfad_zu_ICD-10-CM_Dateien") 
DATA_PATH = Path("Pfad_zu_zuvor_gespeicherten_Daten")

XML_PATH = BASE_PATH / "icd10cm-tabular-2023.xml"
CODES_PATHS = {
    2015: BASE_PATH / "icd10cm_codes_2015.txt",
    2019: BASE_PATH / "icd10cm_codes_2019.txt",
    2023: BASE_PATH / "icd10cm-codes-2023.txt"
}
MIMIC_CODES_PATH = DATA_PATH / "icd10_clean_codelevel.csv"


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


def get_direct_parent_code(code):
    code = normalize_code(code)
    if code is None:
        return None

    n = len(code)
    if n <= 3:
        return None
    return code[:-1]


def get_all_prefix_nodes(code):
    code = normalize_code(code)
    if code is None:
        return []

    n = len(code)
    if n < 3:
        return []

    return [code[:k] for k in range(3, n + 1)]


# ============================================================
# 2) Tabular XML parsen
# ============================================================

tree = ET.parse(XML_PATH)
root = tree.getroot()

xml_records = []

def walk_diag(diag_elem, parent_code=None, depth=1, chapter_desc=None, section_desc=None):
    raw_code = get_text(diag_elem, "name")
    raw_desc = get_text(diag_elem, "desc")

    code = normalize_code(raw_code)

    if code is not None:
        xml_records.append({
            "icd_code": code,
            "code_description_tabular": raw_desc,
            "parent_code_xml": parent_code,
            "depth_xml": depth,
            "chapter_desc": chapter_desc,
            "section_desc": section_desc
        })
        this_parent = code
        next_depth = depth + 1
    else:
        this_parent = parent_code
        next_depth = depth

    for child_diag in diag_elem.findall("diag"):
        walk_diag(
            diag_elem=child_diag,
            parent_code=this_parent,
            depth=next_depth,
            chapter_desc=chapter_desc,
            section_desc=section_desc
        )


chapters = root.findall(".//chapter")

for chapter in chapters:
    chapter_desc = get_text(chapter, "desc")

    sections = chapter.findall("section")
    if sections:
        for section in sections:
            section_desc = get_text(section, "desc")
            for diag in section.findall("diag"):
                walk_diag(
                    diag_elem=diag,
                    parent_code=None,
                    depth=1,
                    chapter_desc=chapter_desc,
                    section_desc=section_desc
                )
    else:
        for diag in chapter.findall("diag"):
            walk_diag(
                diag_elem=diag,
                parent_code=None,
                depth=1,
                chapter_desc=chapter_desc,
                section_desc=None
            )

xml_df = (
    pd.DataFrame(xml_records)
    .drop_duplicates(subset=["icd_code"])
    .copy()
)

print("Anzahl XML-Codes:", xml_df["icd_code"].nunique())
print("\nCodelängen in der XML:")
print(xml_df["icd_code"].str.len().value_counts().sort_index())


# ============================================================
# 3) Offizielle ICD-10-CM Code-Dateien laden und vereinen
# ============================================================

def load_icd10cm_codefile(txt_path, year):
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
                        "icd_code_raw": raw_code,
                        "icd_code_clean": clean_code,
                        "description_official": description,
                        "code_year": year
                    })

    return pd.DataFrame(rows)


code_dfs = []

for year, file_path in CODES_PATHS.items():
    df_year = load_icd10cm_codefile(file_path, year)
    code_dfs.append(df_year)

    print(f"\nAnzahl Codes in offizieller Datei {year}: {df_year.shape[0]}")
    print(df_year["icd_code_clean"].str.len().value_counts().sort_index())

codes_union_df = pd.concat(code_dfs, ignore_index=True).copy()

# Für jeden Code festhalten, in welchen Jahren er vorkommt
code_years_union_df = (
    codes_union_df
    .groupby("icd_code_clean")
    .agg(
        icd_code_raw=("icd_code_raw", "first"),
        description_official=("description_official", "first"),
        first_year=("code_year", "min"),
        last_year=("code_year", "max"),
        n_years=("code_year", "nunique"),
        years_present=("code_year", lambda x: sorted(set(x)))
    )
    .reset_index()
)

codes_df = code_years_union_df.copy()

print("\nAnzahl eindeutiger Codes in vereinigten offiziellen Dateien:", codes_df.shape[0])
print("Codelängen in vereinigter offizieller Codebasis:")
print(codes_df["icd_code_clean"].str.len().value_counts().sort_index())


# ============================================================
# 4) Vollständige Knotenmenge aufbauen
#    - alle offiziellen Codes
#    - alle notwendigen Präfixe
#    - alle XML-Knoten
# ============================================================

all_codes = set()
official_code_set = set(codes_df["icd_code_clean"].dropna())
xml_code_set = set(xml_df["icd_code"].dropna())

# 4.1 Alle offiziellen Codes und deren Präfixe
for code in official_code_set:
    for prefix in get_all_prefix_nodes(code):
        all_codes.add(prefix)

# 4.2 Zusätzlich alle XML-Knoten absichern
all_codes.update(xml_code_set)

# Nur gültige Längen behalten
all_codes = {c for c in all_codes if is_valid_icd10cm_length(c)}

all_codes_df = pd.DataFrame({"icd_code": sorted(all_codes)})

print("\nAnzahl Knoten im vollständigen Graphen:", all_codes_df.shape[0])
print("Codelängen im vollständigen Graphen:")
print(all_codes_df["icd_code"].str.len().value_counts().sort_index())


# ============================================================
# 5) Metadaten für vollständigen Graphen zusammensetzen
# ============================================================

# XML-Metadaten
xml_meta_df = xml_df[[
    "icd_code",
    "code_description_tabular",
    "chapter_desc",
    "section_desc"
]].copy()

# Offizielle Beschreibungen
official_meta_df = codes_df[[
    "icd_code_clean",
    "description_official"
]].rename(columns={"icd_code_clean": "icd_code"}).copy()

graph_df = all_codes_df.merge(
    xml_meta_df,
    on="icd_code",
    how="left"
).merge(
    official_meta_df,
    on="icd_code",
    how="left"
)

# Knotentyp
graph_df["is_official_code"] = graph_df["icd_code"].isin(official_code_set).astype(int)
graph_df["is_in_xml"] = graph_df["icd_code"].isin(xml_code_set).astype(int)

graph_df["node_type"] = np.select(
    [
        (graph_df["is_official_code"] == 1) & (graph_df["is_in_xml"] == 1),
        (graph_df["is_official_code"] == 1) & (graph_df["is_in_xml"] == 0),
        (graph_df["is_official_code"] == 0) & (graph_df["is_in_xml"] == 1),
        (graph_df["is_official_code"] == 0) & (graph_df["is_in_xml"] == 0),
    ],
    [
        "official_and_xml",
        "official_only",
        "xml_only",
        "inserted_prefix_only"
    ],
    default="unknown"
)

# Beschreibung bevorzugt aus XML, sonst aus offizieller Datei
graph_df["code_description"] = graph_df["code_description_tabular"].combine_first(
    graph_df["description_official"]
)

# Parent streng nach direktem Präfix
graph_df["parent_code"] = graph_df["icd_code"].map(get_direct_parent_code)

# Root-Status
graph_df["is_root"] = graph_df["parent_code"].isna().astype(int)


# ============================================================
# 6) Kapitel-/Section-Infos auf ergänzte Knoten vererben
#    Falls ein eingefügter Knoten nicht in XML war, werden Infos
#    vom nächsten vorhandenen Präfix-Vorfahren übernommen.
# ============================================================

xml_lookup = xml_df.set_index("icd_code")[["chapter_desc", "section_desc"]].to_dict("index")

def inherit_xml_context(code):
    """
    Sucht den nächsten Präfix-Vorfahren, der in XML vorkommt,
    und übernimmt chapter_desc/section_desc.
    """
    code = normalize_code(code)
    if code is None:
        return pd.Series([None, None])

    for k in range(len(code), 2, -1):
        candidate = code[:k]
        if candidate in xml_lookup:
            return pd.Series([
                xml_lookup[candidate].get("chapter_desc"),
                xml_lookup[candidate].get("section_desc")
            ])
    return pd.Series([None, None])

missing_context_mask = graph_df["chapter_desc"].isna() | graph_df["section_desc"].isna()

graph_df.loc[missing_context_mask, ["chapter_desc", "section_desc"]] = (
    graph_df.loc[missing_context_mask, "icd_code"]
    .apply(inherit_xml_context)
    .values
)


# ============================================================
# 7) Kinderstruktur des vollständigen Graphen
# ============================================================

children_dict = defaultdict(list)
parent_dict = {}

for row in graph_df[["icd_code", "parent_code"]].itertuples(index=False):
    parent_dict[row.icd_code] = row.parent_code
    if pd.notna(row.parent_code):
        children_dict[row.parent_code].append(row.icd_code)

for code in graph_df["icd_code"]:
    _ = children_dict[code]


# ============================================================
# 8) Diagnosekapitel als Feature generieren
# ============================================================

# Funktion zum Extrahieren des Kapitel-Codes (z.B. A00-B99)
def extract_chapter_code(chapter_desc):
    if pd.isna(chapter_desc):
        return None
    
    match = re.search(r"\(([A-Z]\d{2}-[A-Z0-9]{2,3})\)", chapter_desc)
    if match:
        return match.group(1)
    return None

# Neue Spalte erzeugen
graph_df["chapter_code"] = graph_df["chapter_desc"].apply(extract_chapter_code)

# Kontrolle
print(graph_df[["chapter_desc", "chapter_code"]].drop_duplicates().head(30))


# ============================================================
# 9) Semantische Metriken berechnen
# ============================================================

# 9.1 Tiefe und normalisierte Tiefe 
@lru_cache(maxsize=None)
def compute_depth(code):
    parent = parent_dict.get(code)
    if parent is None or pd.isna(parent):
        return 1
    return compute_depth(parent) + 1

graph_df["depth"] = graph_df["icd_code"].map(compute_depth)

max_depth = graph_df["depth"].max()
graph_df["normalized_depth"] = graph_df["depth"] / max_depth


# 9.2 Blattinddikator
graph_df["is_leaf"] = graph_df["icd_code"].map(
    lambda c: 1 if len(children_dict[c]) == 0 else 0
)


# 9.3 Anzahl direkter Geschwister inkl. sich selbst
def count_siblings_including_self(code):
    parent = parent_dict.get(code)
    if parent is None or pd.isna(parent):
        return 0
    return len(children_dict[parent])

graph_df["n_direct_siblings"] = graph_df["icd_code"].map(count_siblings_including_self)


# 9.4 Subtree-Größe mit Memoization
@lru_cache(maxsize=None)
def count_descendants(code):
    children = children_dict[code]
    return len(children) + sum(count_descendants(child) for child in children)

graph_df["subtree_size"] = graph_df["icd_code"].map(count_descendants)


# 9.5 Relative Subtree-Größe

# Mittelwert je Tiefe berechnen
mean_subtree_by_depth = graph_df.groupby("depth")["subtree_size"].mean()

# Mapping auf DataFrame
graph_df["mean_subtree_depth"] = graph_df["depth"].map(mean_subtree_by_depth)

# Neue Definition der relativen Subtree-Größe
graph_df["relative_subtree_size"] = np.where(
    graph_df["depth"] == 5,
    1.0,  # alle Blätter auf tiefster Ebene → durchschnittlich
    np.where(
        graph_df["subtree_size"] == 0,
        0.0,  # kein Subtree auf höheren Ebenen → minimal
        graph_df["subtree_size"] / graph_df["mean_subtree_depth"]
    )
)

# Aufräumen
graph_df.drop(columns=["mean_subtree_depth"], inplace=True)

# log-transformierte relativen Subtree-Größe (weil stark rechtsschief verteilt)
graph_df["relative_subtree_size_log"] = np.log1p(
    graph_df["relative_subtree_size"]
)

# ============================================================
# 10) Qualitätschecks des vollständigen Graphen
# ============================================================

print("\nNode types:")
print(graph_df["node_type"].value_counts(dropna=False))

invalid_parent_rows = graph_df[
    graph_df["parent_code"].notna() &
    (~graph_df["parent_code"].isin(graph_df["icd_code"]))
]

print("\nFehlende Parent-Knoten:", invalid_parent_rows.shape[0])

# Prüfung: Parent muss genau eine Stelle kürzer sein
parent_length_errors = graph_df[
    graph_df["parent_code"].notna() &
    ((graph_df["icd_code"].str.len() - graph_df["parent_code"].str.len()) != 1)
]

print("Parent-Längenfehler:", parent_length_errors.shape[0])

# Beispielkontrolle
example_codes = ["E11", "E119"]
print("\nBeispielstruktur:")
print(
    graph_df[graph_df["icd_code"].isin(example_codes)][
        ["icd_code", "parent_code", "chapter_code", "depth", "node_type", "code_description"]
    ].sort_values(["icd_code"])
)


# ============================================================
# 11) Finales Semantik-DataFrame
# ============================================================

semantic_metrics_df = graph_df[[
    "icd_code",
    "code_description",
    "chapter_code",
    "chapter_desc",
    "section_desc",
    "parent_code",
    "depth",
    "normalized_depth",
    "is_leaf",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log",
    "is_official_code",
    "is_in_xml",
    "node_type"
]].copy()

print("\nsemantic_metrics_df:")
print(semantic_metrics_df.head())
print(semantic_metrics_df.shape)


# ============================================================
# 12) MIMIC ICD-10 Diagnosecodes laden und matchen
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

mimic_with_semantics = mimic_unique_codes.merge(
    semantic_metrics_df,
    left_on="icd_code_clean",
    right_on="icd_code",
    how="left",
    suffixes=("_mimic", "_graph")
)

n_total = mimic_with_semantics.shape[0]
n_matched = mimic_with_semantics["depth"].notna().sum()

print("\nEindeutige MIMIC-Diagnosecodes:", n_total)
print("Direkt im vollständigen Graphen gematcht:", n_matched)
print("Nicht gematcht:", n_total - n_matched)
print("Match-Quote:", round(n_matched / n_total, 4))

unmatched = mimic_with_semantics[
    mimic_with_semantics["depth"].isna()
].copy()

print("\nNicht gematchte Codes:")
print(unmatched[["icd_code_mimic", "code_description_mimic"]].head(30))


# ============================================================
# 13) Einbettung künstlich eingefügter Knoten prüfen
# ============================================================

# Mehrere künstlich eingefügte Knoten mit Parent anzeigen
inserted_examples = semantic_metrics_df.loc[
    semantic_metrics_df["node_type"] == "inserted_prefix_only",
    ["icd_code", "parent_code"]
].head(10)

print(inserted_examples)

# Einen künstlich eingefügten Präfix-Knoten auswählen
example_inserted_code = (
    semantic_metrics_df.loc[
        semantic_metrics_df["node_type"] == "inserted_prefix_only", "icd_code"
    ]
    .iloc[0]
)

print("Beispiel künstlich eingefügter Code:", example_inserted_code)

# Parent und direkte Kinder dieses Knotens anzeigen
example_parent = semantic_metrics_df.loc[
    semantic_metrics_df["icd_code"] == example_inserted_code, "parent_code"
].iloc[0]

example_children = semantic_metrics_df.loc[
    semantic_metrics_df["parent_code"] == example_inserted_code, "icd_code"
].tolist()

print("Parent:", example_parent)
print("Kinder:", example_children)


# ============================================================
# 14) Statistik zu den Metriken im gesamten Hierarchiegraphen
# ============================================================

# ------------------------------------------
# 14.1) Gesamtstatistik
# ------------------------------------------

metric_cols = [
    "depth",
    "normalized_depth",
    "is_leaf",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log"
]

# Codelänge ergänzen
semantic_metrics_df["code_length"] = semantic_metrics_df["icd_code"].str.len()

# Nur numerische Metriken für manche Auswertungen
numeric_metric_cols = [
    "depth",
    "normalized_depth",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log"
]

# Gesamtstatistik für alle Knoten
overall_stats = semantic_metrics_df[metric_cols].describe().T

# zusätzliche Kennzahlen ergänzen
overall_stats["missing"] = semantic_metrics_df[metric_cols].isna().sum()
overall_stats["skew"] = semantic_metrics_df[metric_cols].skew(numeric_only=True)

print("\nGesamtstatistik der semantischen Metriken:")
print(overall_stats)


# ------------------------------------------
# 14.2) Metriken nach Tiefe
# ------------------------------------------

metric_cols_no_depth = [
    "normalized_depth",
    "is_leaf",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log"
]

stats_by_depth = (
    semantic_metrics_df
    .groupby("depth")[metric_cols_no_depth]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\nMetriken nach Tiefe:")
print(stats_by_depth)

depth_summary = (
    semantic_metrics_df
    .groupby("depth")
    .agg(
        n_nodes=("icd_code", "count"),
        leaf_share=("is_leaf", "mean"),
        mean_siblings=("n_direct_siblings", "mean"),
        median_siblings=("n_direct_siblings", "median"),
        mean_subtree_size=("subtree_size", "mean"),
        median_subtree_size=("subtree_size", "median"),
        mean_relative_subtree=("relative_subtree_size", "mean"),
        median_relative_subtree=("relative_subtree_size", "median"),
        mean_relative_subtree_log=("relative_subtree_size_log", "mean"),
        median_relative_subtree_log=("relative_subtree_size_log", "median")
    )
)

print("\nKompakte Zusammenfassung nach Tiefe:")
print(depth_summary)


# ------------------------------------------
# 14.3) Metriken nach Kapitel
# ------------------------------------------

stats_by_chapter = (
    semantic_metrics_df
    .groupby("chapter_code")[metric_cols]
    .agg(["count", "mean", "median", "std"])
)

print("\nMetriken nach ICD-Kapitel:")
print(stats_by_chapter)

chapter_summary = (
    semantic_metrics_df
    .groupby("chapter_code")
    .agg(
        n_nodes=("icd_code", "count"),
        mean_depth=("depth", "mean"),
        median_depth=("depth", "median"),
        leaf_share=("is_leaf", "mean"),
        mean_siblings=("n_direct_siblings", "mean"),
        median_siblings=("n_direct_siblings", "median"),
        mean_subtree_size=("subtree_size", "mean"),
        median_subtree_size=("subtree_size", "median"),
        mean_relative_subtree_size=("relative_subtree_size", "mean"),
        median_relative_subtree_size=("relative_subtree_size", "median"),
        mean_subtree_size_log=("relative_subtree_size_log", "mean"),
        median_subtree_size_log=("relative_subtree_size_log", "median")
    )
    .sort_values("n_nodes", ascending=False)
)

print("\nKompakte Kapitel-Zusammenfassung:")
print(chapter_summary)
# sortiert nach n_codes


# ============================================================
# 15) Vergleich: Ganzer Graph vs. MIMIC-Codes
# ============================================================

mimic_code_set = set(df_diag["icd_code_clean"].dropna().unique())

semantic_metrics_df["used_in_mimic"] = (
    semantic_metrics_df["icd_code"].isin(mimic_code_set).astype(int)
)

# Vergleichstabelle
comparison_all_vs_mimic = (
    semantic_metrics_df
    .groupby("used_in_mimic")[metric_cols]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\nVergleich aller Knoten vs. in MIMIC verwendete Codes:")
print(comparison_all_vs_mimic)

# Blattanteil
leaf_comparison = pd.DataFrame({
    "gesamter_graph": {
        "n_codes": len(semantic_metrics_df),
        "n_leaf": semantic_metrics_df["is_leaf"].sum(),
        "n_non_leaf": (semantic_metrics_df["is_leaf"] == 0).sum(),
        "leaf_share": semantic_metrics_df["is_leaf"].mean()
    },
    "mimic_codes": {
        "n_codes": semantic_metrics_df["used_in_mimic"].sum(),
        "n_leaf": semantic_metrics_df.loc[
            semantic_metrics_df["used_in_mimic"] == 1,
            "is_leaf"
        ].sum(),
        "n_non_leaf": (
            semantic_metrics_df.loc[
                semantic_metrics_df["used_in_mimic"] == 1,
                "is_leaf"
            ] == 0
        ).sum(),
        "leaf_share": semantic_metrics_df.loc[
            semantic_metrics_df["used_in_mimic"] == 1,
            "is_leaf"
        ].mean()
    }
}).T

print("\nVergleich des Blattanteils:")
print(leaf_comparison.round(4))


# ============================================================
# 16) Statistik der in MIMIC genutzten Codes
# ============================================================

# ------------------------------------------
# 16.1) Overall Statistik
# ------------------------------------------

mimic_code_freq = (
    df_diag["icd_code_clean"]
    .value_counts(dropna=True)
    .rename_axis("icd_code")
    .reset_index(name="n_occurrences_mimic")
)

semantic_with_freq = semantic_metrics_df.merge(
    mimic_code_freq,
    on="icd_code",
    how="left"
)

semantic_with_freq["n_occurrences_mimic"] = semantic_with_freq["n_occurrences_mimic"].fillna(0)

# nur tatsächlich genutzte Codes
used_codes_with_freq = semantic_with_freq[
    semantic_with_freq["n_occurrences_mimic"] > 0
].copy()

mimic_stats = used_codes_with_freq[metric_cols].describe().T

# zusätzliche Kennzahlen ergänzen
mimic_stats["missing"] = used_codes_with_freq[metric_cols].isna().sum()
mimic_stats["skew"] = used_codes_with_freq[metric_cols].skew(numeric_only=True)

print("\nStatistik der in MIMIC genutzten Codes:")
print(mimic_stats)

# ------------------------------------------
# 16.2) Visualisierung der Verteilungen der Metriken
# ------------------------------------------

semantic_mimic_df = semantic_metrics_df[
    semantic_metrics_df["icd_code"].isin(mimic_code_set)
].copy()


metrics_to_plot = [
    "depth",
    "is_leaf",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log"
]

for col in metrics_to_plot:

    values = semantic_mimic_df[col].dropna()

    if col in ["relative_subtree_size", "relative_subtree_size_log"]:
        values = values.round(2)

    counts = values.value_counts().sort_index()

    # --------------------------------------------------
    # depth
    # --------------------------------------------------
    if col == "depth":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index.astype(str), counts.values, width=0.5)
        ax.set_title(f"Verteilung von {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Anzahl eindeutiger Codes in MIMIC-IV")
        plt.xticks(rotation=0, ha="center")
        fig.tight_layout()
        plt.show()
        continue

    # --------------------------------------------------
    # relative_subtree_size_log
    # --------------------------------------------------
    if col == "relative_subtree_size_log":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index.astype(str), counts.values)
    
        ax.set_title(f"Verteilung von {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Anzahl eindeutiger Codes in MIMIC-IV")
    
        step = 4
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)
    
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        plt.show()
        
        # Plot 2: Zoom ohne dominante Werte
        counts_zoom = counts[~counts.index.isin([0.00, 0.69])]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts_zoom.index.astype(str), counts_zoom.values)
        ax.set_title(f"Verteilung von {col} – ohne 0 und 0.69")
        ax.set_xlabel(col)
        ax.set_ylabel("Anzahl eindeutiger Codes in MIMIC-IV")

        if len(counts_zoom) > 20:
            step = max(1, len(counts_zoom) // 15)
            for i, label in enumerate(ax.get_xticklabels()):
                label.set_visible(i % step == 0)

        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        plt.show()
        continue

    # --------------------------------------------------
    # Standard
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Verteilung von {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Anzahl eindeutiger Codes in MIMIC-IV")

    if len(counts) > 20:
        step = max(1, len(counts) // 15)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plt.show()


# ------------------------------------------
# 16.3) Korrelationen zwischen Metriken (nur MIMIC-Codes)
# ------------------------------------------

correlation_matrix = semantic_mimic_df[numeric_metric_cols].corr(method="spearman")

print("\nSpearman-Korrelationen zwischen den Metriken:")
print(correlation_matrix)

# Korrelationsmatrix als Graphik darstellen
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    correlation_matrix,
    vmin=-1,
    vmax=1,
    cmap="Blues",
    aspect="equal"
)
# Achsenbeschriftungen
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.index)))

ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)
ax.set_yticklabels(correlation_matrix.index)

# Korrelationswerte in die Felder schreiben
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        value = correlation_matrix.iloc[i, j]
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

# Farbskala und Titel
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Spearman-Korrelationskoeffizient")

ax.set_title("Spearman-Korrelationen der Diagnosecode-Metriken auf Codeebene")

fig.tight_layout()

plt.show()


# ============================================================
# 17) Mortalitätsrate pro ICD-Code berechnen
#       - Seltene Codes entfernen, da deren Mortalitätsrate instabil ist
#       - Ein Code wird hier als selten definiert, wenn er in weniger als 10 Aufenthalten vorkommmt
# ============================================================

# ------------------------------------------------------------
# 17.1) Eindeutige Kombinationen aus Aufenthalt + ICD-Code
#       Falls ein Code innerhalb eines Aufenthalts mehrfach vorkommt,
#       wird hospital_expire_flag per max() konsolidiert.
# ------------------------------------------------------------

df_code_adm = (
    df_diag[["hadm_id", "icd_code_clean", "hospital_expire_flag"]]
    .dropna(subset=["icd_code_clean"])
    .groupby(["hadm_id", "icd_code_clean"], as_index=False)["hospital_expire_flag"]
    .max()
)

min_admissions = 10

mortality_by_code = (
    df_code_adm
    .groupby("icd_code_clean")
    .agg(
        n_admissions=("hadm_id", "nunique"),
        n_deaths=("hospital_expire_flag", "sum")
    )
    .copy()
)

mortality_by_code["mortality_rate_raw"] = (
    mortality_by_code["n_deaths"]
    / mortality_by_code["n_admissions"]
)

# Nur Mortalitätsraten ausreichend häufiger Codes verwenden
mortality_by_code["mortality_rate"] = (
    mortality_by_code["mortality_rate_raw"]
    .where(mortality_by_code["n_admissions"] >= min_admissions)
)

print("\nMortalitätsrate pro ICD-Code:")
print(mortality_by_code.head())


# ------------------------------------------------------------
# 17.2) Analysedatensatz auf Ebene eindeutiger Diagnosecodes
# ------------------------------------------------------------

code_mortality_analysis_df = (
    mimic_with_semantics[
        [
            "icd_code_clean",
            "chapter_code",
            "parent_code",
            "depth",
            "normalized_depth",
            "is_leaf",
            "n_direct_siblings",
            "subtree_size",
            "relative_subtree_size",
            "relative_subtree_size_log",
            "node_type"
        ]
    ]
    .drop_duplicates(subset=["icd_code_clean"])
    .merge(
        mortality_by_code.reset_index(),
        on="icd_code_clean",
        how="left"
    )
    .rename(columns={
        "icd_code_clean": "icd_code"    
    })
)

# Kontrolle: tatsächlich eine Zeile pro Code
assert code_mortality_analysis_df["icd_code"].is_unique

print("\nAnalysedatensatz auf Codeebene:")
print(code_mortality_analysis_df.head())
print("Anzahl eindeutiger Codes:", len(code_mortality_analysis_df))

# Seltene Codes entfernen
code_mortality_analysis_filtered_df = (
    code_mortality_analysis_df[
        code_mortality_analysis_df["n_admissions"] >= min_admissions
    ]
    .copy()
)

print(
    "\nAusgeschlossene Codes mit weniger als 10 Aufenthalten:",
    (code_mortality_analysis_df["n_admissions"] < min_admissions).sum()
)

print(
    "Verbleibende Codes:",
    len(code_mortality_analysis_filtered_df)
)

assert code_mortality_analysis_filtered_df["icd_code"].is_unique
assert code_mortality_analysis_filtered_df["mortality_rate"].notna().all()
assert (
    code_mortality_analysis_filtered_df["n_admissions"] >= min_admissions
).all()


# ------------------------------------------------------------
# 17.3) Mortalitätswerte an eindeutige MIMIC-Codes anhängen
# ------------------------------------------------------------

mimic_with_semantics["n_admissions"] = (
    mimic_with_semantics["icd_code_clean"]
    .map(mortality_by_code["n_admissions"])
)

mimic_with_semantics["n_deaths"] = (
    mimic_with_semantics["icd_code_clean"]
    .map(mortality_by_code["n_deaths"])
)

mimic_with_semantics["mortality_rate"] = (
    mimic_with_semantics["icd_code_clean"]
    .map(mortality_by_code["mortality_rate"])
)

print("\nMIMIC-Codes mit Semantik und Mortalität:")
print(mimic_with_semantics.head())


# ------------------------------------------------------------
# 17.4) Code-Spalten vereinheitlichen und redundante Spalten entfernen
# ------------------------------------------------------------

mimic_with_semantics["icd_code"] = (
    mimic_with_semantics["icd_code_clean"]
)

mimic_with_semantics = mimic_with_semantics.drop(
    columns=[
        "icd_code_mimic",
        "icd_code_clean",
        "icd_code_graph",
        "code_description_graph",
        "is_official_code",
        "is_in_xml"
    ],
    errors="ignore"
)

# ICD-Code als erste Spalte anordnen
cols = (
    ["icd_code"]
    + [col for col in mimic_with_semantics.columns if col != "icd_code"]
)

mimic_with_semantics = mimic_with_semantics[cols]


# ------------------------------------------------------------
# 17.5) Checks
# ------------------------------------------------------------

print("\nCheck Mortalitätsraten:")
print(mimic_with_semantics["mortality_rate"].describe())

print("\nBeispiele mit hoher Mortalitätsrate:")
print(
    mimic_with_semantics.sort_values("mortality_rate", ascending=False)[
        ["icd_code", "n_admissions", "n_deaths", "mortality_rate"]
    ].head(10)
)

print("\nCodes insgesamt:")
print(len(mimic_with_semantics))

print("\nCodes mit gültiger Mortalitätsrate:")
print(mimic_with_semantics["mortality_rate"].notna().sum())

print("\nSeltene Codes mit fälschlich vorhandener Mortalitätsrate:")
print(
    mimic_with_semantics.loc[
        mimic_with_semantics["n_admissions"] < min_admissions,
        "mortality_rate"
    ].notna().sum()
)


# ============================================================
# 18) Korrelationsanalyse zwischen Diagnosemetriken und Mortalitätsrate
# ============================================================

# Bereits gefilterter Datensatz mit genau einer Zeile pro Diagnosecode
df_corr = code_mortality_analysis_filtered_df.copy()
df_corr["mortality_rate"] = pd.to_numeric(df_corr["mortality_rate"], errors="coerce")

assert df_corr["icd_code"].is_unique
assert df_corr["mortality_rate"].notna().all()
assert (df_corr["n_admissions"] >= min_admissions).all()

print("\nDatengrundlage der Korrelationsanalyse:")
print("Anzahl eindeutiger Codes:", len(df_corr))

# ------------------------------------------------------------
# 18.1) Pearson- und Spearman-Korrelationen
# ------------------------------------------------------------

corr_metric_cols = [
    "normalized_depth",
    "is_leaf",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log",
    "n_admissions"  # Kontrollvariable
]

corr_results = []

for metric in corr_metric_cols:
    sub = df_corr[[metric, "mortality_rate"]].dropna()

    if len(sub) < 3:
        print(f"{metric}: zu wenige Beobachtungen (n={len(sub)})")
        continue

    if sub[metric].nunique() < 2 or sub["mortality_rate"].nunique() < 2:
        print(f"{metric}: zu wenig Varianz für eine Korrelation")
        continue

    pearson_r, pearson_p = pearsonr(sub[metric], sub["mortality_rate"])
    spearman_rho, spearman_p = spearmanr(sub[metric], sub["mortality_rate"])

    corr_results.append({
        "metric": metric,
        "n": len(sub),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p
    })

corr_results_df = pd.DataFrame(corr_results)

if not corr_results_df.empty:
    corr_results_df = (
        corr_results_df
        .sort_values("spearman_rho", key=lambda x: x.abs(), ascending=False)
        .reset_index(drop=True)
    )

print("\nKorrelationen der Diagnosemetriken mit der Mortalitätsrate:")
print(corr_results_df.round(6).to_string(index=False))

# ------------------------------------------------------------
# 18.2) Visualisierung der Spearman-Korrelationen
# ------------------------------------------------------------

plot_df = (
    corr_results_df[
        corr_results_df["metric"] != "n_admissions"
    ]
    .sort_values("spearman_rho")
    .copy()
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(
    plot_df["metric"],
    plot_df["spearman_rho"]
)

ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Spearman-Korrelationskoeffizient")
ax.set_ylabel("")
ax.set_title(
    "Spearman-Korrelation der kontinuierlichen\n"
    "Diagnosemetriken mit der Mortalitätsrate"
)

limit = max(0.1, plot_df["spearman_rho"].abs().max() * 1.25)
offset = limit * 0.03
ax.set_xlim(-limit, limit)

for bar, value in zip(bars, plot_df["spearman_rho"]):
    x_pos = value + offset if value >= 0 else value - offset
    ax.text(
        x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center",
        ha="left" if value >= 0 else "right"
    )

fig.tight_layout()
plt.show()


# ============================================================
# 19) Kategorien der Diagnosemetriken bilden
# ============================================================

df_metric_categories = code_mortality_analysis_df.copy()

# ------------------------------------------------------------
# 19.1) Tiefe: Kategorie je Ebene
#       - Ergibt auf Codeebene wenig Sinn, aber auf Aufenthaltsebene kann man daraus
#         aggregierte Metriken bauen, die jeweils den Anteil der Codes pro Ebene ausgeben
# ------------------------------------------------------------

# Kategorie je Ebene
df_metric_categories["depth_ord"] = df_metric_categories["depth"].astype("Int64") # bereits 1–5 → perfekt für Spearman

# ------------------------------------------------------------
# 19.2) Geschwisterzahl
# ------------------------------------------------------------

siblings_col = "n_direct_siblings"

# Schwellen durch Terzile bestimmen
q33_siblings = df_metric_categories[siblings_col].quantile(0.33)
q66_siblings = df_metric_categories[siblings_col].quantile(0.66)

print("\nSchwellen n_direct_siblings:")
print("q33:", q33_siblings)
print("q66:", q66_siblings)
# q33 = 3, q66 = 6

df_metric_categories["siblings_3cat"] = pd.cut(
    df_metric_categories[siblings_col],
    bins=[-np.inf, q33_siblings, q66_siblings, np.inf],
    labels=["low", "medium", "high"],
    include_lowest=True
)

print("\nVerteilung Geschwisterzahl:")
print(df_metric_categories["siblings_3cat"].value_counts(dropna=False))

"""
Grenzen:
- low    = <= 3
- medium = 4–6
- high   = > 6
"""

# ------------------------------------------------------------
# 19.3) Relative Subtree-Größe log
#    - binär statt 3 Kategorien
#    - Grund: stark zero-inflated, 3 Kategorien wären extrem unbalanciert
# ------------------------------------------------------------

subtree_col = "relative_subtree_size_log"

df_metric_categories["rel_subtree_log_cat"] = np.select(
    [
        df_metric_categories[subtree_col].isna(),
        df_metric_categories[subtree_col].eq(0),
        df_metric_categories[subtree_col].gt(0)
    ],
    [
        np.nan,
        "zero",
        "nonzero"
    ],
    default=np.nan
)

# "nan" als String vermeiden
df_metric_categories["rel_subtree_log_cat"] = df_metric_categories["rel_subtree_log_cat"].replace("nan", np.nan)

print("Verteilung rel. Subtree-Größe log:")
print(df_metric_categories["rel_subtree_log_cat"].value_counts(dropna=False))


# ============================================================
# 20) Merge mit vollständigem MIMIC-Datensatz (inkl. Behandlungscodes)
# ============================================================

# Linke Tabelle: vollständiger Codelevel-Datensatz mit Diagnosen + Behandlungen
df_mimic_all = df_icd10_clean.copy()

# Merge-Key erzeugen
df_mimic_all["icd_code"] = df_mimic_all["icd_code"].map(normalize_code)

# Rechte Tabelle: nur relevante Semantik-/Mortalitätsspalten auswählen
semantics_lookup = mimic_with_semantics[[
    "icd_code",
    "chapter_code",
    "chapter_desc",
    "section_desc",
    "parent_code",
    "depth",
    "normalized_depth",
    "is_leaf",
    "n_direct_siblings",
    "subtree_size",
    "relative_subtree_size",
    "relative_subtree_size_log",
    "node_type",
    "n_admissions",
    "n_deaths",
    "mortality_rate"
]].copy()

semantics_lookup = semantics_lookup.rename(
    columns={"chapter_code": "icd10_chapter"}
)

# Sicherstellen: eine Zeile pro ICD-Code
semantics_lookup = semantics_lookup.drop_duplicates(subset=["icd_code"])

# Merge
mimic_all_with_semantics = df_mimic_all.merge(
    semantics_lookup,
    on="icd_code",
    how="left"
)


# Checks
print("\nAnzahl der Zeilen im vollständigen Codelevel-Datensatz nach Codetyp:")
print(
    mimic_all_with_semantics["code_type"]
    .value_counts(dropna=False)
)

print("\nAnteil der Zeilen mit erfolgreich gematchten Diagnose-Semantikmetriken:")
print(
    mimic_all_with_semantics
    .groupby("code_type")["depth"]
    .apply(lambda x: x.notna().mean())
    .rename("Anteil mit vorhandener depth")
)

print(
    "\nAnteil der Zeilen mit gültiger Mortalitätsrate "
    f"(nur Diagnosecodes mit mindestens {min_admissions} Aufenthalten):"
)
print(
    mimic_all_with_semantics
    .groupby("code_type")["mortality_rate"]
    .apply(lambda x: x.notna().mean())
    .rename("Anteil mit vorhandener Mortalitätsrate")
)


# ============================================================
# 21) Code-Prävalenz und Information Content (IC) berechnen
# ============================================================

# ------------------------------------------------------------
# 21.1) Diagnosecodes
# ------------------------------------------------------------

diag = mimic_all_with_semantics[
    (mimic_all_with_semantics["code_type"] == "diagnosis")
    & mimic_all_with_semantics["icd_code"].notna()
].copy()

# Anzahl der Aufenthalte mit mindestens einer ICD-10-Diagnose
n_diag_stays = diag["hadm_id"].nunique()

# Anzahl eindeutiger Aufenthalte je Diagnosecode
diag_ic = (
    diag
    .groupby("icd_code")["hadm_id"]
    .nunique()
    .reset_index(name="n_stays_diag")
)

# Prävalenz relativ zu allen Aufenthalten mit mindestens einer Diagnose
diag_ic["prevalence_diag"] = (
    diag_ic["n_stays_diag"] / n_diag_stays
)

# Information Content
diag_ic["ic_diag"] = -np.log(
    diag_ic["prevalence_diag"]
)

# ------------------------------------------------------------
# 21.2) Behandlungscodes
# ------------------------------------------------------------

proc = mimic_all_with_semantics[
    (mimic_all_with_semantics["code_type"] == "procedure")
    & mimic_all_with_semantics["icd_code"].notna()
].copy()

# Anzahl der Aufenthalte mit mindestens einer ICD-10-Behandlung
n_proc_stays = proc["hadm_id"].nunique()

# Anzahl eindeutiger Aufenthalte je Behandlungscode
proc_ic = (
    proc
    .groupby("icd_code")["hadm_id"]
    .nunique()
    .reset_index(name="n_stays_proc")
)

# Prävalenz relativ zu allen Aufenthalten mit mindestens einer Behandlung
proc_ic["prevalence_proc"] = (
    proc_ic["n_stays_proc"] / n_proc_stays
)

# Information Content
proc_ic["ic_proc"] = -np.log(
    proc_ic["prevalence_proc"]
)


# ------------------------------------------------------------
# 21.3) Prävalenz und IC mit dem vollständigen Codelevel-Datensatz mergen
# ------------------------------------------------------------

diag_ic["code_type"] = "diagnosis"
proc_ic["code_type"] = "procedure"

mimic_all_with_semantics = mimic_all_with_semantics.merge(
    diag_ic[
        [
            "icd_code",
            "code_type",
            "prevalence_diag",
            "ic_diag"
        ]
    ],
    on=["icd_code", "code_type"],
    how="left"
)

mimic_all_with_semantics = mimic_all_with_semantics.merge(
    proc_ic[
        [
            "icd_code",
            "code_type",
            "prevalence_proc",
            "ic_proc"
        ]
    ],
    on=["icd_code", "code_type"],
    how="left"
)

# ------------------------------------------------------------
# 21.4) Checks
# ------------------------------------------------------------

print("\nPrävalenz und IC nach Codetyp:")

print(
    mimic_all_with_semantics
    .groupby("code_type")[
        [
            "prevalence_diag",
            "ic_diag",
            "prevalence_proc",
            "ic_proc"
        ]
    ]
    .apply(lambda x: x.notna().mean())
)

print("\nDiagnose-IC:")
print(
    mimic_all_with_semantics.loc[
        mimic_all_with_semantics["code_type"] == "diagnosis",
        "ic_diag"
    ].describe()
)

print("\nBehandlungs-IC:")
print(
    mimic_all_with_semantics.loc[
        mimic_all_with_semantics["code_type"] == "procedure",
        "ic_proc"
    ].describe()
)


# ============================================================
#  22) Export
# ============================================================

# Kategorien je eindeutigem Diagnosecode
category_lookup = df_metric_categories[
    [
        "icd_code",
        "depth_ord",
        "rel_subtree_log_cat",
        "siblings_3cat"
    ]
].drop_duplicates(subset=["icd_code"])

category_lookup["code_type"] = "diagnosis"

# Kategorien an vollständigen Codelevel-Datensatz anhängen
mimic_all_with_semantics = mimic_all_with_semantics.merge(
    category_lookup,
    on=["icd_code", "code_type"],
    how="left"
)

# Check
print(mimic_all_with_semantics.columns)


OUTPUT_PATH = (DATA_PATH / "mimic_with_semantics_categories_mortality_codelevel.csv")
mimic_all_with_semantics.to_csv(OUTPUT_PATH, index=False)


