"""
ERZEUGUNG HISTORISCHER PATIENTENMERKMALE

Das Skript erzeugt auf Aufenthaltsebene Merkmale zur bisherigen
Krankenhaushistorie eines Patienten.
Abschließend werden die erzeugten historischen Merkmale exportiert.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# PFAD ANPASSEN!
path = Path("Pfad_zu_zuvor_gespeicherten_Daten")
df_icd10_codelevel = pd.read_csv(path / "mimic_with_semantics_categories_mortality_codelevel.csv")

# Datums-/Zeitspalten in datetime umwandeln
datetime_cols = ["admittime", "dischtime"]

for col in datetime_cols:
    df_icd10_codelevel[col] = pd.to_datetime(
        df_icd10_codelevel[col],
        errors="coerce"
    )
    
# =========================
# Historische Features vorbereiten
# =========================

# ---------------------------------
# 1) Aufenthaltsebene erzeugen
# ---------------------------------
# Ziel: genau eine Zeile pro Aufenthalt (hadm_id)

stay_history = (
    df_icd10_codelevel[
        ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]
    ]
    .drop_duplicates(subset=["hadm_id"])   # sicherstellen: 1 Zeile pro Aufenthalt
    .sort_values(["subject_id", "admittime", "hadm_id"])  # chronologische Reihenfolge
    .copy()
)

# ---------------------------------
# 2) Length of Stay berechnen
# ---------------------------------
stay_history["los_days"] = (
    stay_history["dischtime"] - stay_history["admittime"]
).dt.total_seconds() / (24 * 3600)

# Negativen LOS prüfen und bereinigen (negativen LOS auf Null setzen)
print("Negative los_days vorher:")
print((stay_history["los_days"] < 0).sum())

stay_history["los_days"] = stay_history["los_days"].clip(lower=0)
print("Negative los_days nachher:")
print((stay_history["los_days"] < 0).sum())

# ---------------------------------
# 3) Anzahl vorheriger Aufenthalte
# ---------------------------------
# cumcount zählt: wie viele Aufenthalte gab es davor?
# Wichtig: aktueller Aufenthalt wird NICHT mitgezählt

stay_history["n_prev_admissions"] = stay_history.groupby("subject_id").cumcount()

# ---------------------------------
# 4) Notfall-Indikator
# ---------------------------------
stay_history["is_emergency_admission"] = stay_history["admission_type"].astype(str).str.upper().isin(
    ["EW EMER.", "URGENT", "DIRECT EMER."]
).astype(int)

# Anzahl vorheriger Notfallaufnahmen
stay_history["n_prev_emergency_admissions"] = (
    stay_history.groupby("subject_id")["is_emergency_admission"]
    .cumsum()
    - stay_history["is_emergency_admission"]  # aktuellen Aufenthalt wieder abziehen
)

# ---------------------------------
# 5) Zeit seit letzter Entlassung
# ---------------------------------
stay_history["prev_dischtime"] = stay_history.groupby("subject_id")["dischtime"].shift(1)

stay_history["time_since_last_discharge_days"] = (
    (stay_history["admittime"] - stay_history["prev_dischtime"])
    .dt.total_seconds() / (24 * 3600)
)

# ---------------------------------
# 6) Wiederaufnahme innerhalb 30 Tage
# ---------------------------------
stay_history["had_admission_last_30d"] = np.where(
    stay_history["prev_dischtime"].notna()
    &
    ((stay_history["admittime"] - stay_history["prev_dischtime"]) <= pd.Timedelta(days=30)),
    1,
    0
)

# ---------------------------------
# 7) Maximaler und mittlerer vergangener LOS
# ---------------------------------
stay_history["max_past_los"] = (
    stay_history.groupby("subject_id")["los_days"]
    .transform(lambda s: s.shift(1).expanding().max())
)

stay_history["avg_past_los"] = (
    stay_history.groupby("subject_id")["los_days"]
    .transform(lambda s: s.shift(1).expanding().mean())
)





# ======================================
# Historische Diagnose- und Behandlungscodes
# ======================================

# Diagnose- und Behandlungs-Daten trennen
diag_df = df_icd10_codelevel[
    df_icd10_codelevel["code_type"] == "diagnosis"
].copy()

proc_df = df_icd10_codelevel[
    df_icd10_codelevel["code_type"] == "procedure"
].copy()


# --------------------------------------
# Mapping: Diagnosecode -> ICD-10-Kapitel
# --------------------------------------
code_to_chapter = (
    diag_df
    .dropna(subset=["icd_code", "icd10_chapter"])
    .drop_duplicates(subset=["icd_code"])
    .set_index("icd_code")["icd10_chapter"]
    .to_dict()
)


# --------------------------------------
# Codes pro Aufenthalt sammeln
# --------------------------------------
diag_stay = (
    diag_df
    .groupby(["subject_id", "hadm_id", "admittime"])["icd_code"]
    .apply(lambda x: set(x.dropna()))
    .reset_index(name="diag_codes_current_stay")
)

proc_stay = (
    proc_df
    .groupby(["subject_id", "hadm_id", "admittime"])["icd_code"]
    .apply(lambda x: set(x.dropna()))
    .reset_index(name="proc_codes_current_stay")
)


# --------------------------------------
# Diagnose- und Behandlungscodes zusammenführen
# --------------------------------------
hist_codes_stay = stay_history[["subject_id", "hadm_id", "admittime"]].copy()

hist_codes_stay = hist_codes_stay.merge(
    diag_stay,
    on=["subject_id", "hadm_id", "admittime"],
    how="left"
)

hist_codes_stay = hist_codes_stay.merge(
    proc_stay,
    on=["subject_id", "hadm_id", "admittime"],
    how="left"
)


# fehlende Sets durch leere Sets ersetzen
hist_codes_stay["diag_codes_current_stay"] = hist_codes_stay["diag_codes_current_stay"].apply(
    lambda x: x if isinstance(x, set) else set()
)

hist_codes_stay["proc_codes_current_stay"] = hist_codes_stay["proc_codes_current_stay"].apply(
    lambda x: x if isinstance(x, set) else set()
)


# ICD-10-Kapitel nur aus Diagnosecodes ableiten
hist_codes_stay["diag_chapters_current_stay"] = hist_codes_stay["diag_codes_current_stay"].apply(
    lambda codes: set(
        code_to_chapter[c]
        for c in codes
        if c in code_to_chapter and pd.notna(code_to_chapter[c])
    )
)

hist_codes_stay = hist_codes_stay.sort_values(
    ["subject_id", "admittime", "hadm_id"]
).copy()


# --------------------------------------
# Historie pro Patient aufbauen
# --------------------------------------
seen_diag_codes_per_subject = {}
seen_proc_codes_per_subject = {}
seen_diag_chapters_per_subject = {}

n_prev_unique_diag_codes = []
prev_diag_codes_list = []

n_prev_unique_proc_codes = []
prev_proc_codes_list = []

n_prev_unique_diag_chapters = []
prev_diag_chapters_list = []

for row in hist_codes_stay.itertuples(index=False):
    subj = row.subject_id

    current_diag_codes = row.diag_codes_current_stay
    current_proc_codes = row.proc_codes_current_stay
    current_diag_chapters = row.diag_chapters_current_stay

    if subj not in seen_diag_codes_per_subject:
        seen_diag_codes_per_subject[subj] = set()
        seen_proc_codes_per_subject[subj] = set()
        seen_diag_chapters_per_subject[subj] = set()

    # Historie VOR aktuellem Aufenthalt
    prev_diag_codes = seen_diag_codes_per_subject[subj].copy()
    prev_proc_codes = seen_proc_codes_per_subject[subj].copy()
    prev_diag_chapters = seen_diag_chapters_per_subject[subj].copy()

    n_prev_unique_diag_codes.append(len(prev_diag_codes))
    prev_diag_codes_list.append(sorted(prev_diag_codes))

    n_prev_unique_proc_codes.append(len(prev_proc_codes))
    prev_proc_codes_list.append(sorted(prev_proc_codes))

    n_prev_unique_diag_chapters.append(len(prev_diag_chapters))
    prev_diag_chapters_list.append(sorted(prev_diag_chapters))

    # erst danach aktuellen Aufenthalt zur Historie hinzufügen
    seen_diag_codes_per_subject[subj].update(current_diag_codes)
    seen_proc_codes_per_subject[subj].update(current_proc_codes)
    seen_diag_chapters_per_subject[subj].update(current_diag_chapters)


hist_codes_stay["n_prev_unique_diag_codes"] = n_prev_unique_diag_codes
hist_codes_stay["prev_diag_codes"] = prev_diag_codes_list

hist_codes_stay["n_prev_unique_proc_codes"] = n_prev_unique_proc_codes
hist_codes_stay["prev_proc_codes"] = prev_proc_codes_list

hist_codes_stay["n_prev_unique_diag_chapters"] = n_prev_unique_diag_chapters
hist_codes_stay["prev_icd10_chapters"] = prev_diag_chapters_list


# =========================
# Merge mit den anderen historischen Features
# =========================
stay_history = stay_history.merge(
    hist_codes_stay[[
        "hadm_id",
        "n_prev_unique_diag_codes",
        "prev_diag_codes",
        "n_prev_unique_proc_codes",
        "prev_proc_codes",
        "n_prev_unique_diag_chapters",
        "prev_icd10_chapters"
    ]],
    on="hadm_id",
    how="left"
)


# =========================
# Check
# =========================

# Korrektheit von ersten Aufenthalten prüfen
first_stays = stay_history.groupby("subject_id").head(1)

print(first_stays[[
    "subject_id",
    "n_prev_admissions",
    "n_prev_emergency_admissions",
    "max_past_los",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters",
    "prev_diag_codes",
    "prev_proc_codes",
    "prev_icd10_chapters"
]].head())

n_first_stays = len(first_stays)
print("Anzahl erster Aufenthalte:", n_first_stays)
print("Anzahl erster Aufenthalte, wo max_past_los = NaN:")
print(first_stays["max_past_los"].isna().value_counts())

hist_cols = [
    "n_prev_admissions",
    "n_prev_emergency_admissions",
    "time_since_last_discharge_days",
    "max_past_los",
    "avg_past_los",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters"
]

for col in hist_cols:
    n_nan = first_stays[col].isna().sum()
    print(f"{col}: {n_nan} / {n_first_stays}")


# Prüfen, ob voriger Aufenthalt vor Beginn des aktuellen Aufenthalts beendet ist
test = stay_history.copy()
test["check"] = (
    test["prev_dischtime"].isna() |
    (test["prev_dischtime"] <= test["admittime"])
)

print("Ist jeder vorige Aufenthalt vor Beginn des aktuellen Aufenthalts beendet?")
print(test["check"].value_counts())
# 10 Fälle mit Überschneidung sind ggf. administrative Inkonsistenzen


# Validieren, dass aktueller LOS nicht in max_past_los / avg_past_los enthalten ist
first_stays = stay_history.groupby("subject_id").head(1)

print("Erste Aufenthalte: max_past_los überall NaN?")
print(first_stays["max_past_los"].isna().all())

print("Erste Aufenthalte: avg_past_los überall NaN?")
print(first_stays["avg_past_los"].isna().all())


# zweiter Aufenthalt
second_stays = (
    stay_history
    .groupby("subject_id")
    .nth(1)
    .dropna(subset=["max_past_los"])
)

print("Zweite Aufenthalte:")
print(second_stays[[
    "los_days",
    "max_past_los",
    "avg_past_los",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters"
]].head())


# dritter Aufenthalt
third_stays = (
    stay_history
    .groupby("subject_id")
    .nth(2)
    .dropna(subset=["max_past_los"])
)

print("Dritte Aufenthalte:")
print(third_stays[[
    "los_days",
    "max_past_los",
    "avg_past_los",
    "n_prev_unique_diag_codes",
    "n_prev_unique_proc_codes",
    "n_prev_unique_diag_chapters"
]].head())


# Ausprägungen von einem konkreten Patienten mit sehr vielen Aufenthalten prüfen
pid = stay_history["subject_id"].value_counts().idxmax()

print(stay_history.loc[
    stay_history["subject_id"] == pid,
    [
        "admittime",
        "dischtime",
        "los_days",
        "n_prev_admissions",
        "n_prev_emergency_admissions",
        "time_since_last_discharge_days",
        "max_past_los",
        "avg_past_los",
        "n_prev_unique_diag_codes",
        "n_prev_unique_proc_codes",
        "n_prev_unique_diag_chapters",
        "prev_diag_codes",
        "prev_proc_codes",
        "prev_icd10_chapters"
    ]
])


# Validieren, dass erste Aufenthalte keine historischen Codes/Kapitel haben
print("Historische Diagnosecodes bei ersten Aufenthalten:")
print(stay_history.groupby("subject_id").first()["n_prev_unique_diag_codes"].head())

print("Historische Behandlungscodes bei ersten Aufenthalten:")
print(stay_history.groupby("subject_id").first()["n_prev_unique_proc_codes"].head())

print("Historische Diagnosekapitel bei ersten Aufenthalten:")
print(stay_history.groupby("subject_id").first()["n_prev_unique_diag_chapters"].head())


# Listenlängen müssen zu Count-Variablen passen
check_diag_codes = (
    stay_history["n_prev_unique_diag_codes"] ==
    stay_history["prev_diag_codes"].apply(len)
)

check_proc_codes = (
    stay_history["n_prev_unique_proc_codes"] ==
    stay_history["prev_proc_codes"].apply(len)
)

check_diag_chapters = (
    stay_history["n_prev_unique_diag_chapters"] ==
    stay_history["prev_icd10_chapters"].apply(len)
)

print("Diagnosecode-Counts konsistent:")
print(check_diag_codes.value_counts())

print("Behandlungscode-Counts konsistent:")
print(check_proc_codes.value_counts())

print("Diagnosekapitel-Counts konsistent:")
print(check_diag_chapters.value_counts())


# Anteil nicht-leerer historischer Listen
for col in ["prev_diag_codes", "prev_proc_codes", "prev_icd10_chapters"]:
    non_empty = stay_history[col].apply(len) > 0
    print(col)
    print("Anteil nicht-leer:", non_empty.mean())
    print("Anzahl nicht-leer:", non_empty.sum())
    print("Max Länge:", stay_history[col].apply(len).max())
    print()


# Prüfen, ob Listen wirklich Listen sind
for col in ["prev_diag_codes", "prev_proc_codes", "prev_icd10_chapters"]:
    print(col)
    print(stay_history[col].apply(type).value_counts())
    print()


# Harte Validierung
assert first_stays["n_prev_admissions"].eq(0).all()
assert first_stays["n_prev_emergency_admissions"].eq(0).all()
assert first_stays["max_past_los"].isna().all()
assert first_stays["avg_past_los"].isna().all()
assert first_stays["time_since_last_discharge_days"].isna().all()

assert first_stays["n_prev_unique_diag_codes"].eq(0).all()
assert first_stays["n_prev_unique_proc_codes"].eq(0).all()
assert first_stays["n_prev_unique_diag_chapters"].eq(0).all()

assert first_stays["prev_diag_codes"].apply(len).eq(0).all()
assert first_stays["prev_proc_codes"].apply(len).eq(0).all()
assert first_stays["prev_icd10_chapters"].apply(len).eq(0).all()

assert check_diag_codes.all()
assert check_proc_codes.all()
assert check_diag_chapters.all()


print(stay_history.columns)


# =========================
# Export
# =========================

stay_history.to_csv(path / "historical_features_staylevel.csv", index=False)

