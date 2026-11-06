# Vorhersage der Krankenhausmortalität anhand hierarchischer ICD-10-Merkmale

**Masterarbeit: Integration taxonomiebasierter semantischer Merkmale von medizinischen Klassifikationscodes zur Verbesserung der Mortalitätsprognose in Machine-Learning-Modellen**

Machine Learning · Klinische Datenanalyse · Feature Engineering · Explainable AI

## Projektübersicht

Medizinische Klassifikationscodes wie ICD-10 werden in ML-Modellen häufig als hochdimensionale binäre Merkmale (Multi-Hot-Encoding) repräsentiert. Dabei bleiben Informationen über die hierarchische Struktur und die Beziehungen zwischen einzelnen Codes weitgehend unberücksichtigt.

Diese Masterarbeit untersucht, inwieweit explizit aus der ICD-10-Taxonomie abgeleitete Merkmale zur Vorhersage der Krankenhausmortalität beitragen können. Ergänzend werden verteilungsbasierte Code-Merkmale berücksichtigt.

Ziel ist es, den prognostischen Nutzen dieser Merkmale im Vergleich zu konventionellen ICD-10-Repräsentationen zu bewerten und zu untersuchen, ob sie eine kompaktere und besser nachvollziehbare Abbildung medizinischer Codeinformationen ermöglichen.

Die Analysen basieren auf dem klinischen Datensatz MIMIC-IV und umfassen die Entwicklung verschiedener Merkmalsrepräsentationen, das Training und die Evaluation von ML-Modellen sowie deren Interpretation mittels SHAP.

## Verwendete Technologien und Methoden

- **Programmiersprache:** Python
- **Datenverarbeitung und Analyse:** pandas, NumPy, SciPy
- **Machine Learning:** XGBoost, scikit-learn
- **Visualisierung:** Matplotlib, Seaborn
- **Modellauswahl:** Logistische Regression mit L2- bzw. Elastic-Net-Regularisierung, Random Forest und XGBoost (beste Performance: XGBoost)
- **Modellbewertung:** Average Precision (AP), AUROC, Cross-Validation, patientenbasiertes Bootstrapping
- **Modellinterpretation:** SHAP
- **Feature Engineering:** Hierarchische ICD-10-Merkmale, Code-Prävalenzen bzw. Information Content, aggregierte Diagnose- und Behandlungscodemerkmale

## Methodisches Vorgehen

### 1. Datenaufbereitung

Aufbereitung und Verknüpfung der für die Analyse relevanten Informationen aus dem MIMIC-IV-Datensatz, insbesondere zu Krankenhausaufenthalten, Diagnosen und medizinischen Prozeduren.

Die Untersuchungseinheit bildet der Krankenhausaufenthalt. Als Zielvariable dient die Krankenhausmortalität. Die finale Analysestichprobe umfasst 254.396 Krankenhausaufenthalte von 122.932 eindeutigen Patient:innen.

### 2. Feature Engineering

Entwicklung zusätzlicher Merkmale zur Beschreibung der semantischen und statistischen Eigenschaften medizinischer Klassifikationscodes.

Hierzu gehören insbesondere:

- **Hierarchische Merkmale:** Hierarchietiefe, Blattknotenindikator, Anzahl direkter Geschwisterknoten und logarithmierte relative Teilbaumgröße
- **Gruppenbasierte Merkmale:** Zugehörigkeit zu Diagnosekapiteln (ICD-10-Codes werden auf einer übergeordneten, taxonomisch zusammengehörigen Ebene zusammengefasst)
- **Verteilungsbasierte Merkmale:** Information Content (Seltenheit eines Codes)

### 3. Modelltraining und Evaluation

Training und Vergleich von XGBoost-Modellen mit unterschiedlichen Merkmalsrepräsentationen:

- **Multi-Hot-Baseline:** Direkte binäre Repräsentation der ICD-10-Codes
- **Erweiterte Modelle:** Kombination der Multi-Hot-Repräsentation mit taxonomischen und verteilungsbasierten Code-Merkmalen
- **Dimensionsreduziertes Modell:** Ausschließliche Verwendung der entwickelten semantischen Code-Merkmale
- **CCSR-Benchmark:** Vergleich mit einer kategorienbasierten Repräsentation medizinischer Codes

In einem zusätzlichen experimentellen Ansatz werden ausschließlich Codeinformationen früherer Krankenhausaufenthalte berücksichtigt, um den Einfluss der zeitlichen Informationsverfügbarkeit und potenzieller Datenlecks zu untersuchen (Data bzw. Label Leakage).

### 4. Modellinterpretation

Mithilfe von SHAP werden die Beiträge einzelner Merkmale zu den Modellvorhersagen analysiert. Dabei wird insbesondere untersucht, welche der entwickelten Merkmalskategorien tatsächlich für die Vorhersagen genutzt werden.

## Ausgewählte Ergebnisse

Die wichtigsten Ergebnisse der Hauptuntersuchung und des ergänzenden CCSR-Vergleichs sind in der folgenden Tabelle zusammengefasst:

| Merkmalsrepräsentation | Average Precision (AP) |
|---|---:|
| ICD-10 Multi-Hot-Baseline | 0,810 |
| Multi-Hot + alle semantischen Merkmale | 0,818 |
| Ausschließlich semantische Merkmale | 0,538 |
| CCSR-basierte Repräsentation | 0,773 |

### Zentrale Erkenntnisse

- Die Erweiterung der Multi-Hot-Repräsentation um semantische Merkmale führt zu einer geringfügigen Verbesserung der Vorhersageleistung, die jedoch nicht statistisch signifikant ist.
- Das dimensionsreduzierte Modell mit ausschließlich semantischen Merkmalen bewahrt einen wesentlichen Teil der prognostischen Information, bleibt aber deutlich hinter der direkten ICD-10-Repräsentation zurück.
- Bei ausschließlicher Verwendung von Codeinformationen früherer Aufenthalte sinkt die Prognoseleistung aller Modelle erheblich.
- Die CCSR-basierte Repräsentation übertrifft das dimensionsreduzierte Semantik-Modell, erreicht jedoch nicht die Leistung der Multi-Hot-Baseline.
- SHAP-Analysen zeigen, dass insbesondere verteilungsbasierte Merkmale und hierarchische Diagnosecodemetriken relevante Beiträge zu den Modellentscheidungen leisten.

**Fazit:** Explizite taxonomiebasierte ICD-10-Merkmale verbessern die Prognoseleistung gegenüber der direkten Code-Repräsentation nicht nachweisbar. Sie ermöglichen jedoch eine kompaktere und besser nachvollziehbare Abbildung medizinischer Codeinformationen. Die Bewertung solcher Repräsentationen sollte daher neben der Prognoseleistung auch Dimensionalität, Interpretierbarkeit und tatsächliche Merkmalsnutzung berücksichtigen.

## Projektstruktur

```text
icd10-mortality-prediction/
├── README.md
├── requirements.txt
└── source code/
    └── Python-Skripte zur Datenaufbereitung,
        Merkmalsgenerierung, Modellierung
        und Ergebnisauswertung
```

Der Ordner `source code` enthält die Python-Skripte zur Umsetzung der einzelnen Analyseschritte. Die Skripte sind entsprechend ihrer auszuführenden Reihenfolge nummeriert.

## Installation und Voraussetzungen

Die für die Ausführung benötigten Python-Bibliotheken sind in der Datei `requirements.txt` dokumentiert und können mit folgendem Befehl installiert werden:

```bash
pip install -r requirements.txt
```

Für die vollständige Reproduktion der Analysen ist zusätzlich ein autorisierter Zugang zum MIMIC-IV-Datensatz erforderlich. Die Dateipfade müssen entsprechend der lokalen Arbeitsumgebung eingerichtet werden.

## Datengrundlage und Verfügbarkeit

Die Arbeit verwendet den öffentlich dokumentierten, jedoch zugangsbeschränkten klinischen Datensatz [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/) über PhysioNet.

Der Zugriff auf die Patient:innendaten setzt eine entsprechende Zugangsberechtigung voraus. **Die zugrundeliegenden Patient:innendaten sind daher nicht Bestandteil dieses GitHub-Repositories.**

## Hintergrund

Dieses Projekt entstand im Rahmen einer Masterarbeit an der Freien Universität Berlin zur Untersuchung des Beitrags taxonomiebasierter ICD-10-Merkmale zur Vorhersage der Krankenhausmortalität.

Das Repository dokumentiert die praktische Umsetzung eines Machine-Learning-Workflows von der Aufbereitung klinischer Daten über die Entwicklung und Evaluation unterschiedlicher Merkmalsrepräsentationen bis hin zur Interpretation der Modellentscheidungen.
