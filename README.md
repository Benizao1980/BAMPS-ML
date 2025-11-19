# BAMPS-ML: AMR prediction, evaluation & mapping

End-to-end pipeline for predicting **S/I/R** classes and **MICs** from AMRFinder feature tables, plus post-hoc **confusion matrices**, **MIC panel plots**, and an **interactive geospatial map** (with optional country-level choropleth).  
Designed around tidy outputs (`sample`, `pred`) to keep downstream merges simple and robust.

---

## Table of contents

- [What’s implemented](#whats-implemented)
- [Installation](#installation)
- [Inputs](#inputs)
- [Typical workflow](#typical-workflow)
- [Data schemas](#data-schemas)
- [Repository layout](#repository-layout)
- [Files to commit vs ignore](#files-to-commit-vs-ignore)
- [Repro commands (Nov 19)](#repro-commands-nov-19)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## What’s implemented

- **Prediction**
  - **Classification (S/I/R)** → `preds_<antibiotic>_SIR.tsv`
  - **Regression (MIC)** → `preds_<antibiotic>_mic.tsv`
  - Safe feature alignment: **missing training features → 0**; **unseen prediction features → dropped**, with informative logs.

- **Batch runner:** `scripts/predict_all.py`
  - Runs all available models in a directory (classification and/or regression).
  - Writes tidy per-drug TSVs.
  - Optional extras: **confusion matrices** (SIR vs truth) and **MIC panel** plots.
  - Optional: kicks off map generation by producing standardised tidy outputs that join to geo metadata.

- **Evaluation & visualisation**
  - `scripts/plot_SIR_confusion.py` — confusion matrices from predicted S/I/R vs truth (truth may be SIR or MICs mapped via breakpoints).
  - `scripts/plot_predicted_mic_panel.py` — multi-drug MIC panel (optional log scale) with training accuracy badges.
  - `scripts/plot_isolates_map.py` — interactive Folium map:
    - points coloured by **S/I/R** or **MIC**,
    - optional **marker clustering** summarising **%R** and **n**,
    - optional **country choropleth**: `prevalence_R` / `median_mic` / `count`,
    - static PNG/SVG export when `geopandas` is available,
    - merge diagnostics printed to help align IDs.

> ✅ **Achieved so far:** integrated batch predictions; produced tidy SIR & MIC outputs for multiple drugs; added confusion-matrix & MIC-panel plotting CLIs; built interactive map CLI with clustering + choropleth; added robust ID handling & debug prints; documented full workflow.

---

## Installation

```bash
# (recommended) create an environment
conda create -n bamps_ml python=3.10 -y
conda activate bamps_ml

# install dependencies
pip install -r requirements.txt

# optional (for static map export & ISO conversion)
# pip install geopandas country_converter
