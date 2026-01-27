# BAMPS-ML

**BAMPS-ML** is a reproducible machine-learning workflow for predicting antimicrobial susceptibility phenotypes (MICs or S/I/R) from bacterial whole-genome sequence data. It is designed to compare **rule-based AMR determinant calls** (e.g. AMRFinderPlus) with **statistical/ML prediction**, and to support biological interpretation of resistance architecture.

Demonstration use cases: 
- Hybrid model building for Carbapenem resistance in *Acinetobacter baumanii* [documentation](https://github.com/Benizao1980/BAMPS-ML/blob/main/docs/workflow.md)
- Broad spectrum resistance in *Campylobacter jejuni* [TBC](tbc)

## What this repo does

* Builds feature matrices from one or more genomic “views”:

  * **AMR determinants** (AMRFinderPlus) ✅
  * Gene content / annotations (Bakta / pangenome) *(optional; in progress)*
  * Genome-wide variation (unitigs / GWAS matrices) *(optional; in progress)*
* Trains models for:

  * **Regression** (MIC prediction; log2 MIC internally)
  * **Classification** (S/I/R; via breakpoint mapping)
* Produces auditable run artefacts:

  * per-antibiotic models (`*.pkl`)
  * per-antibiotic plots (`*.png`)
  * `training_summary.tsv`, logs, environment snapshot

## Repository layout

* `scripts/` : CLI entrypoints (feature building, training, prediction, plotting)
* `bamps_ml/` : core library modules
* `data/` : example datasets (not distributed publicly unless stated)
* `outputs/` : generated outputs

  * `outputs/runs/<run_id>/` stores a complete reproducible run (models, plots, logs, versions)

## Quick start (AMRFinder → MIC panel training)

This “golden path” builds AMRFinder features and trains **one model per antibiotic column** found in `data/mic_values.norm.csv` (after dropping missing labels per antibiotic).

```bash
conda activate BAMPY
scripts/run_golden_path.sh
```

Outputs are written to `outputs/runs/.../` including:

* `models/<antibiotic>_regression.pkl`
* `plots/<antibiotic>_regression.png`
* `models/training_summary.tsv`
* `logs/*.log`, `versions.txt`, `run_meta.txt`

## Documentation

* End-to-end workflow: `docs/WORKFLOW.md` (data layout → feature extraction → training → evaluation → interpretation)

## Status

This repository contains code and documentation used for the *A. baumannii* AMR prediction study (Pascoe & Mourkas et al., in preparation). It is actively being cleaned and documented; optional feature views (gene content, unitigs/GWAS, mobile elements) are present but may still be under refinement.

## Known issues / platform notes

* **XGBoost on older glibc (e.g. Puma)**
  You may see a warning that your system has `glibc < 2.28` and that future XGBoost wheels may stop supporting it.

  * This is a *warning*, not an error, but it may become a breaking issue with future XGBoost releases.
  * Recommended workaround on older systems: use `--classifier lgbm` (LightGBM) or `--classifier ridge` for regression baselines.
  * If you need XGBoost specifically, pin to a compatible version in your environment (e.g. via conda), and record versions in `outputs/runs/<run>/versions.txt`.

* **Antibiotic “panel” behaviour**
  `scripts/train_model.py` trains **one model per antibiotic column** in the MIC/phenotype table (after dropping missing labels per antibiotic).
  This is why a single run produces multiple `*_regression.pkl` files plus `training_summary.tsv`.

* **Label sparsity drives performance**
  Some antibiotics may have substantially fewer usable labels (e.g. tobramycin), which will reduce stability and inflate uncertainty. Always report `n` per antibiotic (captured in `training_summary.tsv`).
