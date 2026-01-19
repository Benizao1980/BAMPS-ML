# BAMPS-ML

**BAMPS-ML** is a reproducible machine-learning workflow for predicting antimicrobial susceptibility phenotypes (MICs or S/I/R) from bacterial whole-genome sequence data. It is designed to compare **rule-based AMR determinant calls** (e.g. AMRFinderPlus) with **statistical/ML prediction**, and to support biological interpretation of resistance architecture.

Demonstration use case: *Acinetobacter baumannii*.

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
