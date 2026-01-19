# BAMPS-ML example workflow (used in Pascoe & Mourkas et al.)

Detailed workflow to use **BAMPS-ML** to predict antimicrobial susceptibility profiles and MIC values from genomic data, as desceribed in **Pascoe & Mourkas *et al.* (TBC)** 

## Overview

Inputs:
- Assemblies (FASTA) or genomes with consistent sample IDs
- Phenotype table (MICs or S/I/R) per antibiotic

Outputs:
- Trained models per antibiotic (and per feature-set)
- Benchmarking metrics (ROC/AUC, PR-AUC, calibration, confusion matrices)
- Interpretation outputs (top features, SHAP summaries)
- Reproducible run folder with configs and logs

## 1) Environment
Option A: conda (recommended)
```bash
conda env create -f environment.yml
conda activate bamps-ml
```

Option B: pip

```bash
pip install -r requirements.txt
```

## 2) Data layout
Expected paths (example):

```bash
data/
  genomes/
    SAMPLE_001.fasta
    SAMPLE_002.fasta
  phenotypes/
    phenotypes.tsv
```

phenotypes.tsv minimum columns:
- sample_id
- one column per antibiotic (either MIC numeric or categorical label)

See **docs/DATA_LAYOUT.md** for details.

## 3) Feature extraction

### 3.1 AMR determinants (AMRFinderPlus)
Run AMRFinderPlus externally or via wrapper, then generate a per-sample feature matrix.

Example:

```bash
python -m bamps_ml.cli features amrfinder \
  --genomes data/genomes \
  --out results/features/amrfinder.tsv
```

### 3.2 Annotations / gene content (Bakta)
```bash
python -m bamps_ml.cli features annotations \
  --bakta_dir results/bakta \
  --out results/features/annotations.tsv
```

### 3.3 Unitigs (optional, GWAS-style)
```bash
python -m bamps_ml.cli features unitigs \
  --assemblies data/genomes \
  --out results/features/unitigs.tsv
```

### 3.4 MOB-suite context (optional)
```bash
python -m bamps_ml.cli features mob \
  --mobsuite_dir results/mob \
  --out results/features/mob.tsv
```

## 4) Train models
Train one model per antibiotic per feature set.

Example (AMRFinder features):

```bash
python -m bamps_ml.cli train \
  --features results/features/amrfinder.tsv \
  --phenotypes data/phenotypes/phenotypes.tsv \
  --antibiotic imipenem \
  --model xgboost \
  --split lineage_aware \
  --out results/models/imipenem_amrfinder
```

Repeat for additional antibiotics and feature sets.

## 5) Evaluate
```bash
python -m bamps_ml.cli evaluate \
  --model_dir results/models/imipenem_amrfinder \
  --out results/eval/imipenem_amrfinder
```

Standard outputs:
- ROC curve + AUC
- Precision–recall curve + PR-AUC
- Confusion matrix at chosen threshold
- Calibration plot

## 6) Interpretability
```bash
python -m bamps_ml.cli interpret \
  --model_dir results/models/imipenem_amrfinder \
  --method shap \
  --out results/interpret/imipenem_amrfinder
```

Outputs:
- Global feature ranking
- SHAP summary plots (global + per-class if relevant)
- Per-sample explanations for selected isolates

## 7) Reproducibility checklist

Each run folder should contain:
- config.yaml
- versions.txt (tool + package versions)
- splits.tsv (train/test assignments)
- logs and seed values
