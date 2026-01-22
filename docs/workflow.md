# BAMPS-ML example workflow (used in Pascoe & Mourkas et al.)

Detailed workflow to use **BAMPS-ML** to predict antimicrobial susceptibility profiles and MIC values from genomic data, as desceribed in **Pascoe & Mourkas *et al.* (TBC)** 
This is a reproducible end-to-end workflow that prioritizes a single path run that can be extended to additional antibiotics and feature sets.

## Overview

Inputs:
- Assemblies: `data/contigs280/` (FASTA contigs; filenames must map to sample IDs)
- Phenotypes: `data/mic_values.norm.csv` (MIC table; includes imipenem/meropenem etc.)
- Config: `config/config_ACB_reg.yaml` (regression / MIC modelling)

Download contigs here: 

```bash
wget link to dataset contigs on figshare
```

Outputs:
- Trained models per antibiotic (and per feature-set)
- Benchmarking metrics (ROC/AUC, PR-AUC, calibration, confusion matrices)
- Interpretation outputs (top features, SHAP summaries)
- Reproducible run folder with configs and logs

## Environment
Option A: conda (recommended)

```bash
conda env create bamps-ml
conda activate bamps-ml
```

Option B: pip

```bash
pip install -r requirements.txt
```

## Data layout
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
- 'sample_id'
- one column per antibiotic (either MIC numeric or categorical label)

---
# Normalise phenotype calling (optional)
---

to ensure the MIC file contains S/I/R labels or that the config defines breakpointing; you have breakpoints.py and CLSI configs

---
# BUILD A MODEL
---

## Feature extraction

### Step 1: Generate AMR determinant features (AMRFinderPlus)

Run AMRFinderPlus externally or via the wrapper script, then generate a per-sample feature matrix.

```bash
python scripts/run_amrfinder.py \
  --genome-dir data/contigs280 \
  --output-dir outputs/amrfinder/Russia280 \
  --threads 8
```

**OUTPUTS:** This produces individual files per genome and a combined AMR feature table in the output directory 
- outputs/amrfinder/Russia280/amr_presence_absence.tsv
- outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv (is this produced by default?)
- outputs/amrfinder/Russia280/raw/*.amrfinder.tsv

### Step 2A: Train an MIC prediction model (regression; example: imipenem)

The AMRFinder feature table uses sample as the ID column.
The MIC table contains both sample and ID; BAMPS-ML automatically prefers sample for joining.

```bash
python scripts/train_model.py \
  --feature-table outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv \
  --mic-file data/mic_values.norm.csv \
  --task regression \
  --classifier xgb \
  --model-dir outputs/ml_models/imipenem_amrfinder_mic \
  --plot-dir outputs/plots/imipenem_amrfinder_mic \
  --random-state 1 \
  --n-jobs 8 \
  --self-test
```

### Step 2B: Train an S/R prediction model (classification; example: imipenem)

```bash
python scripts/train_model.py \
  --feature-table outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv \
  --mic-file data/mic_values.norm.csv \
  --task classification \
  --classifier xgb \
  --model-dir outputs/ml_models/imipenem_amrfinder_SIR \
  --plot-dir outputs/plots/imipenem_amrfinder_SIR \
  --random-state 1 \
  --n-jobs 8 \
  --self-test
```

*Notes:*
- If XGBoost becomes problematic on older glibc systems, use '--classifier lgbm' (preferred) or '--classifier ridge' (regression baseline).
- '--self-test' holds back 25% of the training split as an inner validation set.
- Use '--test-size' to control the outer hold-out split fraction.
- Use '--tune' for randomized hyperparameter search, and '--bootstrap-reps' to quantify uncertainty.

*coming soon*
- model selection comparisons
- hyperparameter tuning
- refinement for under / oversampling
- bootstraping and propagation of uncertainty
- fold validation

## Step 3 (optional): Add hybrid features (e.g., PYSEER GWAS output)

### Merge PYSEER GWAS outputs 
Helper script to merge GWAS outputs from gene presence / absence, SNP and unitigs (kmer) analyses: 'Bens_script' (add link)

```bash
how to run
```

### Pass GWAS features directly:

```bash
python scripts/train_model.py \
  --feature-table outputs/amrfinder/contigs280/amr_features.tsv \
  --mic-file data/mic_values.norm.csv \
  --task regression \
  --classifier xgb \
  --gwas-table outputs/gwas/unitigs_matrix.tsv \
  --gwas-top-k 5000 \
  --amr-prefix AMR_ \
  --gwas-prefix GWAS_ \
  --model-dir outputs/ml_models/imipenem_hybrid_mic \
  --plot-dir outputs/plots_hybrid/imipenem_hybrid_mic \
  --random-state 1 \
  --n-jobs 8 \
  --self-test
```

### Rerun for multiple antibioitcs and feature groups

Add `scripts/run_golden_path.sh`
A single shell script that reproduces the golden path with variables at the top (so you don’t retype commands).

---
FEATURE DISCOVERY
---

GWAS x3
Combine
BLAST in genomes
Buidling a hybrid model

---
# VALIDATE MODEL
---

A validation dataset of sequenced genomes, with correspondign MIC data was downloaded from the [Acinetobacter baumannii IOI Collection v1 database](https://bioinf.ineosoxford.ox.ac.uk/bigsdb?db=ioi_abaumannii_isolates) (n=672; login required)

[Validation dataset dashboard](figures/validation_dataset.png)

**Figure X:** Dashbaord overview of validation dataset.

## Input
- validation contigs: `data/contigs_validation_dataset/`
- validation phenotypes (truth MICs): `data/phenotypes_validation/validation_dataset_MIC.csv`
  - must include an ID column matching contig IDs (or whatever ID BAMPS uses)
- trained models: `outputs/runs/001_.../models/*_regression.pkl`
- optional model training summary: `outputs/runs/001_.../models/training_summary.tsv`

You can also download the contigs from here:

```bash
wget -c https://figshare.com/ndownloader/files/61226248?private_link=5e5b1065edad79c38c3a
```

### Step A — Build AMRFinder features for validation contigs

```bash
python scripts/run_amrfinder.py \
  --genome-dir data/contigs_validation_dataset \
  --output-dir outputs/amrfinder/validation \
  --threads 8
```

This produces (example):
- `outputs/amrfinder/validation/amr_presence_absence.tsv`

### Step B — Predict MICs on validation isolates using trained models

**Batch prediction + plots + metrics (recommended)**

This will:
- generate one tidy TSV per antibiotic (preds/preds_<antibiotic>_mic.tsv)
- generate a MIC panel plot (preds/validation_MIC_panel.png + .svg)
- compute paper-grade metrics per antibiotic (preds/validation_metrics.tsv)
- optionally generate lineage-stratified panels if `--lineage-col` is provided

```bash
python scripts/predict_all.py \
  --feature-table outputs/amrfinder/validation/amr_presence_absence.tsv \
  --model-dir outputs/runs/001_Russia280_AMRFinder_MIC_panel_xgb/models \
  --outdir preds \
  --tasks regression \
  --to-mic \
  --panel-out preds/validation_MIC_panel \
  --panel-truth data/phenotypes_validation/validation_dataset_MIC.csv \
  --metrics-out preds/validation_metrics.tsv \
  --lineage-col lineage \
  --top-lineages 6
```

If your feature table does not have an explicit sample/id column, pass `--id-col`:

```bash
python scripts/predict_all.py ... --id-col <column_name>
```

**Single antibiotic (regression; MIC)**

```bash
python scripts/predict.py \
  --feature-table outputs/amrfinder/validation/amr_presence_absence.tsv \
  --model-dir outputs/runs/001_Russia280_AMRFinder_MIC_panel_xgb/models \
  --antibiotic imipenem \
  --task regression \
  --to-mic \
  --output preds/preds_imipenem_mic.tsv
```

**Single antibiotic (classification; S/I/R)**

```bash
python scripts/predict.py \
  --feature-table outputs/amrfinder/validation/amr_presence_absence.tsv \
  --model-dir outputs/runs/001_Russia280_AMRFinder_MIC_panel_xgb/models \
  --antibiotic imipenem \
  --task classification \
  --output preds/preds_imipenem_SIR.tsv
```

### Step C — Score predictions against known validation MICs
The pipeline writes preds/validation_metrics.tsv with per-antibiotic validation performance.

Minimum “paper-grade” validation metrics per antibiotic:
-R² on log2(MIC) (regression goodness-of-fit)
- MAE in log2 units (average error in dilution steps)
- Within ±1 dilution accuracy (clinically intuitive)

These metrics are also auto-embedded into the MIC panel plot when available.

## (optional) Compare hybrid models

---
# MAKE PREDICTIONS
---

Prediction dataset: pubMLST (add link)

Input
- contigs
- validated model.pkl
- metadata.csv

Download contigs here: 

```bash
wget link to dataset contigs on figshare
```

## Use constructed model to make predictions 

Using prediction script

```bash
how to run it
```

## Outputs:
-
-
-

Breakdown outputs by metadata (country, host, etc.)
Self test?
Metrics

## (optional) Compare hybrid models

---

## Reproducibility checklist (minimum)
For each run, record:
- exact command used
- git commit hash (if available)
- config YAML used
- random seed
- software versions (conda list > versions.txt)

## Software and links
AMRfinderPlus
PROKKA / BAKTA
PIRATE
PIRATE post-processing scripts
PYSEER
PYSEER post-processing scripts

FigShare: dataset [contigs]
Figshare: validation dataset [contigs]
Microreact: dataset
Microreact: validation dataset
Microreact: Global prediction dataset
IOI Collections: [*A. baumanii*](https://bioinf.ineosoxford.ox.ac.uk/bigsdb?db=ioi_abaumannii_isolates)
PubMLST database (*A. baumanii*)

---
## Please cite: 
Pascoe & Mourkas TBC
