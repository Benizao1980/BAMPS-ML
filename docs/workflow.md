# Hybrid model building for Carbapenem resistance in *A. baumanii*

Detailed workflow to use **BAMPS-ML** to predict antimicrobial susceptibility profiles and MIC values from genomic data, as desceribed in **Pascoe & Mourkas *et al.* (TBC)** This is a reproducible end-to-end workflow that prioritises a single path run that can be extended to additional antibiotics and feature sets.

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
- ``sample_id``
- one column per antibiotic (either MIC numeric or categorical label)

---
# DATASET
---

Download contigs here: 

```bash
cd ~/BAMPY_ML/data

wget -c link to dataset contigs on figshare
tar -zxfv contigs.russi280.tar.gz
```


## Normalise phenotype calling (optional)

If MIC tables contain raw values only, BAMPS-ML can derive S/I/R labels using ``breakpoints.py`` and a chosen standard (e.g. CLSI / EUCAST), or per-antibiotic overrides defined in the config.

```bash
insert
```

---
# BUILD A MODEL
---

Inputs:
- Assemblies: ``data/contigs280/`` (FASTA contigs; filenames must map to sample IDs)
- Phenotypes: ``data/mic_values.norm.csv`` (MIC table; includes imipenem/meropenem etc.)
- Config: ``config/config_ACB_reg.yaml`` (regression / MIC modelling)

Outputs:
- Trained models per antibiotic (and per feature-set)
- Benchmarking metrics (ROC/AUC, PR-AUC, calibration, confusion matrices)
- Interpretation outputs (top features, SHAP summaries)
- Reproducible run folder with configs and logs

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
- ``outputs/amrfinder/Russia280/amr_presence_absence.tsv``
- ``outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv``
- ``outputs/amrfinder/Russia280/raw/*.amrfinder.tsv``

The combined feature table uses **sample ID** as the primary key.

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

*Notes*
- ``--self-test`` performs an internal 75/25 split for quick diagnostics.
- Use ``--tune`` for hyperparameter optimisation.
- Use ``--bootstrap-reps`` to quantify uncertainty.
- On older systems, prefer ``--classifier lgbm`` or ``ridge`` over XGBoost.

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

*coming soon*
- refinement for under / oversampling
- bootstraping and propagation of uncertainty
- fold validation

### Rerun for multiple antibioitcs

The script ``scripts/run_golden_path.sh`` reproduces the model building path with variables at the top (i.e. run for additional antibiotics). Including the flag ``--all`` buidls models or all antibioitcs included as columns in your input phenotype file.

```bash
insert
```

### Self test evaluation

``Add detail``
+ confusion matrices
  
*Interpretation: The baseline AMRFinder-only model shows moderate performance but systematic under-prediction of high MICs. This behaviour motivates hyperparameter tuning, and retraining.*

---
# MODEL TUNING AND RETRAINING
---

We tune hyperparameters on the *training dataset only* (Russia280). 
The model can be tuned using different ML classifiers and a selection (xgBoost, LGMboost, ridge, linear regression) of common hyperparameter values using ``tune_model.py``

```bash
python scripts/tune_model.py \
  --feature-table outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv \
  --mic-file data/mic_values.norm.csv \
  --task regression \
  --classifier xgb \
  --antibiotics imipenem meropenem \
  --log2 \
  --n-iter 80 \
  --cv 5 \
  --n-jobs 8 \
  --outdir outputs/tuning/imi_mer_xgb_log2
```

This produces:
- ``tuning/summary.tsv``
- ``tuning/cv_results_<antibiotic>.tsv``
- ``models/<antibiotic>__<task>__<classifier>.pkl``
- ``models/<antibiotic>__...__metadata.yaml``

And we can retrain our model using --tune to select a tuned model, or --best to choose the best performing model from summary.tsv

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

---
# VALIDATE MODEL
---

A validation dataset of sequenced genomes, with correspondign MIC data was downloaded from the [Acinetobacter baumannii IOI Collection v1 database](https://bioinf.ineosoxford.ox.ac.uk/bigsdb?db=ioi_abaumannii_isolates) (n=672; login required)

![Validation dataset dashboard](../figures/validation_dataset.png)

**Figure 1: Dashbaord overview of validation dataset.** The validation dataset includes 672 *A. baumannii* genomes with matched MIC data, primarily from North America, spanning 2004–2023. Assemblies show consistent genome sizes (~3.9 Mb) and encompass diverse Pasteur MLST lineages. Meropenem susceptibility profiles reveal a high prevalence of resistance alongside susceptible isolates, providing a robust and clinically relevant benchmark for evaluating MIC prediction performance, particularly for carbapenems.

This validation step quantifies:
- Absolute MIC prediction accuracy
- Improvement from ML vs rule-based AMR
- Added value of GWAS-derived determinants
- Lineage-specific performance gains

## Download data an clean input
Download contig files and correspondign phenotype data from the *A. baumanii* IOI collection:

![Export contig files for download](link)

You can also download the contigs from here:

```bash
cd /<your_location>/BAMPS_ML/data

wget -c https://figshare.com/ndownloader/files/61226248?private_link=5e5b1065edad79c38c3a
tar -zxfv contigs_validation.tar.gz
```

![Export MIC data for download](link)

Check and clean data formats:

```
insert
```

## Inputs
- Validation contigs: ``data/contigs_validation_dataset/``
- Validation phenotypes (MICs): ``data/phenotypes_validation/validation_dataset_MIC.csv``
  - Supported formats:
    - **wide**: one row per sample, MIC columns per antibiotic (e.g., ``imipenem``, ``meropenem``, ...)
    - **long**: columns ``sample``, ``antibiotic``, ``mic``
- Trained models: ``outputs/runs/<RUN_NAME>/models/*_regression.pkl``

The plotting/metrics script auto-detects wide vs long and standardises internally.

### Step A — Build AMRFinder features for validation contigs

```bash
# Build validation AMRFinder features
python scripts/run_amrfinder.py \
  --genome-dir data/contigs_validation_dataset \
  --output-dir outputs/amrfinder/validation \
  --threads 8
```

This produces (example):
- ``outputs/amrfinder/validation/amr_presence_absence.tsv``

### Step B — Predict MICs on validation isolates using trained models

**Batch prediction + plots + metrics**

This writes one tidy TSV per antibiotic (prefixed ``preds_...``) and generates MIC panel plots.

```bash
# Predict on validation set (tuned models)
python scripts/predict_all.py \
  --feature-table outputs/amrfinder/validation/amr_presence_absence.tsv \
  --model-dir outputs/runs/002_Russia280_AMRFinder_IMI_MER_xgb_tuned/models \
  --outdir preds/validation_tuned \
  --tasks regression \
  --panel-out preds/validation_tuned/MIC_panel \
  --panel-order imipenem meropenem \
  --panel-truth data/phenotypes_validation/validation_dataset_MIC.csv \
  --panel-truth-id-col id
```

Outputs: 
- ``preds/preds_<antibiotic>_mic.tsv`` (one per antibiotic)
- ``preds/validation_MIC_panel.png`` + ``.svg``
- ``preds/validation_metrics.tsv`` (per-antibiotic validation metrics)
- (Optional) lineage-stratified panels: ``preds/validation_MIC_panel.lineage_<X>.png`` + ``.svg`` (top ``N`` lineages)

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

### Step C — Evaluate baseline performance
# Evaluate vs truth
```bash
python scripts/evaluate_mic_predictions.py \
  --pred preds/validation_tuned/preds_*_mic.tsv \
  --truth data/phenotypes_validation/validation_dataset_MIC.csv \
  --outdir preds/validation_tuned/eval \
  --truth-id-col id \
  --make-plots
  ```

The pipeline writes ``preds/validation_metrics.tsv`` with per-antibiotic validation performance.

Minimum “paper-grade” validation metrics per antibiotic:
- R² on log2(MIC) (regression goodness-of-fit)
- MAE in log2 units (average error in dilution steps)
- Within ±1 dilution accuracy (clinically intuitive)

Alternatively, run all tuning and retrainign stesp by using: 
```bash
tbc
```

![Model evaluation](../figures/model_evaluation.png)

**Figure X:** Model evaluation.

*Interpretation:* On the external validation set, tuned AMRFinder-only models show clear separation between low and high MICs for both carbapenems, with improved fit compared to baseline. Residual plots indicate remaining systematic error at the highest MICs (notably for imipenem), suggesting additional genetic determinants beyond curated AMR calls—motivating hybrid AMR+GWAS feature models.

---
FEATURE DISCOVERY (optional)
---

## GWAS feature construction
To capture genetic determinants not represented in curated AMR databases, we incorporate GWAS-derived unitigs and collapse them into biologically interpretable locus-level features.

### Step 1 — Combine GWAS hits

Merge GWAS outputs (e.g. imipenem, meropenem):

```bash
python scripts/build_unitig_to_locus.py \
  --inputs IMI_hits.csv MER_hits.csv \
  --labels IMI MER \
  --out-prefix gwas_combined
```

Outputs:

```
gwas_combined_unitig_to_locus.tsv
gwas_combined_locus_summary.tsv
```

### Step 2 — Build GWAS unitig FASTA

```bash
python scripts/build_gwas_fasta.py \
  --inputs IMI_hits.csv MER_hits.csv \
  --out combined_gwas_unitigs_unique.fasta \
  --deduplicate
```

### Step 3 — Screen genomes for GWAS unitigs

```bash
python scripts/screen_unitigs_in_genomes.py \
  --unitigs combined_gwas_unitigs_unique.fasta \
  --genome-dir data/contigs280 \
  --out-prefix outputs/gwas/gwas_features_russia280 \
  --threads 8
```

Output:
```
*_presence_absence.tsv (unitig-level matrix)
```

### Step 4 — Convert unitigs to locus-level features

```bash
python scripts/build_matrix_compatible_mapping.py \
  --fasta combined_gwas_unitigs_unique.fasta \
  --mapping gwas_combined_unitig_to_locus.tsv \
  --out gwas_matrix_compatible_mapping.tsv
```

```bash
python scripts/collapse_unitig_matrix_to_loci.py \
  --matrix outputs/gwas/gwas_features_russia280_presence_absence.tsv \
  --mapping gwas_matrix_compatible_mapping.tsv \
  --out-prefix outputs/gwas/gwas_features_russia280
```

Output:

```
*_locus_presence_absence.tsv
```

This reduces high-dimensional unitig space into:
- interpretable genomic loci
- robust, non-redundant features

### Step 5 — Build hybrid feature matrix

Combine AMR + GWAS features:

```bash
python scripts/merge_feature_tables.py \
  --amr outputs/amrfinder/Russia280/amr_presence_absence.norm.tsv \
  --gwas outputs/gwas/gwas_features_russia280_locus_presence_absence.tsv \
  --out outputs/features/hybrid_russia280.tsv
```

---
# EVALUATE HYBRID MODELS
---

```bash
python scripts/train_model.py \
  --feature-table outputs/features/hybrid_validation.tsv \
  --mic-file data/phenotypes_validation/validation_dataset_MIC.csv \
  --task regression \
  --classifier xgb \
  --model-dir outputs/ml_models/imipenem_hybrid \
  --plot-dir outputs/plots/imipenem_hybrid \
  --self-test
```


![Hybrid Model evaluation](../figures/hybrid_model_evaluation.png)

**Figure 4:** GWAS-informed hybrid models improve carbapenem MIC prediction

(A) Baseline AMRFinder-only model performance shows high specificity but reduced sensitivity for elevated MICs.
(B) GWAS-derived loci capture additional genomic variation associated with resistance.
(C) Hybrid models integrating AMR determinants and GWAS loci significantly improve prediction accuracy, particularly for high MIC isolates.
(D) Error distribution (log2 MIC) demonstrates reduced systematic underprediction in hybrid models.

*Interpretation:*
While curated AMR determinants capture canonical resistance mechanisms, GWAS-derived loci reveal additional genetic signals—likely reflecting regulatory variation, compensatory mutations, and lineage-specific adaptations. Integrating these signals improves predictive performance and highlights the polygenic architecture of carbapenem resistance in *A. baumannii*.

---
# MAKE GLOBAL PREDICTIONS
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
