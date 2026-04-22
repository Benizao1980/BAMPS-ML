<p align="center">
  <img src="bampsml_logo_variantB_ladder.svg" width="360" alt="BAMPS-ML logo">
</p>

# BAMPS-ML

*Interpretable prediction of antimicrobial resistance phenotypes from bacterial genomes.*

BAMPS-ML is a framework for predicting antimicrobial resistance phenotypes (MICs or S/I/R)
from whole-genome sequence data, with an explicit focus on **biological interpretation**.

The starting assumption is that resistance phenotypes rarely arise from single genes in
isolation. In practice, they reflect interactions between **known resistance determinants,
genetic background, and population structure**. Models that ignore this context may perform
well under cross-validation while remaining difficult to interpret or generalise.

BAMPS-ML is designed to keep those constraints visible.

---

## What BAMPS-ML actually does

At a practical level, BAMPS-ML builds and evaluates resistance prediction models using one
or more complementary genomic “views”.

Feature matrices can be derived from:

- **AMR determinants** (via AMRFinderPlus)
- Gene content and functional annotation (Bakta / pangenome) *(optional; in progress)*
- Genome-wide sequence variation (unitigs / GWAS matrices) *(optional; in progress)*

These feature sets can be used independently or combined, depending on the question being
asked.

For each antibiotic, BAMPS-ML supports:

- **Regression** models  
  (MIC prediction; MICs are internally transformed to log₂ scale)

- **Classification** models  
  (S/I/R prediction via breakpoint mapping)

Models are trained **per antibiotic**, reflecting the fact that resistance architecture,
label availability, and uncertainty often differ substantially across drugs.

---

## Reproducible outputs

Each run produces a complete, auditable record, including:

- per-antibiotic trained models (`*.pkl`)
- per-antibiotic evaluation plots (`*.png`)
- `training_summary.tsv` summarising performance and sample sizes
- logs and software/environment snapshots

Outputs are written to a self-contained run directory:

outputs/runs/<run_id>/

This directory contains everything required to reproduce or audit a given analysis.

---

## Repository layout

- `scripts/`  
  Command-line entry points for feature construction, training, prediction, and plotting

- `bamps_ml/`  
  Core library modules

- `data/`  
  Example datasets (not distributed publicly unless explicitly stated)

- `outputs/`  
  Generated outputs, organised by run

---

## Quick start

### AMRFinder → MIC panel training

This “golden path” builds AMRFinder-based features and trains **one model per antibiotic
column** found in `data/mic_values.norm.csv` (after dropping isolates with missing labels
for each antibiotic).

```bash
conda activate BAMPY
scripts/run_golden_path.sh
```

Outputs include:
- `models/<antibiotic>_regression.pkl`
- `plots/<antibiotic>_regression.png`
- `models/training_summary.tsv`
- `logs/*.log, versions.txt, run_meta.txt`

Documentation
A detailed end-to-end workflow is provided in:
```
docs/WORKFLOW.md
```
(data layout → feature extraction → training → evaluation → interpretation)

Status
This repository contains code and documentation used for the *A. baumannii* AMR prediction
study (Pascoe & Mourkas et al., in preparation).

The codebase is actively being cleaned and documented. Optional feature views (gene
content, unitigs/GWAS, mobile element context) are present but may still be under
refinement.

*Notes*
### Antibiotic panel behaviour
Training is performed per antibiotic column after dropping isolates with missing
labels. As a result, a single run produces multiple trained models alongside
`training_summary.tsv`.

### Label sparsity
Some antibiotics have substantially fewer usable labels, which can reduce model stability
and inflate uncertainty. Always report sample sizes per antibiotic, as captured in
`training_summary.tsv`.
