# BAMPS-ML

**BAMPS-ML** is an interpretable machine-learning workflow for predicting antimicrobial resistance (AMR) phenotypes from bacterial whole-genome sequence data. It supports multiple genomic feature “views” (e.g., AMRFinder determinants, gene content, genome-wide sequence variation) and is designed to be reproducible, auditable, and extensible across pathogens.

Primary use case: antibiotic susceptibility prediction and biological interpretation of resistance architecture, demonstrated using *Acinetobacter baumannii*.

## Key ideas
- Compare **rule-based AMR prediction** (known determinants) vs **ML** on the same isolates.
- Support multiple feature sets:
  - AMR determinants (AMRFinderPlus)
  - Annotation/gene content (Bakta ± pangenome)
  - Unitigs / genome-wide variation (optional)
  - Mobile element context (MOB-suite; optional)
- Provide **interpretable outputs** (feature importance / SHAP) to connect predictions to biology.

## Quick start
See **docs/WORKFLOW.md** for an end-to-end run: data layout → feature extraction → training → evaluation → interpretation.

## Repo status
This repository contains the workflow code and documentation used for the *A. baumannii* AMR prediction study.
