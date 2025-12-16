# Machine-Learning–Based Prediction of Antimicrobial MICs in *Acinetobacter baumannii*

## Study Overview

We developed a genome-based machine-learning (ML) framework to predict minimum inhibitory concentrations (MICs) for *Acinetobacter baumannii*. The approach integrates curated antimicrobial resistance (AMR) determinants with genome-wide association study (GWAS) features derived from multiple variant classes. Models were trained on a well-characterised reference dataset, externally validated using an independent collection, and subsequently applied at scale to public genomes to explore temporal and geographic resistance patterns.

---

## Genome Datasets

### Training Dataset

Model training was performed using a collection of *A. baumannii* isolates from Russia (*n* = 280), for which high-quality genome assemblies and corresponding phenotypic MIC data were available. This dataset has been extensively characterised previously and includes diverse resistance profiles suitable for supervised learning.

---

### External Validation Dataset

To assess model generalisability, we used an independent *A. baumannii* collection curated at the Ineos Oxford Institute (IOI), comprising *n* = 671 genomes. This dataset spans multiple geographic regions and lineages and was not used at any stage of model training or optimisation.

An overview of the validation dataset, including phylogenetic structure and metadata, is available via Microreact:  
https://microreact.org/project/iKZYYn4RTveDvfDPn18A6J-a-baumanii-validation-dataset-ioi-collection

---

### Public Genome Dataset

Following validation, the trained model was applied to a large publicly available *A. baumannii* genome collection obtained from pubMLST.

- **Database:** pubMLST  
- **Date accessed:** <insert date>  
- **Sample size:** *n* = x,xxx genomes  

This dataset was used exclusively for inference and exploratory analyses and did not contribute to model training or tuning.

---

## AMR Determinant Detection

All genomes across training, validation, and public datasets were screened for known AMR genes and resistance-associated mutations using **AMRFinderPlus**. The same software version and database release were used throughout to ensure consistency.

```bash
# Run AMRFinderPlus on genome assemblies
<insert command>
```

Detected AMR determinants were encoded as binary (presence/absence) or categorical features and included directly as predictors in downstream ML models.

---

## GWAS Feature Generation and Integration

### Source GWAS Analyses

Genome-wide association analyses were previously conducted using **pyseer**, generating statistically significant associations between MIC phenotypes and genomic features from multiple classes:

- Gene presence/absence variants  
- Unitig-based variants  
- Single nucleotide polymorphisms (SNPs)  

Each GWAS output captures complementary aspects of resistance-associated genomic variation.

---

### Combining GWAS Feature Classes

GWAS outputs from gene, unitig, and SNP analyses were combined into a unified feature catalogue, retaining only features passing predefined significance and quality thresholds.

```bash
# Combine GWAS outputs across feature classes
<insert command>
```

---

### Mapping GWAS Features to Genomes

For each dataset (training, validation, and public), GWAS features were identified and encoded within individual genomes to generate a consistent feature matrix compatible with trained models.

```bash
# Identify GWAS features in a given dataset
<insert command>
```

All feature matrices were harmonised to ensure identical feature ordering and encoding across datasets.

---

## Machine-Learning Model Training

### Model Types

We evaluated multiple supervised ML approaches to predict MIC values, including:

- Linear regression–based models  
- Regularised linear models  
- Tree-based ensemble models  
- Gradient boosting methods (including XGBoost)  

Models were trained using a combined feature set comprising AMRFinder-derived AMR determinants and GWAS-derived genomic features.

---

### Hyperparameter Optimisation

For each model class, hyperparameters were optimised using cross-validation within the training dataset. Optimisation aimed to maximise predictive performance while limiting overfitting.

```bash
# Train models and perform hyperparameter optimisation
<insert command>
```

The best-performing model was selected based on cross-validated performance metrics and retained for downstream validation.

---

## External Validation and Model Evaluation

### Feature Extraction in Validation Dataset

AMR determinants and GWAS features were identified in the IOI validation dataset using the same procedures applied to the training dataset.

```bash
# Run AMRFinderPlus on validation dataset
<insert command>

# Identify GWAS features in validation dataset
<insert command>
```

---

### MIC Prediction

The trained ML model was applied to the validation dataset to generate MIC predictions for each isolate.

```bash
# Predict MICs in validation dataset
<insert command>
```

---

### Performance Metrics

Predicted MICs were compared against experimentally measured values using multiple complementary metrics, including:

- Accuracy within ±1 and ±2 doubling dilutions  
- F1 score  
- Error distributions and residual analyses  

```bash
# Compute and plot prediction performance metrics
<insert command>
```

These analyses were used to quantify model accuracy, robustness, and generalisability across diverse lineages.

---

## Application to Public Genome Collections

### Feature Extraction and MIC Prediction

The validated model was applied to the pubMLST genome collection. AMR determinants and GWAS features were identified prior to MIC prediction.

```bash
# Run AMRFinderPlus on pubMLST genomes
<insert command>

# Identify GWAS features in pubMLST dataset
<insert command>

# Predict MICs in pubMLST dataset
<insert command>
```

---

## Temporal and Geographic Analyses

Predicted MICs from the pubMLST dataset were aggregated and analysed to explore resistance trends across:

- Sampling year  
- Geographic region  
- Phylogenetic lineage  

```bash
# Compute temporal and geographic summaries
<insert commands>
```

These analyses enable large-scale inference of resistance dynamics beyond the limits of phenotypically characterised datasets.

---

## Reproducibility and Consistency

All analyses were performed using consistent software versions, databases, and feature encodings across datasets. Training, validation, and inference datasets were strictly separated to prevent information leakage. Scripts used for feature extraction, model training, and prediction are available upon request or via the associated repository.

---

## Summary

This ML framework integrates curated AMR determinants and GWAS-derived genomic features to enable accurate MIC prediction in *A. baumannii*. External validation and large-scale application demonstrate its utility for both resistance prediction and population-level epidemiological analyses.
