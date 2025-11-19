# BAMPS-ML: AMR prediction, evaluation & mapping

End-to-end pipeline for predicting S/I/R classes and MICs from AMRFinder feature tables, plus post-hoc confusion matrices, MIC panel plots, and an interactive geospatial map (with optional country-level choropleth).
Designed around tidy outputs (sample, pred) to make downstream merging simple and robust.

What’s implemented (so far)

Prediction

Classification (S/I/R) → preds_<antibiotic>_SIR.tsv

Regression (MIC) → preds_<antibiotic>_mic.tsv

Schema-safe feature alignment: missing training features → 0; unseen prediction features → drop, with informative logs.

Batch runner: scripts/predict_all.py

Runs all available models in a directory (classification &/or regression).

Writes tidy per-drug TSVs.

Optional extras: confusion matrices (from SIR vs truth) and MIC panel plots.

Evaluation & visualisation

scripts/plot_SIR_confusion.py — confusion matrices from predicted S/I/R vs truth (truth can be SIR or MICs mapped via breakpoints).

scripts/plot_predicted_mic_panel.py — MIC panel across drugs (optional log scale), with accuracy badges from training summary.

scripts/plot_isolates_map.py — interactive Folium map with:

point colouring by S/I/R or MIC,

optional marker clustering showing %R and n,

optional country-level choropleth: prevalence_R / median_mic / count,

static PNG/SVG export when geopandas is available.

Extra merge diagnostics in the map script (SIR counts, matched rows, example joined records).

Installation
# (recommended) create an environment
conda create -n bamps_ml python=3.10 -y
conda activate bamps_ml

# install deps
pip install -r requirements.txt
# optional (for static map export & ISO conversion)
# pip install geopandas country_converter


Core deps: numpy, pandas, scikit-learn, xgboost, matplotlib, pyyaml, folium, branca
Optional: geopandas, country_converter

Note: XGBoost may warn about older glibc on some clusters — safe to ignore for now.

Inputs

AMRFinder features (prediction input)
outputs/amrfinder/<dataset>/amr_presence_absence.tsv — wide binary matrix.

Models (trained pickles, named by convention)
Placed in e.g. outputs/ml_models/ or models/:

ciprofloxacin_classification.pkl
meropenem_classification.pkl
imipenem_regression.pkl
...
training_summary.tsv            # optional for MIC panel badges


Truth data (optional; for evaluation/plots)

data/truth_MICs.csv — per-sample MICs (will be mapped to S/I/R using config/config.yaml), or

a CSV with S/I/R calls directly.

Breakpoints (for MIC→SIR in evaluation)

config/config.yaml

Geospatial metadata (optional; for map)

data/geo_global.csv with columns ID, lat, lon, Country

World GeoJSON: data/world_countries_iso_a3.geojson

Typical workflow
1) Run predictions for all available models
python scripts/predict_all.py \
  --feature-table outputs/amrfinder/Global_dataset/amr_presence_absence.tsv \
  --model-dir outputs/ml_models \
  --outdir outputs/preds/Nov19_global \
  --tasks classification regression


Outputs (tidy TSVs with sample, pred):

outputs/preds/Nov19_global/
  preds_ciprofloxacin_SIR.tsv
  preds_colistin_SIR.tsv
  preds_imipenem_SIR.tsv
  preds_meropenem_SIR.tsv
  preds_<antibiotic>_mic.tsv    # multiple drugs

2) (Optional) Confusion matrices (SIR vs truth)
python scripts/plot_SIR_confusion.py \
  --pred-glob 'outputs/preds/Nov19_global/preds_*_SIR.tsv' \
  --truth data/truth_MICs.csv \
  --outdir outputs/figs/confusion \
  --order ciprofloxacin colistin imipenem meropenem \
  --config config/config.yaml \
  --id-col ID


If truth file has S/I/R, it’s used directly.

If it has MICs, mapping to S/I/R uses config/config.yaml.

3) (Optional) MIC panel plot
python scripts/plot_predicted_mic_panel.py \
  --pred-glob 'outputs/preds/Nov19_global/preds_*_mic.tsv' \
  --order meropenem imipenem ciprofloxacin colistin \
  --out outputs/figs/mic_panel/NOV19_global_panel \
  --config config/config.yaml \
  --summary outputs/ml_models/training_summary.tsv \
  --log10 \
  --truth data/truth_MICs.csv --truth-id-col ID --truth-lower

4) (Optional) Interactive map
python scripts/plot_isolates_map.py \
  --metadata data/geo_global.csv \
  --id-col ID \
  --country-col Country \
  --pred-file outputs/preds/Nov19_global/preds_meropenem_SIR.tsv \
  --value sir \
  --choropleth --metric prevalence_R \
  --world-geojson data/world_countries_iso_a3.geojson \
  --geojson-iso3-key ISO3166-1-Alpha-3 \
  --title "Meropenem S/I/R (Global)" \
  --out-html outputs/maps/global_meropenem.html \
  --cluster


Tips

The map script prints merge diagnostics (SIR counts, matched rows, example joined records).

If you see many None in SIR counts and 0 matched rows, your sample IDs in predictions don’t match ID in geo_global.csv — normalise/clean IDs so they match exactly.

Data schemas

AMRFinder features — amr_presence_absence.tsv

Rows: samples; Columns: binary features (e.g., GENE:aadA27).

Prediction step:

Adds missing training features with 0,

Drops unseen features at prediction time.

Predictions (tidy)

preds_<antibiotic>_SIR.tsv → columns: sample, pred (S|I|R)

preds_<antibiotic>_mic.tsv → columns: sample, pred (float MIC)

Geospatial metadata — data/geo_global.csv

Required columns: ID, lat, lon, Country (string Country names are OK)

ID must match sample values in prediction TSVs.

Repository layout (recommended)
.
├── README.md
├── LICENSE
├── CITATION.cff                       # optional
├── requirements.txt
├── environment.yml                    # optional (conda)
├── .gitignore
├── config/
│   └── config.yaml                    # MIC→SIR breakpoints
├── scripts/
│   ├── run_amrfinder.py
│   ├── train_model.py
│   ├── predict.py
│   ├── predict_mic.py
│   ├── predict_all.py
│   ├── plot_SIR_confusion.py
│   ├── plot_predicted_mic_panel.py
│   └── plot_isolates_map.py
├── data/
│   ├── world_countries_iso_a3.geojson
│   └── geo_global.csv                 # TEMPLATE ONLY (example rows)
└── outputs/                           # (ignored) predictions, figures, maps

Files to commit vs ignore

Commit these

All scripts in scripts/

config/config.yaml

data/world_countries_iso_a3.geojson

data/geo_global.csv as a template (no real coordinates)

requirements.txt, README.md, LICENSE, .gitignore (+ optional CITATION.cff, environment.yml)

Do NOT commit (or use Git-LFS)

Large model pickles (*.pkl) → release asset / Zenodo / Git-LFS

outputs/ directory (derived artifacts)

Any sensitive/raw data (truth MICs with identifiers, real coordinates)

Minimal .gitignore

outputs/
*.pkl
*.png
*.svg
*.html
.DS_Store
__pycache__/
*.pyc
.ipynb_checkpoints/

Repro commands used on Nov 19 (for provenance)
# Predictions (classification + regression)
python scripts/predict_all.py \
  --feature-table outputs/amrfinder/Global_dataset/amr_presence_absence.tsv \
  --model-dir outputs/ml_models \
  --outdir outputs/preds/Nov19_global \
  --tasks classification regression

# Confusion matrices (SIR vs truth)
python scripts/plot_SIR_confusion.py \
  --pred-glob 'outputs/preds/Nov19_global/preds_*_SIR.tsv' \
  --truth data/truth_MICs.csv \
  --outdir outputs/figs/confusion \
  --order ciprofloxacin colistin imipenem meropenem \
  --config config/config.yaml \
  --id-col ID

# MIC panel
python scripts/plot_predicted_mic_panel.py \
  --pred-glob 'outputs/preds/Nov19_global/preds_*_mic.tsv' \
  --order meropenem imipenem ciprofloxacin colistin \
  --out outputs/figs/mic_panel/NOV19_global_panel \
  --config config/config.yaml \
  --summary outputs/ml_models/training_summary.tsv \
  --log10 \
  --truth data/truth_MICs.csv --truth-id-col ID --truth-lower

# Interactive map
python scripts/plot_isolates_map.py \
  --metadata data/geo_global.csv \
  --id-col ID --country-col Country \
  --pred-file outputs/preds/Nov19_global/preds_meropenem_SIR.tsv \
  --value sir \
  --choropleth --metric prevalence_R \
  --world-geojson data/world_countries_iso_a3.geojson \
  --geojson-iso3-key ISO3166-1-Alpha-3 \
  --title "Meropenem S/I/R (Global)" \
  --out-html outputs/maps/global_meropenem.html \
  --cluster

Troubleshooting

Map shows 0% R and grey points
The prediction TSV didn’t merge with the geo table (ID mismatch). Check:

head -n5 outputs/preds/Nov19_global/preds_meropenem_SIR.tsv
head -n5 data/geo_global.csv


Ensure sample (predictions) matches ID (geo). Normalise case, strip prefixes/suffixes if needed.

Confusion plotting “No such file or directory”
Make sure --truth points to the file you actually have (earlier example used a missing data/mic_global.csv).

MIC panel can’t find training summary
Use --summary outputs/ml_models/training_summary.tsv (or the correct path to your summary TSV).

“No models found”
--model-dir must contain files named <antibiotic>_classification.pkl and/or <antibiotic>_regression.pkl.

Roadmap

Add plot_isolates_map.py ID-normalisation flags (lowercasing, strip prefix/suffix) to reduce merge friction.

Package the CLI (pip install -e .), add unit tests, and set up CI.

Optional: publish models as release assets with a helper downloader.

Citation
If you use this code, please cite the repository (and any companion manuscript when available). A CITATION.cff can be added for GitHub-native citation metadata.
