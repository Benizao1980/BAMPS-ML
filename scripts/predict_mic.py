#!/usr/bin/env python3
"""
BAMPS-ML :: Prediction CLI
——————————————
Load trained models and predict S/I/R or MIC for new genomes.

Usage:
    python scripts/predict_mic.py \
      --model-dir outputs/ml_models \
      --features new_feature_table.tsv \
      --out predictions.tsv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from bamps_ml.utils import load_config  # to get breakpoint mapping, if needed


def load_models(model_dir: Path):
    """Return dict {antibiotic: model} for every *.pkl in model_dir."""
    models = {}
    for pkl in sorted(model_dir.glob("*.pkl")):
        # expect filenames like "ciprofloxacin_classification.pkl" or "ertapenem_regression.pkl"
        name = pkl.stem  # e.g. "ciprofloxacin_classification"
        ab, task = name.rsplit("_", 1)
        models[ab] = {"task": task, "model": pickle.load(open(pkl, "rb"))}
    return models


def predict_classification(model, X):
    """
    Predict S/I/R string and probability of non-susceptible.
    - if model is binary: proba = model.predict_proba[:, 1]
      and we map 0→"S", 1→"R"  (since you trained 0=S, 1=NS)
    - if model is multi: proba_NS = sum(proba for classes 1 and 2),
      and we choose argmax(proba) → label 0/1/2 → map back {0:"S",1:"I",2:"R"}.
    """
    proba = model.predict_proba(X)  # shape = (n_samples, n_classes)
    if proba.shape[1] == 2:
        # binary: [prob_S, prob_NS]
        prob_NS = proba[:, 1]
        label_int = (prob_NS > 0.5).astype(int)
    else:
        # multi: classes = [S(0), I(1), R(2)]
        # probability of NS = P(class=1) + P(class=2)
        prob_NS = proba[:, 1] + proba[:, 2]
        label_int = np.argmax(proba, axis=1)

    int_to_SIR = {0: "S", 1: "I", 2: "R"}
    labels = [int_to_SIR[i] for i in label_int]
    return labels, prob_NS


def predict_regression(model, X):
    """
    Predict log2(MIC). Return both log2 and back-transformed MIC.
    (Back-transform: MIC = 2**(predicted_log2_mic))
    """
    pred_log2 = model.predict(X)
    # If negative or non-integer floats, you may choose to round
    pred_MIC = np.power(2.0, pred_log2)
    return pred_log2, pred_MIC


def main():
    p = argparse.ArgumentParser(description="BAMPS-ML: Predict S/I/R or MIC")
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Directory containing '*_classification.pkl' and/or '*_regression.pkl'")
    p.add_argument("--features", required=True, type=Path,
                   help="TSV: rows=samples, cols=features (same as training input)")
    p.add_argument("--out", required=True, type=Path,
                   help="Path to write wide TSV of predictions")
    args = p.parse_args()

    # 1) Load feature table
    df_feat = pd.read_csv(args.features, sep="\t", index_col=0).fillna(0)
    samples = df_feat.index.tolist()

    # 2) Load models
    models = load_models(args.model_dir)
    if not models:
        raise ValueError(f"No .pkl models found in {args.model_dir}")

    # 3) For each antibiotic/model, align feature columns, predict
    results = pd.DataFrame(index=samples)
    for ab, info in models.items():
        mdl = info["model"]
        task = info["task"]
        # Reindex feature table to match training columns, fill missing columns with 0
        # Assumes the model object has .get_booster().feature_names stored (XGBoost).
        feat_names = mdl.get_booster().feature_names
        X_new = df_feat.reindex(columns=feat_names, fill_value=0)

        if task == "classification":
            labels, prob_ns = predict_classification(mdl, X_new)
            results[f"{ab}_pred"]        = labels
            results[f"{ab}_prob_NS"]     = prob_ns
        else:  # regression
            pred_log2, pred_MIC = predict_regression(mdl, X_new)
            results[f"{ab}_log2MIC"]     = pred_log2
            results[f"{ab}_pred_MIC"]     = pred_MIC

    # 4) Write out
    results.to_csv(args.out, sep="\t")
    print(f"Predictions written → {args.out}")


if __name__ == "__main__":
    main()
