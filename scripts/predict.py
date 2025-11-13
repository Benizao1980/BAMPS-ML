#!/usr/bin/env python3
"""
BAMPS-ML :: Predictor
-----------------------------------
Apply a trained model (per antibiotic) to a feature matrix.

Examples:
  # S/I/R classification
  python scripts/predict.py \
    --feature-table outputs/amrfinder/amr_presence_absence.tsv \
    --model-dir outputs/ml_models \
    --antibiotic meropenem \
    --task classification \
    --output outputs/preds/preds_meropenem_SIR.tsv

  # MIC regression (back to raw mg/L)
  python scripts/predict.py \
    --feature-table outputs/amrfinder/amr_presence_absence.tsv \
    --model-dir outputs/ml_models \
    --antibiotic meropenem \
    --task regression \
    --to-mic \
    --output outputs/preds/preds_meropenem_mic.tsv
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-table", required=True, type=Path,
                   help="TSV with samples as rows and features as columns (index col = sample).")
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Directory containing <antibiotic>_<task>.pkl")
    p.add_argument("--antibiotic", required=True,
                   help="Antibiotic name (must match model filename prefix).")
    p.add_argument("--task", choices=["classification", "regression"], default="classification",
                   help="classification = S/I/R; regression = log2(MIC)")
    p.add_argument("--to-mic", action="store_true",
                   help="(regression only) exponentiate log2 predictions back to MIC (mg/L)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output TSV path")
    return p.parse_args()


def load_and_align_features(X_path: Path, model) -> pd.DataFrame:
    """Load feature table and align its columns to exactly what the model expects."""
    X = pd.read_csv(X_path, sep="\t", index_col=0)
    X.columns = X.columns.astype(str)

    # Get expected feature names from model
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        try:
            expected = model.get_booster().feature_names  # xgboost
        except Exception:
            expected = None
    if expected is None:
        # Fallback: assume current columns (won't fix a mismatch, but avoids crash)
        expected = list(X.columns)

    expected = [str(c) for c in expected]
    current = set(map(str, X.columns))

    missing = [c for c in expected if c not in current]
    extra = [c for c in X.columns if c not in set(expected)]

    # Add missing training features as zeros (gene absent)
    for c in missing:
        X[c] = 0
    # Drop unseen features at predict time
    if extra:
        X.drop(columns=extra, inplace=True, errors="ignore")

    # Exact order + numeric coercion
    X = X.reindex(columns=expected, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0)

    LOG = logging.getLogger("bamps_ml.predict")
    if not LOG.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if missing:
        LOG.info("Added %d missing training features as 0 (e.g., %s)", len(missing), ", ".join(missing[:5]))
    if extra:
        LOG.info("Dropped %d unseen prediction features (e.g., %s)", len(extra), ", ".join(extra[:5]))

    return X


def main():
    args = parse_args()

    # Load model first (so we can query expected features)
    model_path = args.model_dir / f"{args.antibiotic}_{args.task}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    # Load & align features
    X = load_and_align_features(args.feature_table, model)

    # Predict
    if args.task == "classification":
        yhat = model.predict(X)
        # Some wrappers return probabilities (n_samples × n_classes)
        if hasattr(yhat, "ndim") and yhat.ndim == 2:
            yhat = np.asarray(yhat).argmax(axis=1)
        yhat = np.asarray(yhat).astype(int)

        # Determine number of classes and map to names
        n_classes = getattr(model, "n_classes_", None)
        if n_classes is None:
            n_classes = int(yhat.max() + 1) if yhat.size else 1
        # Training reindexed labels to 0..K-1 with S=0 < I=1 < R=2
        names_by_k = {1: ["S"], 2: ["S", "R"], 3: ["S", "I", "R"]}
        names = names_by_k.get(n_classes, ["S", "I", "R"][:n_classes])
        pred_labels = [names[i] if 0 <= i < len(names) else "S" for i in yhat]

        result_df = pd.DataFrame({"sample": X.index, "prediction": pred_labels})

    else:  # regression: model predicts log2(MIC)
        log2_pred = model.predict(X)
        if args.to_mic:
            mic = np.power(2.0, log2_pred)
            result_df = pd.DataFrame({"sample": X.index, "prediction": mic})
        else:
            result_df = pd.DataFrame({"sample": X.index, "prediction": log2_pred})

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, sep="\t", index=False)
    print(f"Predictions written → {args.output}")


if __name__ == "__main__":
    main()
