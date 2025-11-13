#!/usr/bin/env python3
"""
BAMPS-ML :: Model Trainer (enhanced)
-----------------------------------
Train per-antibiotic models from a feature matrix and MIC phenotypes.
* **Classification** (S/I/R) or **Regression** (log₂-MIC)
* **Confusion-matrix PNGs** (classification)
* **SHAP beeswarm PNGs** for feature importance
* **Optional bootstrap evaluation** (--bootstrap-reps N)
* **Self-test split**: 25 % of the *train* data can be held back for an inner validation (--self-test)

Typical CLI
-----------
python scripts/train_model.py \
  --feature-table outputs/amr_presence_absence.tsv \
  --mic-file data/mic_values_test88.csv \
  --model-dir outputs/ml_models \
  --plot-dir outputs/plots \
  --task classification \
  --cv-folds 5 \
  --bootstrap-reps 10
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor

from bamps_ml.utils import load_config, resolve_breakpoints, mic_to_category
from bamps_ml.plotting import PALETTE, confusion_plot, shap_beeswarm
from bamps_ml.evaluation import (
    classification_metrics,
    regression_metrics,
    write_training_summary,
    to_dict as eval_to_dict,
)

LOG = logging.getLogger("bamps_ml.train_model")
plt.rcParams["figure.dpi"] = 150  # decent resolution


def cross_val_metrics_classification(
    model_fn, X: pd.DataFrame, y: pd.Series, K: int, seed: int
) -> dict:
    """Return mean/std of accuracy and macro-F1 across K stratified folds."""
    # Clamp folds so each class has at least one sample per fold
    try:
        min_class = int(y.value_counts().min())
        safe_K = max(2, min(K, min_class))
        if safe_K < K:
            LOG.warning("Reducing CV folds from %d to %d due to small class sizes", K, safe_K)
    except Exception:
        safe_K = max(2, K)
    skf = StratifiedKFold(n_splits=safe_K, shuffle=True, random_state=seed)
    accs: list[float] = []
    f1s: list[float] = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        m = model_fn()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_va)
        accs.append(accuracy_score(y_va, y_pred))
        f1s.append(f1_score(y_va, y_pred, average="macro"))
    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs, ddof=1)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s, ddof=1)),
    }


def cross_val_metrics_regression(
    model_fn, X: pd.DataFrame, y: pd.Series, K: int, seed: int
) -> dict:
    """Return mean/std of MAE and R² across K folds (unstratified)."""
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    maes: list[float] = []
    r2s: list[float] = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        m = model_fn()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_va)
        maes.append(mean_absolute_error(y_va, y_pred))
        r2s.append(r2_score(y_va, y_pred))
    return {
        "cv_MAE_mean": float(np.mean(maes)),
        "cv_MAE_std": float(np.std(maes, ddof=1)),
        "cv_R2_mean": float(np.mean(r2s)),
        "cv_R2_std": float(np.std(r2s, ddof=1)),
    }

def prepare_labels(mic: pd.Series, task: str, bp: dict) -> pd.Series:
    """
    If task == "classification", map raw MIC → "S"/"I"/"R" strings (drop NaN).
    If task == "regression", return log₂(MIC) (drop NaN).
    """
    if task == "classification":
        return mic.apply(lambda x: mic_to_category(x, bp)).dropna()
    elif task == "regression":
        return np.log2(mic).dropna()
    else:
        raise ValueError("task must be 'classification' or 'regression'")

def build_model(task: str, n_classes: int | None, random_state: int):
    """
    Return an XGBoost model:
      - if task == "classification": a binary:logistic (if n_classes == 2)
        or multi:softprob (if n_classes > 2) classifier.
      - if task == "regression": a reg:squarederror regressor.
    """
    if task == "classification":
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        return XGBClassifier(
            objective=objective,
            num_class=(n_classes if n_classes and n_classes > 2 else None),
            eval_metric="logloss",
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    else:  # regression
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )

def eval_metrics(model, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> tuple[dict, np.ndarray|None]:
    """
    If task == "classification", return ({"accuracy": ...}, confusion_matrix).
    If task == "regression", return ({"MAE": ..., "R2": ...}, None).
    """
    y_pred = model.predict(X_test)
    if task == "classification":
         # Robust multi-class handling: if probabilities (n_samples x n_classes) → argmax
        if hasattr(y_pred, "ndim") and getattr(y_pred, "ndim", 1) == 2:
            y_pred = np.asarray(y_pred).argmax(axis=1)
        # Use new evaluation helpers (with bootstrap CIs)
        rpt, cm, labels = classification_metrics(
            y_test.values, y_pred, labels=None, n_boot=500, confidence=0.95
        )
        return eval_to_dict(rpt), cm
    else:
        # regression mode → use new evaluation helper (R²/MAE/RMSE + CIs)
        rpt = regression_metrics(y_test.values, y_pred, n_boot=500, confidence=0.95)
        return eval_to_dict(rpt), None

def bootstrap_eval(model_fn, X: pd.DataFrame, y: pd.Series, reps: int, seed: int) -> dict:
    """
    Perform `reps` bootstrap draws (with replacement) from (X,y).
    Each draw is of size len(X).  We skip any draw where `y_b` has only a single class.
    Then we train a classification model on (X_b, y_b) and evaluate accuracy on the same (X_b,y_b).
    Return { "mean": {...}, "std": {...} } of the scalar metrics (only “accuracy” here).
    """
    rng = random.Random(seed)
    metrics_list: list[dict] = []

    for _ in range(reps):
        idx = rng.choices(range(len(X)), k=len(X))
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        # If the bootstrap draw has only one class, skip it entirely.
        
        # (1) shift so labels start at 0, but then remap to consecutive integers
        y_b = y_b - y_b.min()    # e.g. {1,2} → {0,1}
        if y_b.nunique() < 2:    # skip any bootstrap draw that has only one class
            continue

        # (2) build a mapping from the present classes → 0..(n_classes-1)
        unique_labels = sorted(y_b.unique())          # e.g. [0,2] or [0,1,2]
        mapping = {orig: i for i, orig in enumerate(unique_labels)}
        y_b_mapped = y_b.map(mapping)

        n_cls = len(unique_labels)
        m = build_model("classification", n_cls, seed)
        m.fit(X_b, y_b_mapped)
        scalars, _ = eval_metrics(m, X_b, y_b_mapped, "classification")  # use the remapped y_b here
        metrics_list.append(scalars)

    if not metrics_list:
        return {"mean": {}, "std": {}}

    df_boot = pd.DataFrame(metrics_list)
    return df_boot.agg(["mean", "std"]).to_dict()


def main():
    p = argparse.ArgumentParser(description="BAMPS-ML model trainer (enhanced)")
    p.add_argument("--feature-table", required=True, type=Path)
    p.add_argument("--mic-file", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--plot-dir", required=True, type=Path)
    p.add_argument("--config", default="config/config.yaml", type=Path)
    p.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--bootstrap-reps",
        type=int,
        default=0,
        help="Number of bootstrap replicates (0 = skip).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Run K‐fold CV on the *training* split (K ≥ 2).",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Hold back 25% of the training set for an inner validation.",
    )
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cfg = load_config(args.config)
    bp_table = resolve_breakpoints(cfg["breakpoint_standard"], cfg.get("mic_thresholds"))

    # ── Load feature table (presence/absence) and MIC file ────────────────────
    X = pd.read_csv(args.feature_table, sep="\t", index_col=0)
    mic_df = pd.read_csv(args.mic_file)

    # If there’s a “fasta_file” column, strip the extension
    if "fasta_file" in mic_df.columns:
        mic_df["sample"] = mic_df["fasta_file"].apply(lambda x: Path(str(x)).stem)
        mic_df = mic_df.set_index("sample")
    elif "ID" in mic_df.columns:
        mic_df = mic_df.set_index("ID")
    else:
        # Otherwise, whatever the first column is, use that as sample ID
        mic_df = mic_df.set_index(mic_df.columns[0])

    # Keep only the shared sample IDs between X and mic_df
    shared = X.index.intersection(mic_df.index)
    X, mic_df = X.loc[shared], mic_df.loc[shared]
    LOG.info("Shared samples: %d", len(shared))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []

    for ab in cfg["antibiotics"]:
        if ab not in mic_df.columns:
            continue
        LOG.info("— %s —", ab)

        # 1) Compute “raw” labels: either S/I/R (classification) or log2(MIC) (regression).
        y_cat = prepare_labels(mic_df[ab], args.task, bp_table.get(ab, {}))

        # Classification: count n_S, n_I, n_R *before* encoding to integers
        if args.task == "classification":
            n_S = int((y_cat == "S").sum())
            n_I = int((y_cat == "I").sum())
            n_R = int((y_cat == "R").sum())
            mic_vals = mic_df[ab].loc[y_cat.index].astype(float)
            mic_mean, mic_min, mic_max = (
                float(mic_vals.mean()),
                float(mic_vals.min()),
                float(mic_vals.max()),
            )
            # Now map “S”→0, “I”→1, “R”→2 and drop any NaN
            y = y_cat.map({"S": 0, "I": 1, "R": 2}).dropna().astype(int)
        else:
            # Regression: y_cat already holds log2(MIC) for every sample
            y = y_cat.astype(float)
            n_S = n_I = n_R = None
            mic_mean, mic_min, mic_max = (
                float(y.mean()),
                float(y.min()),
                float(y.max()),
            )

        # Subset X to only those samples for which we have a valid y
        X_ab = X.loc[y.index]
        if len(set(y)) < 2:
            LOG.warning("Skipping %s (fewer than 2 distinct labels)", ab)
            continue

        # --- Reindex labels to 0..K-1 (handles S/R-only etc.) ----------------
        # y currently uses global codes {0:"S", 1:"I", 2:"R"}; present may be {0,2}, etc.
        present = np.sort(pd.unique(y))
        lab2idx = {lab: i for i, lab in enumerate(present)}  # e.g., {0:0, 2:1}
        _sir = {0: "S", 1: "I", 2: "R"}
        idx2sir = {i: _sir.get(orig, str(orig)) for orig, i in lab2idx.items()}  # e.g., {0:"S",1:"R"}
        y_enc = y.map(lab2idx).astype(int)

        # 2) Split into train/test (with stratification if classification)
        stratify_arg = y_enc if args.task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_ab,
            y_enc,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_arg,
        )

        # 3) If requested, hold back 25% of *training* set as an inner validation
        if args.self_test:
            stratify_inner = y_train if args.task == "classification" else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.25,
                random_state=args.random_state,
                stratify=stratify_inner,
            )

        # 4) Re-index labels to 0..K-1 for this antibiotic (handles S/R-only etc.)
        if args.task == "classification":
            present_labels = np.sort(pd.unique(y_train))  # original numeric labels in {0,1,2} subset
            lab2idx = {lab: i for i, lab in enumerate(present_labels)}
            idx2sir_full = {0: "S", 1: "I", 2: "R"}
            idx2name = {i: idx2sir_full.get(lab, str(lab)) for lab, i in lab2idx.items()}
            y_train_enc = y_train.map(lab2idx).astype(int)
            y_test_enc  = y_test.map(lab2idx).astype(int)
        else:
            y_train_enc, y_test_enc = y_train, y_test

        # 4a) Build appropriate model_fn (use encoded label space)
        if args.task == "classification":
            model_fn = lambda: build_model("classification", len(set(y_train_enc)), args.random_state)
        else:
            model_fn = lambda: build_model("regression", None, args.random_state)

        model = model_fn()
        model.fit(X_train, y_train_enc)

        # 5) Evaluate on test set
        scalars, cm = eval_metrics(model, X_test, y_test_enc, args.task)

        # 6) Optionally do bootstrap on the training set
        if args.bootstrap_reps:
            scalars["bootstrap"] = bootstrap_eval(
                model_fn, X_train, y_train_enc, args.bootstrap_reps, args.random_state
            )

        # 7) Optionally run K-fold CV on the *training* split
        if args.cv_folds and args.cv_folds >= 2:
            if args.task == "classification":
                cv_stats = cross_val_metrics_classification(
                    lambda: build_model("classification", len(set(y_train_enc)), args.random_state),
                    X_train,
                    y_train_enc,
                    args.cv_folds,
                    args.random_state,
                )
            else:  # regression
                cv_stats = cross_val_metrics_regression(
                    lambda: build_model("regression", None, args.random_state),
                    X_train,
                    y_train,
                    args.cv_folds,
                    args.random_state,
                )
            scalars.update(cv_stats)

        # 8) If regression, produce a “true vs predicted” scatter
        if args.task == "regression":
            y_true = y_test
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", s=30)
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("True log₂(MIC)", fontweight="bold")
            ax.set_ylabel("Predicted log₂(MIC)", fontweight="bold")
            ax.set_title(f"{ab} regression", fontsize=10)
            ax.set_aspect("equal", "box")
            plt.tight_layout()
            reg_png = args.plot_dir / f"{ab}_regression.png"
            fig.savefig(reg_png, dpi=150)
            plt.close(fig)

        # 9) Save the fitted model to disk
        mdl_path = args.model_dir / f"{ab}_{args.task}.pkl"
        pickle.dump(model, open(mdl_path, "wb"))

        # 10) If classification, produce confusion‐matrix + SHAP plot
        if args.task == "classification":
            cm_png = args.plot_dir / f"{ab}_confusion.png"
            # Use present-class names (e.g., ["S","R"] if no "I")
            lab_names = [idx2name[i] for i in range(cm.shape[0])]
            confusion_plot(cm, lab_names, cm_png)
            shap_png = args.plot_dir / f"{ab}_shap.png"
            shap_beeswarm(model, X_ab, shap_png)


        # 11) Finally, build a summary row for this antibiotic:
        row: dict = {
            "antibiotic": ab,
            "n_samples": int(len(y_cat)),
            "n_S": n_S,
            "n_I": n_I,
            "n_R": n_R,
            "mic_mean": mic_mean,
            "mic_min": mic_min,
            "mic_max": mic_max,
        }
        # merge in accuracy, bootstrap‐ and CV‐metrics
        row.update(scalars)
        summary.append(row)

    # 12) Write out the complete summary table
    if summary:
        out_tsv = write_training_summary(summary, args.model_dir / "training_summary.tsv")
        LOG.info("Training summary saved → %s", out_tsv)
    else:
         LOG.warning("No models trained – check logs for issues.")

if __name__ == "__main__":
    main()
