#!/usr/bin/env python3
from __future__ import annotations
"""
bamps_ml.evaluation — metrics, bootstrap CIs, CV aggregation, summaries
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_label_preds(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """If y_pred are probabilities (n_samples, n_classes), convert via argmax."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)
    return y_true, y_pred


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = 42,
    stratify: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Generic bootstrap CI for a metric. If `stratify` is provided (e.g., class labels),
    resample within strata to preserve class balance.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if stratify is None:
        idx = np.arange(n)
        samples = [rng.choice(idx, size=n, replace=True) for _ in range(n_boot)]
    else:
        # stratified resampling
        samples = []
        classes = np.unique(stratify)
        per_class_idx = {c: np.where(stratify == c)[0] for c in classes}
        for _ in range(n_boot):
            boots = []
            for c in classes:
                grp = per_class_idx[c]
                boots.append(rng.choice(grp, size=len(grp), replace=True))
            samples.append(np.concatenate(boots))

    stats = []
    for s in samples:
        try:
            stats.append(metric_fn(y_true[s], y_pred[s]))
        except Exception:
            # if metric fails on degenerate resample, skip
            continue

    if not stats:
        return np.nan, np.nan

    lo = (1 - confidence) / 2
    hi = 1 - lo
    q_lo, q_hi = np.quantile(stats, [lo, hi])
    return float(q_lo), float(q_hi)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    accuracy: float
    accuracy_lo: float
    accuracy_hi: float
    f1_macro: float
    f1_macro_lo: float
    f1_macro_hi: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float


def classification_metrics(
    y_true,
    y_pred,
    labels: Iterable[int] | None = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
) -> tuple[ClassificationReport, np.ndarray, list]:
    """
    Compute core metrics + bootstrap CIs for accuracy and f1_macro.
    Returns (report_dataclass, confusion_matrix, label_list_used).
    """
    y_true, y_pred = _ensure_label_preds(y_true, y_pred)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Bootstrap (stratify by true labels to keep class balance)
    acc_lo, acc_hi = _bootstrap_ci(
        y_true, y_pred, metric_fn=lambda a, b: accuracy_score(a, b),
        n_boot=n_boot, confidence=confidence, stratify=y_true
    )
    f1_lo, f1_hi = _bootstrap_ci(
        y_true, y_pred, metric_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        n_boot=n_boot, confidence=confidence, stratify=y_true
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = ClassificationReport(
        accuracy=float(acc),
        accuracy_lo=acc_lo, accuracy_hi=acc_hi,
        f1_macro=float(f1m),
        f1_macro_lo=f1_lo, f1_macro_hi=f1_hi,
        balanced_accuracy=float(bacc),
        precision_macro=float(prec),
        recall_macro=float(rec),
    )
    return report, cm, list(labels)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

@dataclass
class RegressionReport:
    r2: float
    r2_lo: float
    r2_hi: float
    mae: float
    mae_lo: float
    mae_hi: float
    rmse: float
    rmse_lo: float
    rmse_hi: float
    pearson_r: float
    spearman_r: float


def regression_metrics(
    y_true,
    y_pred,
    n_boot: int = 1000,
    confidence: float = 0.95,
) -> RegressionReport:
    """
    Compute core regression metrics + bootstrap CIs for R²/MAE/RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = _rmse(y_true, y_pred)

    r2_lo, r2_hi = _bootstrap_ci(
        y_true, y_pred, metric_fn=lambda a, b: r2_score(a, b),
        n_boot=n_boot, confidence=confidence
    )
    mae_lo, mae_hi = _bootstrap_ci(
        y_true, y_pred, metric_fn=lambda a, b: mean_absolute_error(a, b),
        n_boot=n_boot, confidence=confidence
    )
    rmse_lo, rmse_hi = _bootstrap_ci(
        y_true, y_pred, metric_fn=lambda a, b: _rmse(a, b),
        n_boot=n_boot, confidence=confidence
    )

    pr, _ = pearsonr(y_true, y_pred) if np.std(y_pred) > 0 else (np.nan, None)
    sr, _ = spearmanr(y_true, y_pred) if np.std(y_pred) > 0 else (np.nan, None)

    return RegressionReport(
        r2=float(r2), r2_lo=r2_lo, r2_hi=r2_hi,
        mae=float(mae), mae_lo=mae_lo, mae_hi=mae_hi,
        rmse=float(rmse_val), rmse_lo=rmse_lo, rmse_hi=rmse_hi,
        pearson_r=float(pr), spearman_r=float(sr),
    )


# ---------------------------------------------------------------------------
# CV aggregation + summary writing
# ---------------------------------------------------------------------------

def aggregate_cv_metrics(rows: list[dict]) -> pd.DataFrame:
    """
    Take a list of metric dicts from folds and return a tidy DataFrame with mean±sd.
    Example input rows (classification):
      {"fold": 1, "accuracy": 0.86, "f1_macro": 0.81, ...}
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.drop(columns=[c for c in df.columns if c.lower() == "fold" or c.endswith("_label")], errors="ignore") \
           .agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    return agg


def write_training_summary(
    records: list[dict],
    out_path: Path,
) -> Path:
    """
    Write a per-antibiotic summary table to TSV. Each record should already be flat
    (e.g., combining model meta + metrics). Returns the written path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(out_path, sep="\t", index=False)
    return out_path


# ---------------------------------------------------------------------------
# Convenience wrappers to convert dataclasses to dicts
# ---------------------------------------------------------------------------

def to_dict(report_dataclass) -> dict:
    """Convert ClassificationReport/RegressionReport to a plain dict."""
    return asdict(report_dataclass)
