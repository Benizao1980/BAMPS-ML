#!/usr/bin/env python3
from __future__ import annotations

import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from bamps_ml.utils import load_config, resolve_breakpoints, mic_to_category

LABELS = ["S", "I", "R"]

def _norm_labels(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    # Map common variants
    s = s.replace({"SUSC": "S", "SUSCEPTIBLE": "S", "INTERMEDIATE":"I",
                   "RES": "R", "RESISTANT": "R"})
    s = s.where(s.isin(LABELS), np.nan)
    return s

def _load_preds(pred_glob: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(pred_glob)):
        df = pd.read_csv(f, sep=None, engine="python")
        ab = re.search(r"preds[_-](.+?)_SIR\.tsv$", Path(f).name, re.I)
        if not ab:
            continue
        ab = ab.group(1).lower()
        sample_col = "sample" if "sample" in df.columns else df.columns[0]
        pred_col   = "prediction" if "prediction" in df.columns else df.columns[-1]
        t = df[[sample_col, pred_col]].copy()
        t.columns = ["sample", "pred"]
        t["antibiotic"] = ab
        rows.append(t)
    if not rows:
        raise SystemExit(f"No prediction files matched: {pred_glob}")
    out = pd.concat(rows, ignore_index=True)
    out["pred"] = _norm_labels(out["pred"])
    return out

def _load_truth(truth_csv: Path, antibiotics: list[str], id_col: str, config_yaml: Path) -> pd.DataFrame:
    mic = pd.read_csv(truth_csv, sep=None, engine="python")
    if id_col and id_col in mic.columns:
        mic = mic.rename(columns={id_col: "sample"})
    else:
        mic = mic.rename(columns={mic.columns[0]: "sample"})
    mic.columns = [c.strip() for c in mic.columns]

    # try to detect if truth is categorical S/I/R already for the first requested ab
    is_categorical = False
    if antibiotics:
        ab0 = antibiotics[0]
        if ab0 in mic.columns:
            s0 = mic[ab0]
            frac_SIR = s0.astype(str).str.upper().isin(LABELS).mean()
            is_categorical = frac_SIR > 0.8

    # if categorical, just normalize; else interpret as MIC and map to S/I/R using breakpoints
    if is_categorical:
        for ab in antibiotics:
            if ab in mic.columns:
                mic[ab] = _norm_labels(mic[ab])
    else:
        cfg = load_config(str(config_yaml))
        bps = resolve_breakpoints(cfg.get("breakpoint_standard", "EUCAST"),
                                  cfg.get("mic_thresholds"))
        for ab in antibiotics:
            if ab not in mic.columns:  # skip if not present
                continue
            series = pd.to_numeric(mic[ab], errors="coerce")
            mic[ab] = series.map(lambda v: mic_to_category(ab, v, bps) if pd.notna(v) else np.nan)

    keep = ["sample"] + [ab for ab in antibiotics if ab in mic.columns]
    return mic[keep].copy()

def _plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(3.2, 3))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title, weight="bold")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-glob", required=True, help="glob for preds_*_SIR.tsv")
    ap.add_argument("--truth", required=True, type=Path, help="Truth CSV (S/I/R per drug or MICs)")
    ap.add_argument("--id-col", default="", help="ID column in truth (default: first col)")
    ap.add_argument("--order", nargs="+", required=True, help="Antibiotics to evaluate (names as in preds filenames)")
    ap.add_argument("--outdir", required=True, type=Path, help="Directory for PNGs and summary TSV")
    ap.add_argument("--config", default="config/config.yaml", type=Path, help="YAML with breakpoints if truth is MICs")
    args = ap.parse_args()

    order = [a.lower() for a in args.order]
    preds = _load_preds(args.pred_glob)
    truth = _load_truth(args.truth, order, args.id_col, args.config)

    rows = []
    for ab in order:
        p = preds[preds["antibiotic"] == ab][["sample", "pred"]].copy()
        if p.empty or ab not in truth.columns:
            continue
        merged = p.merge(truth[["sample", ab]].rename(columns={ab: "true"}), on="sample", how="inner")
        merged["true"] = _norm_labels(merged["true"])
        merged = merged.dropna(subset=["true", "pred"])
        if merged.empty:
            continue

        y_true = merged["true"].values
        y_pred = merged["pred"].values

        # fix label order to S,I,R for matrix shape
        labels_present = [lab for lab in LABELS if lab in set(y_true) | set(y_pred)]
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)

        # metrics (macro over present labels to avoid zero-division)
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels_present, zero_division=0)
        macro_f1 = float(np.mean(f1))

        rows.append({
            "antibiotic": ab,
            "n": int(len(merged)),
            "accuracy": float(acc),
            "precision_R": float(pr[labels_present.index("R")] if "R" in labels_present else np.nan),
            "recall_R": float(rc[labels_present.index("R")] if "R" in labels_present else np.nan),
            "f1_R": float(f1[labels_present.index("R")] if "R" in labels_present else np.nan),
            "macro_f1": macro_f1,
        })

        # plot
        out_png = args.outdir / f"confusion_{ab}.png"
        _plot_confusion(cm, labels_present, f"{ab.capitalize()} (n={len(merged)})", out_png)
        print(f"[OK] {ab}: n={len(merged)} acc={acc:.3f} macroF1={macro_f1:.3f} → {out_png}")

    # write summary
    if rows:
        df = pd.DataFrame(rows)
        args.outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.outdir / "confusion_summary.tsv", sep="\t", index=False)
        print(f"[OK] summary → {args.outdir / 'confusion_summary.tsv'}")
    else:
        print("[INFO] No overlaps or no valid truth columns; nothing written.")

if __name__ == "__main__":
    main()
