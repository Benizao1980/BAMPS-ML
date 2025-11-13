#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps as mcm

from bamps_ml.utils import load_config, resolve_breakpoints

PALETTE = {
    "ciprofloxacin": "#1f77b4",
    "imipenem":      "#ff7f0e",
    "meropenem":     "#2ca02c",
    "colistin":      "#d62728",
}

def _load_preds(pred_files):
    rows = []
    for f in pred_files:
        df = pd.read_csv(f, sep=None, engine="python")
        m = re.search(r"preds[_-](.+?)[_-]mic\.tsv$", Path(f).name, re.I)
        ab = m.group(1) if m else None
        samp = "sample" if "sample" in df.columns else df.columns[0]
        pred = "prediction" if "prediction" in df.columns else df.columns[-1]
        for _, r in df.iterrows():
            rows.append({"antibiotic": str(ab).lower(), "sample": r[samp], "pred_mic": float(r[pred])})
    out = pd.DataFrame(rows)
    if out.empty or out["antibiotic"].isna().any():
        raise SystemExit("Could not infer antibiotic names from filenames; expected 'preds_<ab>_mic.tsv'.")
    return out

def _load_acc(summary_path):
    acc = {}
    if not summary_path:
        return acc
    t = pd.read_csv(summary_path, sep="\t")
    if "task" in t.columns:
        t = t[t["task"].str.contains("class", case=False, na=False)]
    for _, r in t.iterrows():
        if pd.notna(r.get("antibiotic")) and pd.notna(r.get("accuracy")):
            acc[str(r["antibiotic"]).lower()] = float(r["accuracy"])
    return acc

def _load_truth(mic_file, antibiotics, id_col="", strip_prefix="", strip_suffix="", to_lower=False):
    mic = pd.read_csv(mic_file, sep=None, engine="python")
    # choose id column
    if id_col and id_col in mic.columns:
        mic = mic.rename(columns={id_col: "sample"})
    else:
        mic = mic.rename(columns={mic.columns[0]: "sample"})
    # normalize IDs
    s = mic["sample"].astype(str)
    if strip_prefix:
        s = s.str.replace(f"^{re.escape(strip_prefix)}", "", regex=True)
    if strip_suffix:
        s = s.str.replace(f"{re.escape(strip_suffix)}$", "", regex=True)
    if to_lower:
        s = s.str.lower()
    mic["sample"] = s
    keep = ["sample"] + [c for c in antibiotics if c in mic.columns]
    return mic[keep].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-glob", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--order", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log10", action="store_true")
    ap.add_argument("--truth", default="", help="Optional MIC CSV to overlay observed MICs")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--truth-id-col", default="", help="Column in truth CSV that contains sample IDs (default: first column)")
    ap.add_argument("--truth-strip-prefix", default="", help="Strip this prefix from truth IDs before matching")
    ap.add_argument("--truth-strip-suffix", default="", help="Strip this suffix from truth IDs before matching")
    ap.add_argument("--truth-lower", action="store_true", help="Lowercase truth IDs before matching")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pred_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.pred_glob}")
    order = [o.lower() for o in args.order]

    preds = _load_preds(files)
    acc = _load_acc(args.summary)
    truth = None
    bp = {}
    try:
        cfg = load_config(args.config)
        bp = resolve_breakpoints(cfg.get("breakpoint_standard", "EUCAST"), cfg.get("mic_thresholds"))
    except Exception:
        pass
    if args.truth:
        truth = _load_truth(
            args.truth, order,
            id_col=args.truth_id_col,
            strip_prefix=args.truth_strip_prefix,
            strip_suffix=args.truth_strip_suffix,
            to_lower=args.truth_lower,
        )

    # --- compute per-antibiotic R²/MAE (if truth provided) BEFORE plotting text
    stats_txt = {}
    if truth is not None:
        from sklearn.metrics import r2_score, mean_absolute_error
        rows = []
        for ab in order:
            if ab not in truth.columns:
                continue
            d = preds[preds["antibiotic"] == ab].merge(
                truth[["sample", ab]].rename(columns={ab: "true"}),
                on="sample", how="left"
            ).dropna(subset=["true"])
            if len(d):
                if args.log10:
                    r2 = r2_score(np.log10(d["true"]), np.log10(d["pred_mic"]))
                    mae = mean_absolute_error(np.log10(d["true"]), np.log10(d["pred_mic"]))
                else:
                    r2 = r2_score(d["true"], d["pred_mic"])
                    mae = mean_absolute_error(d["true"], d["pred_mic"])
                rows.append({"antibiotic": ab, "n": int(len(d)), "R2": float(r2), "MAE": float(mae)})
                stats_txt[ab] = f"R²={r2:.2f} | MAE={mae:.2f}"
        if rows:
            t = pd.DataFrame(rows)
            print("\nPer-antibiotic regression fit:")
            print(t.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))

    # --- plotting ---
    rng = np.random.default_rng(42)
    # Wider figure when there are many antibiotics
    figw = max(8.0, 1.4 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(figw, 4.8))

    all_y = []
    norm_per_ab = {}  # per-antibiotic normaliser for white→red
    for ab in order:
        y = preds.loc[preds["antibiotic"] == ab, "pred_mic"].astype(float)
        if len(y):
            all_y.extend(y.values)
            # per-drug normaliser (robust to outliers)
            ylo, yhi = np.nanpercentile(y, [5, 95])
            if args.log10:
                ylo, yhi = max(1e-4, ylo), max(ylo * 1.01, yhi)
            norm_per_ab[ab] = Normalize(vmin=ylo, vmax=yhi)
        else:
            norm_per_ab[ab] = Normalize(vmin=0, vmax=1)

    reds = mcm.get_cmap("Reds")

    # Collect unique breakpoint y-positions so we can label them once at the right margin
    seen_bp = set()  # set of (label, rounded_y)

    for i, ab in enumerate(order, start=1):
        d = preds[preds["antibiotic"] == ab]
        if d.empty:
            # faint “no predictions” note
            ax.text(i, 0.8 if not args.log10 else 10 ** (-0.1), "no predictions",
                    ha="center", va="center", fontsize=9, alpha=0.5, rotation=90,
                    transform=ax.get_xaxis_transform())
            continue

        y = d["pred_mic"].astype(float).values
        x = i + rng.uniform(-0.18, 0.18, size=len(y))
        colors = reds(norm_per_ab[ab](y))
        ax.scatter(x, y, s=20, alpha=0.8, linewidths=0, color=colors)

        # optional observed overlay
        if truth is not None and ab in truth.columns:
            merged = d.merge(
                truth[["sample", ab]].rename(columns={ab: "true_mic"}),
                on="sample", how="left",
            )
            m = merged["true_mic"].notna()
            if m.any():
                xt = i + rng.uniform(-0.28, -0.08, size=m.sum())
                ax.scatter(
                    xt,
                    merged.loc[m, "true_mic"].astype(float),
                    s=22,
                    facecolors="none",
                    edgecolors="#444444",
                    linewidths=1.2,
                )

        # breakpoints (draw per-drug lines; label once at right margin to reduce clutter)
        if ab in bp:
            thr = bp[ab]
            for key in ("S", "I", "R"):
                if key in thr:
                    yb = float(thr[key])
                    ax.axhline(yb, ls="--", lw=0.9, color="#6b8e23", alpha=0.6)
                    # label once per unique y across drugs
                    ident = (key, round(np.log10(yb), 3) if args.log10 else round(yb, 3))
                    if ident not in seen_bp:
                        seen_bp.add(ident)
                        # put label just outside the right edge in axes coords
                        ax.text(1.005, yb, key, va="center", ha="left",
                                fontsize=9, color="#6b8e23",
                                transform=ax.get_yaxis_transform())

    # axes + under-axis stats
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_xticks(range(1, len(order) + 1))
    # Nice labels: abbreviate TMP-SMX and wrap hyphens with a newline
    def _pretty(ab):
        m = {
            "trimethoprim-sulfamethoxazole": "TMP-SMX",
        }
        lab = m.get(ab, ab.capitalize())
        return lab.replace("-", "\n")  # wrap long hyphenated names
    ax.set_xticklabels([_pretty(o) for o in order], rotation=25, ha="right")
    if stats_txt:
        for i, ab in enumerate(order, start=1):
            txt = stats_txt.get(ab)
            if txt:
                ax.text(i, -0.16, txt, ha="center", va="top",
                        fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Predicted MIC (mg/L)")
    ax.set_yscale("log" if args.log10 else "linear")
    if all_y:
        ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
        if np.isfinite(ymin) and ymin > 0:
            ax.set_ylim(ymin * 0.8, ymax * 1.3)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_title("Predicted MIC per antibiotic", weight="bold")

    # save
    fig.tight_layout(rect=[0.04, 0.16, 0.985, 0.93])  # a hair more right margin for S/I/R tags
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print(f"[OK] saved → {out.with_suffix('.png')} and .svg")

if __name__ == "__main__":
    main()