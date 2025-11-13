#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def find_models(model_dir: Path, task: str) -> List[Tuple[str, Path]]:
    suffix = f"_{task}.pkl"
    models = []
    for p in sorted(model_dir.glob(f"*{suffix}")):
        ab = p.name[: -len(suffix)]
        if ab:
            models.append((ab, p))
    return models


def main():
    ap = argparse.ArgumentParser(
        description="Run predictions for all available models (classification and/or regression), and optionally plot a MIC panel."
    )
    # prediction io
    ap.add_argument("--feature-table", required=True, type=Path,
                    help="AMRFinder presence/absence TSV used for prediction.")
    ap.add_argument("--model-dir", required=True, type=Path,
                    help="Directory containing <antibiotic>_<task>.pkl files.")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Where to write prediction TSVs.")

    # what to run
    ap.add_argument("--tasks", nargs="+", choices=["classification", "regression"],
                    default=["classification", "regression"],
                    help="Which tasks to run.")
    ap.add_argument("--antibiotics", nargs="*", default=None,
                    help="Optional whitelist of antibiotics to run (names match model filenames).")
    ap.add_argument("--no-to-mic", action="store_true",
                    help="For regression models, DO NOT convert log2(MIC) back to raw MIC.")

    # script paths
    ap.add_argument("--predict-script", type=Path, default=None,
                    help="Path to predict.py (default: scripts/predict.py next to this file).")
    ap.add_argument("--plot-script", type=Path, default=None,
                    help="Path to plot_predicted_mic_panel.py (default: scripts/plot_predicted_mic_panel.py next to this file).")

    # optional: immediate panel plotting
    ap.add_argument("--panel-out", type=Path, default=None,
                    help="If set, render a MIC panel after regression predictions. Provide a base path (without extension).")
    ap.add_argument("--panel-order", nargs="+", default=None,
                    help="Order of antibiotics along x-axis (e.g., ciprofloxacin imipenem meropenem colistin).")
    ap.add_argument("--panel-summary", type=Path, default=None,
                    help="training_summary.tsv for accuracy badges (optional).")
    ap.add_argument("--panel-truth", type=Path, default=None,
                    help="Path to truth MIC CSV to show overlay & compute R²/MAE (optional).")
    ap.add_argument("--panel-config", type=Path, default=Path("config/config.yaml"),
                    help="YAML config (for breakpoints), default: config/config.yaml")
    ap.add_argument("--panel-log10", action="store_true",
                    help="Plot y-axis on log scale.")

    # truth ID normalization passthroughs (only used if --panel-truth is given and your plotter supports them)
    ap.add_argument("--truth-id-col", default="",
                    help="Column in truth CSV holding sample IDs (plot script option).")
    ap.add_argument("--truth-strip-prefix", default="",
                    help="Strip this prefix from truth IDs (plot script option).")
    ap.add_argument("--truth-strip-suffix", default="",
                    help="Strip this suffix from truth IDs (plot script option).")
    ap.add_argument("--truth-lower", action="store_true",
                    help="Lowercase truth IDs before matching (plot script option).")
    
    # optional: S/I/R confusion matrices after classification
    ap.add_argument("--cm-outdir", type=Path, default=None,
                    help="If set, render S/I/R confusion matrices after classification.")
    ap.add_argument("--cm-truth", type=Path, default=None,
                    help="Truth CSV (S/I/R per drug or MICs). Required if --cm-outdir is set.")
    ap.add_argument("--cm-id-col", default="",
                    help="ID column in truth CSV (default: first col).")
    ap.add_argument("--cm-order", nargs="+", default=None,
                    help="Antibiotics to include in confusion evaluation (default: from model filenames).")
    ap.add_argument("--cm-config", type=Path, default=Path('config/config.yaml'),
                    help="YAML config for MIC→SIR mapping when truth has MICs.")

    args = ap.parse_args()

    # resolve scripts
    scripts_dir = Path(__file__).resolve().parent
    predict_py = args.predict_script or (scripts_dir / "predict.py")
    plot_py = args.plot_script or (scripts_dir / "plot_predicted_mic_panel.py")

    if not predict_py.exists():
        sys.exit(f"[ERROR] Could not find predict.py at: {predict_py}")
    if args.panel_out and not plot_py.exists():
        sys.exit(f"[ERROR] --panel-out was given, but plot script not found at: {plot_py}")

    # new: confusion script
    cm_py = scripts_dir / "plot_SIR_confusion.py"
    if args.cm_outdir and not cm_py.exists():
        sys.exit(f"[ERROR] --cm-outdir was given, but confusion script not found at: {cm_py}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    ran = 0
    failures = 0
    ran_regression_ok = False  # track if we produced any *_mic.tsv

    def run_one(ab: str, task: str):
        nonlocal ran, failures, ran_regression_ok
        out = args.outdir / (f"preds_{ab}_SIR.tsv" if task == "classification"
                             else f"preds_{ab}_{'log2mic' if args.no_to_mic else 'mic'}.tsv")
        cmd = [
            sys.executable, str(predict_py),
            "--feature-table", str(args.feature_table),
            "--model-dir", str(args.model_dir),
            "--antibiotic", ab,
            "--task", task,
            "--output", str(out),
        ]
        if task == "regression" and not args.no_to_mic:
            cmd.append("--to-mic")

        print(f"\n[RUN] {ab}  task={task}  → {out}")
        try:
            subprocess.run(cmd, check=True)
            ran += 1
            if task == "regression" and not args.no_to_mic:
                ran_regression_ok = True
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Prediction failed for {ab} ({task}). Exit code {e.returncode}")
            failures += 1

    # gather models and run predictions
    for task in args.tasks:
        models = find_models(args.model_dir, task)
        if not models:
            print(f"[INFO] No {task} models found in {args.model_dir}")
            continue

        to_run = models
        if args.antibiotics:
            wanted = {a.lower() for a in args.antibiotics}
            to_run = [(ab, p) for ab, p in models if ab.lower() in wanted]
            missing = wanted - {ab.lower() for ab, _ in models}
            if missing:
                print(f"[INFO] Skipping {task} for missing models: {', '.join(sorted(missing))}")

        for ab, _path in to_run:
            run_one(ab, task)

    print(f"\n[DONE] Successful predictions: {ran}  |  failures: {failures}")
    
    # Optionally plot S/I/R confusion matrices
    if args.cm_outdir:
        if not args.cm_truth:
            sys.exit("[ERROR] --cm-outdir requires --cm-truth (CSV with S/I/R or MICs).")
        # derive order if not provided
        if args.cm_order:
            cm_order = args.cm_order
        else:
            cm_order = sorted([p.name.split("_")[1]  # preds_<ab>_SIR.tsv
                               for p in args.outdir.glob("preds_*_SIR.tsv")])
        if not cm_order:
            print("[INFO] No SIR prediction files to evaluate; skipping confusion matrices.")
        else:
            cmd = [
                sys.executable, str(cm_py),
                "--pred-glob", str(args.outdir / "preds_*_SIR.tsv"),
                "--truth", str(args.cm_truth),
                "--outdir", str(args.cm_outdir),
                "--order", *cm_order,
                "--config", str(args.cm_config),
            ]
            if args.cm_id_col:
                cmd += ["--id-col", args.cm_id_col]
            print(f"\n[PLOT] S/I/R confusion → {args.cm_outdir}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Confusion plotting failed (exit {e.returncode}). Command was:\n  {' '.join(cmd)}")

    # optionally plot panel right away (only makes sense for regression→MIC)
    if args.panel_out:
        # Require some mic files
        pred_glob = str(args.outdir / "preds*_mic.tsv")
        # build order
        if args.panel_order:
            order = args.panel_order
        else:
            # derive order from files present (alphabetical)
            order = sorted([p.name.split("_")[1]  # preds_<ab>_mic.tsv
                            for p in args.outdir.glob("preds_*_mic.tsv")])
        if not order:
            print("[INFO] No MIC prediction files to plot; skipping panel.")
            sys.exit(0)

        cmd = [
            sys.executable, str(plot_py),
            "--pred-glob", pred_glob,
            "--order", *order,
            "--out", str(args.panel_out),
            "--config", str(args.panel_config),
        ]
        if args.panel_summary:
            cmd += ["--summary", str(args.panel_summary)]
        if args.panel_log10:
            cmd += ["--log10"]
        if args.panel_truth:
            cmd += ["--truth", str(args.panel_truth)]
            # pass-through normalization knobs (plotter will ignore if unsupported)
            if args.truth_id_col:
                cmd += ["--truth-id-col", args.truth_id_col]
            if args.truth_strip_prefix:
                cmd += ["--truth-strip-prefix", args.truth_strip_prefix]
            if args.truth_strip_suffix:
                cmd += ["--truth-strip-suffix", args.truth_strip_suffix]
            if args.truth_lower:
                cmd += ["--truth-lower"]

        print(f"\n[PLOT] MIC panel → {args.panel_out}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Panel plotting failed (exit {e.returncode}). Command was:\n  {' '.join(cmd)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
