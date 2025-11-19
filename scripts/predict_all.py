#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


def find_models(model_dir: Path, task: str) -> List[Tuple[str, Path]]:
    suffix = f"_{task}.pkl"
    models: List[Tuple[str, Path]] = []
    for p in sorted(model_dir.glob(f"*{suffix}")):
        ab = p.name[: -len(suffix)]
        if ab:
            models.append((ab, p))
    return models


# ---------------------- tidy helpers ------------------------------------

_PRED_LABEL_KEYS = ("prediction", "pred", "pred_label", "label")
_PROB_SETS = (
    ("prob_s", "prob_i", "prob_r"),
    ("s", "i", "r"),
    ("ps", "pi", "pr"),
)
_MIC_KEYS = ("pred_mic", "mic", "value")
_LOG2MIC_KEYS = ("log2mic", "pred_log2mic")

def _norm_sir(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in ("S", "I", "R"):
        return s
    if s in ("SUSC", "SUSCEPTIBLE"):
        return "S"
    if s in ("INTERMEDIATE",):
        return "I"
    if s in ("RES", "RESISTANT"):
        return "R"
    return ""


def _read_tsv_flex(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    # de-BOM & strip
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    # drop accidental index columns
    for junk in ("Unnamed: 0", "index", "#", "Index"):
        if junk in df.columns and df[junk].is_monotonic_increasing:
            try:
                df = df.drop(columns=[junk])
            except Exception:
                pass
    return df


def _tidy_predictions(
    in_path: Path,
    out_path: Path,
    antibiotic: str,
    task: str,
    to_mic: bool,
    geo_csv: Optional[Path],
    geo_id_col: Optional[str],
    geo_country_col: Optional[str],
    geo_lat_col: Optional[str],
    geo_lon_col: Optional[str],
) -> None:
    """
    Standardise a prediction TSV to the contract:
    sample, antibiotic, prediction, pred_mic, prob_S, prob_I, prob_R, model, version, breakpoints
    """
    df = _read_tsv_flex(in_path)

    # ----- identify sample column -----
    lower_map = {c.lower(): c for c in df.columns}
    if "sample" in lower_map:
        sample_col = lower_map["sample"]
    else:
        # assume first column is the id if not named
        sample_col = df.columns[0]
    df = df.rename(columns={sample_col: "sample"})
    df["sample"] = df["sample"].astype(str).str.strip()

    # ----- figure out prediction / probs / mic -----
    pred = pd.Series([""] * len(df), index=df.index, dtype="string")
    prob_S = prob_I = prob_R = pd.Series([pd.NA] * len(df))
    pred_mic = pd.Series([pd.NA] * len(df))
    pred_log2mic = pd.Series([pd.NA] * len(df))

    # explicit label?
    label_col = next((lower_map[k] for k in _PRED_LABEL_KEYS if k in lower_map), None)
    if label_col is not None:
        pred = df[label_col].astype("string").map(_norm_sir)

    # probs present?
    found_probs = None
    for a, b, c in _PROB_SETS:
        if a in lower_map and b in lower_map and c in lower_map:
            found_probs = (lower_map[a], lower_map[b], lower_map[c])
            break
    if found_probs:
        s_col, i_col, r_col = found_probs
        prob_S = pd.to_numeric(df[s_col], errors="coerce")
        prob_I = pd.to_numeric(df[i_col], errors="coerce")
        prob_R = pd.to_numeric(df[r_col], errors="coerce")
        # if no explicit label, derive from max prob
        if label_col is None:
            idx = pd.concat(
                {"S": prob_S, "I": prob_I, "R": prob_R}, axis=1
            ).idxmax(axis=1)
            pred = idx.astype("string")

    # MIC columns (regression)
    mic_col = next((lower_map[k] for k in _MIC_KEYS if k in lower_map), None)
    log2_col = next((lower_map[k] for k in _LOG2MIC_KEYS if k in lower_map), None)

    if mic_col is not None:
        pred_mic = pd.to_numeric(df[mic_col], errors="coerce")
    elif log2_col is not None:
        pred_log2mic = pd.to_numeric(df[log2_col], errors="coerce")
        if to_mic:
            pred_mic = (2.0 ** pred_log2mic).round(3)

    # ----- build tidy frame -----
    out_cols = {
        "sample": df["sample"],
        "antibiotic": antibiotic,
        "prediction": pred.map(_norm_sir),
        "pred_mic": pred_mic,
        "prob_S": prob_S,
        "prob_I": prob_I,
        "prob_R": prob_R,
    }
    # optional provenance if present
    for k in ("model", "version", "breakpoints"):
        if k in lower_map:
            out_cols[k] = df[lower_map[k]]
    # keep log2 if we didn’t convert
    if not to_mic and log2_col is not None:
        out_cols["pred_log2mic"] = pred_log2mic

    tidy = pd.DataFrame(out_cols)

    # ----- optional geo merge -----
    if geo_csv is not None and geo_csv.exists():
        g = _read_tsv_flex(geo_csv)
        # choose id column
        g_lower = {c.lower(): c for c in g.columns}
        gid = geo_id_col or next(iter(g.columns))
        if gid not in g.columns and gid.lower() in g_lower:
            gid = g_lower[gid.lower()]
        g = g.rename(
            columns={
                gid: "sample",
                (geo_country_col or "Country"): "Country",
                (geo_lat_col or "lat"): "lat",
                (geo_lon_col or "lon"): "lon",
            }
        )
        for numc in ("lat", "lon"):
            if numc in g.columns:
                g[numc] = pd.to_numeric(g[numc], errors="coerce")
        g["sample"] = g["sample"].astype(str).str.strip()
        keep = [c for c in ("sample", "Country", "lat", "lon") if c in g.columns]
        tidy = tidy.merge(g[keep], on="sample", how="left")

    # ----- write back (no BOM, no index) -----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_path, sep="\t", index=False)
    # quick sanity log
    if "prediction" in tidy.columns:
        counts = tidy["prediction"].value_counts(dropna=False).to_dict()
        print(f"[TIDY] {out_path.name}: n={len(tidy)}  SIR={counts}")


# ---------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run predictions for all available models (classification and/or regression), and tidy outputs to a uniform schema. Optionally plot a MIC panel and S/I/R confusion."
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
                    help="For regression models, DO NOT convert log2(MIC) back to raw MIC. "
                         "Tidy step will emit pred_log2mic and leave pred_mic blank.")

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

    # OPTIONAL: merge geo so downstream map can read a single tidy file
    ap.add_argument("--geo-csv", type=Path, default=None,
                    help="Optional metadata CSV to merge (to append Country/lat/lon).")
    ap.add_argument("--geo-id-col", default=None,
                    help="ID column name in --geo-csv (default: first column).")
    ap.add_argument("--geo-country-col", default=None,
                    help="Country column in --geo-csv (default: tries Country/iso3/ISO3/etc.).")
    ap.add_argument("--geo-lat-col", default=None,
                    help="Latitude column in --geo-csv (default: lat/Lat/Latitude).")
    ap.add_argument("--geo-lon-col", default=None,
                    help="Longitude column in --geo-csv (default: lon/Long/Longitude).")

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

    def run_one(ab: str, task: str):
        nonlocal ran, failures
        raw_out = args.outdir / (
            f"preds_{ab}_SIR.tsv" if task == "classification"
            else f"preds_{ab}_{'log2mic' if args.no_to_mic else 'mic'}.tsv"
        )

        cmd = [
            sys.executable, str(predict_py),
            "--feature-table", str(args.feature_table),
            "--model-dir", str(args.model_dir),
            "--antibiotic", ab,
            "--task", task,
            "--output", str(raw_out),
        ]
        if task == "regression" and not args.no_to_mic:
            cmd.append("--to-mic")

        print(f"\n[ RUN ] {ab:<14} task={task:<14} → {raw_out.name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Prediction failed for {ab} ({task}). Exit code {e.returncode}")
            failures += 1
            return

        # Tidy in place (write back to same file with standard columns)
        try:
            _tidy_predictions(
                in_path=raw_out,
                out_path=raw_out,
                antibiotic=ab,
                task=task,
                to_mic=(task == "regression" and not args.no_to_mic),
                geo_csv=args.geo_csv,
                geo_id_col=args.geo_id_col,
                geo_country_col=args.geo_country_col,
                geo_lat_col=args.geo_lat_col,
                geo_lon_col=args.geo_lon_col,
            )
            ran += 1
        except Exception as e:
            print(f"[WARN] Tidy failed for {ab} ({task}): {e}")
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

    print(f"\n[DONE] Successful predictions (tidy): {ran}  |  failures: {failures}")

    # ----------------- Optional post-steps -----------------

    # confusion matrices
    if args.cm_outdir:
        if not args.cm_truth:
            sys.exit("[ERROR] --cm-outdir requires --cm-truth (CSV with S/I/R or MICs).")
        if args.cm_order:
            cm_order = args.cm_order
        else:
            cm_order = sorted([p.name.split("_")[1] for p in args.outdir.glob("preds_*_SIR.tsv")])
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

    # MIC panel
    if args.panel_out:
        pred_glob = str(args.outdir / "preds_*_mic.tsv")
        if args.panel_order:
            order = args.panel_order
        else:
            order = sorted([p.name.split("_")[1] for p in args.outdir.glob("preds_*_mic.tsv")])
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
