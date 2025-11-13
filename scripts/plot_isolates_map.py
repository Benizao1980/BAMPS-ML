#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd

import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear

# optional deps for static export & world shapes
try:
    import geopandas as gpd  # for static PNG/SVG and Natural Earth shapes
except Exception:
    gpd = None

try:
    import country_converter as coco  # robust name→ISO3
except Exception:
    coco = None

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

# ---------- helpers -----------------------------------------------------

LAT_CANDS = ["lat", "latitude", "Lat", "Latitude", "LAT"]
LON_CANDS = ["lon", "long", "longitude", "Lon", "Longitude", "LONG"]
SIR_COLORS = {"S": "#2ca02c", "I": "#ffbf00", "R": "#d62728"}

def _detect_latlon(df: pd.DataFrame) -> tuple[str, str]:
    lat = next((c for c in LAT_CANDS if c in df.columns), None)
    lon = next((c for c in LON_CANDS if c in df.columns), None)
    if not lat or not lon:
        raise SystemExit(f"Could not find lat/lon columns. Looked for {LAT_CANDS} and {LON_CANDS}.")
    return lat, lon

def _read_metadata(path: Path, id_col: str|None, country_col: str|None) -> pd.DataFrame:
    meta = pd.read_csv(path, sep=None, engine="python")
    meta.columns = (meta.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)  # drop BOM if present
                .str.strip())
    if id_col is None:
        id_col = meta.columns[0]
    if id_col not in meta.columns:
        raise SystemExit(f"ID column '{id_col}' not in {path}")
    meta = meta.rename(columns={id_col: "sample"})
    lat, lon = _detect_latlon(meta)
    meta = meta.rename(columns={lat: "lat", lon: "lon"})
    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
    meta = meta.dropna(subset=["lat","lon"])
    if country_col and country_col in meta.columns:
        meta = meta.rename(columns={country_col: "country"})
    else:
        # try to guess
        for c in ["country","Country","COUNTRY","iso3","ISO3","iso2","ISO2","nation"]:
            if c in meta.columns:
                meta = meta.rename(columns={c: "country"})
                break
    return meta

def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = (df.columns.astype(str)
              .str.replace("\ufeff", "", regex=False)
              .str.strip())
    sample = "sample" if "sample" in df.columns else df.columns[0]
    pred   = "prediction" if "prediction" in df.columns else df.columns[-1]
    out = df[[sample, pred]].copy()
    out.columns = ["sample", "pred"]
    return out

def _to_iso3(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if coco is not None:
        converted = coco.CountryConverter().convert(s.tolist(), to="ISO3", not_found=None, normalize=True)
        return pd.Series(converted, index=s.index)
    # fallback: assume already ISO3-ish or country names; uppercase and trim
    return s.str.upper()

def _norm_sir(x: str|float) -> str|None:
    s = str(x).strip().upper()
    if s in {"S","I","R"}: return s
    if s in {"SUSC","SUSCEPTIBLE"}: return "S"
    if s in {"INTERMEDIATE"}: return "I"
    if s in {"RES","RESISTANT"}: return "R"
    return None

# ---------- plotting ----------------------------------------------------

def make_interactive_map(df_pts: pd.DataFrame,
                         value_mode: str|None,
                         title: str,
                         agg_df: pd.DataFrame|None,
                         agg_label: str|None,
                         world_geojson: dict|None,
                         out_html: Path,
                         log10: bool,
                         cluster: bool):
    # center
    center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())] if len(df_pts) else [20,0]
    m = folium.Map(location=center, zoom_start=2, tiles="cartodbpositron")

    # Choropleth layer (if provided)
    if agg_df is not None and world_geojson is not None:
        vmin, vmax = np.nanmin(agg_df["value"]), np.nanmax(agg_df["value"])
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
        cmap.caption = agg_label
        folium.Choropleth(
            geo_data=world_geojson,
            data=agg_df,
            columns=["iso3", "value"],
            key_on="feature.properties.iso_a3",
            fill_color="YlOrRd",
            fill_opacity=0.8,
            line_opacity=0.2,
            nan_fill_opacity=0.05,
            highlight=True,
        ).add_to(m)
        cmap.add_to(m)

    # Point layer
    layer_name = title if title else "isolates"
    fg = folium.FeatureGroup(name=layer_name, show=True)
    m.add_child(fg)
    cluster_layer = MarkerCluster() if cluster else None
    if cluster_layer: fg.add_child(cluster_layer)

    # decide coloring
    to_color = None
    legend_html = None

    if value_mode == "sir":
        def to_color(sir):
            return SIR_COLORS.get(_norm_sir(sir), "#7f7f7f")
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                    background: white; padding: 8px 10px; border: 1px solid #ccc;">
          <b>Category</b><br>
          <span style="color:#2ca02c;">&#11044;</span> S&nbsp;&nbsp;
          <span style="color:#ffbf00;">&#11044;</span> I&nbsp;&nbsp;
          <span style="color:#d62728;">&#11044;</span> R
        </div>"""
    elif value_mode == "mic":
        vals = df_pts["mic"].dropna().values
        if len(vals):
            if log10:
                vals = np.log10(vals[vals>0])
            vmin, vmax = np.nanpercentile(vals, [5,95])
            cmap = linear.Reds_09.scale(vmin, vmax)
            cmap.caption = "Predicted MIC" + (" (log10)" if log10 else "")
            def to_color(v):
                if pd.isna(v): return "#7f7f7f"
                x = math.log10(v) if (log10 and v and v>0) else v
                return cmap(x)
            cmap.add_to(m)
        else:
            def to_color(_): return "#7f7f7f"

    for _, r in df_pts.iterrows():
        color = "#1f77b4"
        label = ""
        if value_mode == "sir":
            label = _norm_sir(r.get("sir"))
            color = to_color(label)
        elif value_mode == "mic":
            mic = r.get("mic")
            label = (f"{mic:g}" if pd.notna(mic) else "NA")
            color = to_color(mic)

        popup = folium.Popup(folium.IFrame(
            html=f"<b>{r['sample']}</b>"
                 + (f"<br>Country: {r.get('country','')}" if 'country' in r else "")
                 + (f"<br>Value: {label}" if label else ""),
            width=220, height=90), max_width=260)

        marker = folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=4, color=color, fill=True, fill_opacity=0.9, weight=0, popup=popup
        )
        (cluster_layer or fg).add_child(marker) if cluster_layer else fg.add_child(marker)

    folium.LayerControl(collapsed=False).add_to(m)
    if value_mode == "sir" and legend_html:
        m.get_root().html.add_child(folium.Element(legend_html))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"[OK] interactive map → {out_html}")

def static_export_png_svg(world_gdf, df_pts, agg_df, agg_label, value_mode, out_png: Path|None, out_svg: Path|None, log10: bool):
    if gpd is None:
        print("[INFO] geopandas not available; skipping static PNG/SVG export.")
        return
    # merge choropleth values onto shapes
    plot_gdf = world_gdf.copy()
    if agg_df is not None:
        plot_gdf = plot_gdf.merge(agg_df, left_on="iso_a3", right_on="iso3", how="left")

    # colormap
    cmap = cm.get_cmap("YlOrRd")
    if agg_df is not None and plot_gdf["value"].notna().any():
        vmin, vmax = np.nanmin(plot_gdf["value"]), np.nanmax(plot_gdf["value"])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for ext, outpath in [("png", out_png), ("svg", out_svg)]:
        if outpath is None: continue
        figw = 8.5
        figh = 4.8
        fig, ax = plt.subplots(figsize=(figw, figh))
        ax.set_title("Isolates & country summary", weight="bold")

        # draw countries
        if norm is None:
            plot_gdf.plot(ax=ax, color="#efefef", edgecolor="#cccccc", linewidth=0.3)
        else:
            plot_gdf.plot(ax=ax, column="value", cmap=cmap, linewidth=0.3, edgecolor="#cccccc", missing_kwds={"color":"#f7f7f7"}, legend=True)
            ax.get_legend().set_title(agg_label)

        # draw points
        if len(df_pts):
            if value_mode == "sir":
                cols = df_pts["sir"].map(SIR_COLORS).fillna("#7f7f7f")
            elif value_mode == "mic":
                v = df_pts["mic"].copy()
                if log10:
                    v = np.where(v>0, np.log10(v), np.nan)
                if v.notna().any():
                    vmin, vmax = np.nanpercentile(v.dropna(), [5,95])
                    normp = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    cols = [cmap(normp(x)) if pd.notna(x) else "#7f7f7f" for x in v]
                else:
                    cols = "#7f7f7f"
            else:
                cols = "#1f77b4"
            ax.scatter(df_pts["lon"], df_pts["lat"], s=8, c=cols, alpha=0.9, linewidths=0, zorder=5)

        ax.set_axis_off()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight", dpi=200 if ext=="png" else None)
        plt.close(fig)
        print(f"[OK] static {ext.upper()} → {outpath}")

# ---------- main --------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot isolate locations with optional S/I/R or MIC colouring and a country-level choropleth.")
    ap.add_argument("--metadata", required=True, type=Path, help="CSV: sample ID, latitude, longitude, country (optional).")
    ap.add_argument("--id-col", default=None, help="Sample ID column in metadata (default: first column).")
    ap.add_argument("--country-col", default=None, help="Country column in metadata (name or code).")
    ap.add_argument("--pred-file", type=Path, default=None, help="preds_*_SIR.tsv or preds_*_mic.tsv (optional).")
    ap.add_argument("--value", choices=["sir","mic"], default=None, help="Force interpretation of pred-file (else auto-detect).")
    ap.add_argument("--choropleth", action="store_true", help="Add a country-level choropleth.")
    ap.add_argument("--metric", choices=["prevalence_R","median_mic","count"], default=None,
                    help="Choropleth metric: if SIR → prevalence_R; if MIC → median_mic; else count.")
    ap.add_argument("--world-geojson", type=Path, default=None,
                    help="Optional path to a world GeoJSON; if absent, uses Natural Earth via geopandas.")
    ap.add_argument("--title", default="Isolates", help="Layer title.")
    ap.add_argument("--out-html", required=True, type=Path, help="Output HTML for interactive map.")
    ap.add_argument("--export-png", type=Path, default=None, help="Also save a static PNG (requires geopandas).")
    ap.add_argument("--export-svg", type=Path, default=None, help="Also save a static SVG (requires geopandas).")
    ap.add_argument("--log10", action="store_true", help="Use log10 scale for MIC colouring.")
    ap.add_argument("--cluster", action="store_true", help="Enable point clustering.")
    args = ap.parse_args()

    meta = _read_metadata(args.metadata, args.id_col, args.country_col)

    # attach predictions if provided
    df = meta.copy()
    value_mode = None
    if args.pred_file is not None:
        pr = _read_preds(args.pred_file)
        df = df.merge(pr, on="sample", how="left")
        # decide value type
        value_mode = (args.value or
                      ("sir" if args.pred_file.name.lower().endswith("_sir.tsv") or "sir" in args.pred_file.name.lower()
                       else "mic" if args.pred_file.name.lower().endswith("_mic.tsv") or "mic" in args.pred_file.name.lower()
                       else ("sir" if df["pred"].astype(str).str.upper().isin(["S","I","R"]).mean()>0.8 else "mic")))
        if value_mode == "sir":
            df["sir"] = df["pred"].map(_norm_sir)
        else:
            df["mic"] = pd.to_numeric(df["pred"], errors="coerce")

    # choropleth aggregation
    agg_df = None
    agg_label = None
    world_geojson = None
    world_gdf = None

    if args.choropleth:
        if "country" not in df.columns:
            raise SystemExit("Choropleth requested but no country column in metadata. Use --country-col or add 'country' column.")
        df["iso3"] = _to_iso3(df["country"])
        # pick metric default
        metric = args.metric
        if metric is None:
            metric = "prevalence_R" if value_mode == "sir" else ("median_mic" if value_mode == "mic" else "count")

        grp = df.dropna(subset=["iso3"]).groupby("iso3", as_index=False)
        if metric == "prevalence_R":
            if value_mode != "sir":
                raise SystemExit("--metric prevalence_R requires SIR predictions.")
            agg_df = grp.apply(lambda g: pd.Series({"value": (g["sir"]=="R").mean(), "n": len(g)}))
            agg_label = "% Resistant (predicted)"
        elif metric == "median_mic":
            if value_mode != "mic":
                raise SystemExit("--metric median_mic requires MIC predictions.")
            agg_df = grp["mic"].median().rename(columns={"mic":"value"})
            agg_label = "Median predicted MIC"
        elif metric == "count":
            agg_df = grp.size().rename(columns={"size":"value"})
            agg_label = "Isolate count"

        # world shapes
        if args.world_geojson and Path(args.world_geojson).exists():
            world_geojson = json.loads(Path(args.world_geojson).read_text())
            world_gdf = None
        else:
            if gpd is None:
                raise SystemExit("geopandas is required to auto-load world shapes. Provide --world-geojson to avoid geopandas.")
            world_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world_gdf = world_gdf[world_gdf["iso_a3"] != "-99"]
            world_geojson = world_gdf.__geo_interface__

    # interactive map
    make_interactive_map(df_pts=df,
                         value_mode=value_mode,
                         title=args.title,
                         agg_df=agg_df,
                         agg_label=agg_label,
                         world_geojson=world_geojson,
                         out_html=args.out_html,
                         log10=args.log10,
                         cluster=args.cluster)

    # static exports (PNG/SVG)
    if args.export_png or args.export_svg:
        if world_gdf is None:
            # if user supplied geojson and we want static export, we still need geopandas to read it
            if gpd is None:
                print("[INFO] geopandas not available; cannot produce PNG/SVG without it.")
                return
            world_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world_gdf = world_gdf[world_gdf["iso_a3"] != "-99"]
        static_export_png_svg(world_gdf, df_pts=df, agg_df=agg_df, agg_label=agg_label,
                              value_mode=value_mode, out_png=args.export_png,
                              out_svg=args.export_svg, log10=args.log10)

if __name__ == "__main__":
    main()
