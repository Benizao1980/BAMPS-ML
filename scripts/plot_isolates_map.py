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

def _to_iso3(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if coco is not None:
        cc = coco.CountryConverter()
        out = cc.convert(s.tolist(), to="ISO3", not_found=None, normalize=True)
        iso = pd.Series(out, index=s.index)
        miss = iso.isna().mean()
        if miss > 0:
            print(f"[WARN] ISO3 conversion failed for {miss:.1%} of rows. "
                  f"Examples: {s[iso.isna()].head(5).tolist()}")
        return iso
    return s.str.upper()

def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")

    # clean column names, drop index-y columns
    df.columns = (df.columns.astype(str)
                  .str.replace("\ufeff","", regex=False)
                  .str.strip())
    keep = [c for c in df.columns if not c.lower().startswith("unnamed")]
    if len(keep) != len(df.columns):
        df = df[keep]

    # pick sample/id column robustly
    cl = {c.lower(): c for c in df.columns}
    id_candidates = ["sample", "id", "isolate", "isolate_id", "sample_id", "name"]
    sample_col = next((cl[c] for c in id_candidates if c in cl), None)
    if sample_col is None:
        # last resort: first non-unnamed column
        sample_col = df.columns[0]

    # find label/pred column
    label_candidates = ["sir", "prediction", "pred", "pred_label", "label", "call"]
    label_col = next((cl[c] for c in label_candidates if c in cl), None)

    if label_col is not None:
        out = df[[sample_col, label_col]].copy()
        out.columns = ["sample", "pred"]
        out["sample"] = out["sample"].astype(str).str.strip()
        out["pred"] = out["pred"].astype(str).str.strip()
        return out

    # probabilities S/I/R → argmax
    prob_sets = [
        ("prob_s", "prob_i", "prob_r"),
        ("s", "i", "r"),
        ("ps", "pi", "pr"),
    ]
    for a, b, c in prob_sets:
        if a in cl and b in cl and c in cl:
            sub = df[[sample_col, cl[a], cl[b], cl[c]]].copy()
            sub.columns = ["sample", "S", "I", "R"]
            sub["pred"] = sub[["S","I","R"]].astype(float).idxmax(axis=1)
            return sub[["sample","pred"]]

    # MIC predictions
    mic_candidates = ["mic", "pred_mic", "value"]
    mic_col = next((cl[c] for c in mic_candidates if c in cl), None)
    if mic_col is not None:
        out = df[[sample_col, mic_col]].copy()
        out.columns = ["sample", "pred"]
        return out

    # fallback: last column
    out = df[[sample_col, df.columns[-1]]].copy()
    out.columns = ["sample", "pred"]
    return out

def _norm_sir(x: str|float) -> str|None:
    s = str(x).strip().upper()
    if s in {"S","I","R"}: return s
    if s in {"SUSC","SUSCEPTIBLE"}: return "S"
    if s in {"INTERMEDIATE"}: return "I"
    if s in {"RES","RESISTANT"}: return "R"
    return None

# ---------- plotting ----------------------------------------------------

def make_interactive_map(
    df_pts: pd.DataFrame,
    value_mode: str | None,
    title: str,
    agg_df: pd.DataFrame | None,
    agg_label: str | None,
    world_geojson: dict | None,
    out_html: Path,
    log10: bool,
    cluster: bool,
    iso3_key: str = "iso_a3",
):
    # center & base map
    center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())] if len(df_pts) else [20, 0]
    m = folium.Map(location=center, zoom_start=2, tiles="cartodbpositron")

    # ----- feature group (layer) -----
    layer_name = title if title else "isolates"
    fg = folium.FeatureGroup(name=layer_name, show=True)
    m.add_child(fg)

    # ----- choropleth (optional) -----
    if agg_df is not None and world_geojson is not None:
        vmin, vmax = np.nanmin(agg_df["value"]), np.nanmax(agg_df["value"])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
        cmap.caption = agg_label

        props0 = set(world_geojson["features"][0]["properties"].keys())
        candidates = [iso3_key, "iso_a3", "ISO_A3", "ADM0_A3", "adm0_a3", "WB_A3", "wb_a3", "ISO3166-1-Alpha-3"]
        iso_prop = next((k for k in candidates if k in props0), None)
        if iso_prop is None:
            raise SystemExit(
                f"Could not find an ISO3 property in GeoJSON. Tried {candidates}. "
                f"Available keys include: {sorted(list(props0))[:12]}"
            )
        print(f"[INFO] Using GeoJSON ISO3 property: {iso_prop}")

        folium.Choropleth(
            geo_data=world_geojson,
            data=agg_df,
            columns=["iso3", "value"],
            key_on=f"feature.properties.{iso_prop}",
            fill_color="YlOrRd",
            fill_opacity=0.8,
            line_opacity=0.2,
            nan_fill_opacity=0.05,
            highlight=True,
        ).add_to(m)
        cmap.add_to(m)

    # ----- bi-variate cluster: size = n, colour = %R (white→red) -----
    cluster_layer = None
    if cluster:
        icon_create_js = """
        function(cluster){
            const markers = cluster.getAllChildMarkers();
            const n = markers.length;

            // count Resistant by inspecting the rendered color
            let r = 0;
            markers.forEach(m => {
                let col = '';
                if (m && m.options) {
                  col = (m.options.fillColor || m.options.color || '').toString().toLowerCase().replace(/\\s+/g,'');
                }
                if (col === '#d62728' || col.includes('214,39,40')) {
                  r += 1; // treat as Resistant
                }
            });

            const p = n ? (r / n) : 0;
            const pct = Math.round(p * 100);

            const R = Math.round(255*(1-p) + 214*p);
            const G = Math.round(255*(1-p) +  39*p);
            const B = Math.round(255*(1-p) +  40*p);
            const color = `rgb(${R},${G},${B})`;

            const size = Math.max(28, Math.min(56, 28 + 8 * Math.log10(n + 1)));
            const html = `
              <div style="
                background:${color};
                color:#000;
                border:1px solid rgba(0,0,0,0.35);
                box-shadow:0 0 2px rgba(0,0,0,0.25);
                border-radius:${size/2}px;
                width:${size}px; height:${size}px;
                display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                font-weight:700; line-height:1;">
                <div>${pct}%</div>
                <div style="font-size:10px; line-height:10px;">n=${n}</div>
              </div>`;
            return new L.DivIcon({html: html, className:'marker-cluster', iconSize:[size,size]});
        }
        """

        cluster_layer = MarkerCluster(icon_create_function=icon_create_js)
        fg.add_child(cluster_layer)

        # tooltip: show %R and n on hover
        cluster_js = f"""
        (function(){{
                var layer = {cluster_layer.get_name()};
                layer.on('clustermouseover', function (a) {{
                var markers = a.layer.getAllChildMarkers();
                var n = markers.length, r = 0;
                markers.forEach(function(m){{ if ((m.options && m.options.sir) === 'R') r += 1; }});
                var pct = n ? Math.round(100*r/n) : 0;
                a.layer.bindTooltip('%R: ' + pct + '% (n=' + n + ')', {{permanent:false, direction:'top', opacity:0.9}}).openTooltip();
            }});
            layer.on('clustermouseout', function (a) {{ a.layer.unbindTooltip(); }});
        }})();
        """
        m.get_root().html.add_child(folium.Element("<script>" + cluster_js + "</script>"))

    # ----- point coloring -----
    legend_html = None
    if value_mode == "sir":
        def to_color(sir): return SIR_COLORS.get(_norm_sir(sir), "#7f7f7f")
        legend_html = """<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                          background: white; padding: 8px 10px; border: 1px solid #ccc;">
                          <b>Category</b><br>
                          <span style="color:#2ca02c;">&#11044;</span> S&nbsp;&nbsp;
                          <span style="color:#ffbf00;">&#11044;</span> I&nbsp;&nbsp;
                          <span style="color:#d62728;">&#11044;</span> R
                         </div>"""
    elif value_mode == "mic":
        vals = df_pts["mic"].dropna().values
        if len(vals):
            if log10: vals = np.log10(vals[vals > 0])
            vmin, vmax = np.nanpercentile(vals, [5, 95])
            reds = linear.Reds_09.scale(vmin, vmax)
            reds.caption = "Predicted MIC" + (" (log10)" if log10 else "")
            def to_color(v):
                if pd.isna(v): return "#7f7f7f"
                x = math.log10(v) if (log10 and v and v > 0) else v
                return reds(x)
            reds.add_to(m)
        else:
            def to_color(_): return "#7f7f7f"
    else:
        def to_color(_): return "#1f77b4"

    # ----- add points -----
    for _, r in df_pts.iterrows():
        if value_mode == "sir":
            label = _norm_sir(r.get("sir"))
            val_for_color = label
        elif value_mode == "mic":
            mic = r.get("mic")
            label = (f"{mic:g}" if pd.notna(mic) else "NA")
            val_for_color = r.get("mic")
        else:
            label = ""
            val_for_color = None

        color = to_color(val_for_color)

        popup = folium.Popup(folium.IFrame(
            html=f"<b>{r['sample']}</b>"
                 + (f"<br>Country: {r.get('country','')}" if 'country' in r else "")
                 + (f"<br>Value: {label}" if label else ""),
            width=220, height=90), max_width=260)

        marker = folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=4,
            color=color,
            fill=True,
            fill_color=color,  # used by cluster %R logic
            fill_opacity=0.9,
            weight=0,
            popup=popup,
            sir=(label or "")
        )
        (cluster_layer or fg).add_child(marker)

    folium.LayerControl(collapsed=False).add_to(m)
    if value_mode == "sir" and legend_html:
        m.get_root().html.add_child(folium.Element(legend_html))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"[OK] interactive map → {out_html}")

def static_export_png_svg(world_gdf, df_pts, agg_df, agg_label, value_mode,
                          out_png: Path|None, out_svg: Path|None, log10: bool):
    # Static export only if geopandas is available
    if gpd is None:
        print("[INFO] geopandas not available; skipping static PNG/SVG export.")
        return

    # Merge choropleth values onto shapes
    plot_gdf = world_gdf.copy()
    if agg_df is not None:
        plot_gdf = plot_gdf.merge(agg_df, left_on="iso_a3", right_on="iso3", how="left")

    # Colormap & normalization (Matplotlib only)
    cmap = cm.get_cmap("YlOrRd")
    norm = None
    if agg_df is not None and plot_gdf["value"].notna().any():
        vals = plot_gdf["value"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if vmin == vmax:   # avoid zero-range legends
                vmin = 0.0
                vmax = vmax if vmax > 0 else 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ext, outpath in [("png", out_png), ("svg", out_svg)]:
        if outpath is None:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.set_title("Isolates & country summary", weight="bold")

        # Draw countries
        if norm is None:
            plot_gdf.plot(ax=ax, color="#efefef", edgecolor="#cccccc", linewidth=0.3)
        else:
            plot = plot_gdf.plot(
                ax=ax, column="value", cmap=cmap, linewidth=0.3,
                edgecolor="#cccccc", missing_kwds={"color": "#f7f7f7"}, legend=True
            )
            leg = ax.get_legend()
            if leg is not None and agg_label:
                leg.set_title(agg_label)

        # Draw points
        if len(df_pts):
            if value_mode == "sir":
                cols = df_pts["sir"].map(SIR_COLORS).fillna("#7f7f7f")
            elif value_mode == "mic":
                v = pd.to_numeric(df_pts["mic"], errors="coerce")
                if log10:
                    v = np.where(v > 0, np.log10(v), np.nan)
                if np.isfinite(v).any():
                    vmin_p, vmax_p = np.nanpercentile(v[np.isfinite(v)], [5, 95])
                    normp = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)
                    cols = [cmap(normp(x)) if np.isfinite(x) else "#7f7f7f" for x in v]
                else:
                    cols = "#7f7f7f"
            else:
                cols = "#1f77b4"
            ax.scatter(df_pts["lon"], df_pts["lat"], s=8, c=cols, alpha=0.9, linewidths=0, zorder=5)

        ax.set_axis_off()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight", dpi=200 if ext == "png" else None)
        plt.close(fig)
        print(f"[OK] static {ext.upper()} saved to {outpath}")

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
    ap.add_argument("--geojson-iso3-key", default="iso_a3", help="Property in the GeoJSON with ISO3 (e.g., iso_a3, ISO_A3, ADM0_A3).")
    args = ap.parse_args()

    meta = _read_metadata(args.metadata, args.id_col, args.country_col)

    # attach predictions if provided
    df = meta.copy()
    value_mode = None
    if args.pred_file is not None:
        pr = _read_preds(args.pred_file)
        pr["sample"] = pr["sample"].astype(str).str.strip()
        df["sample"] = df["sample"].astype(str).str.strip()
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
    if "sir" in df.columns:
        print("[DEBUG] SIR counts:", df["sir"].value_counts(dropna=False).to_dict())
        print("[DEBUG] Example joined rows:\n",
                df[["sample","country","sir"]].dropna(subset=["sir"]).head(10))

    matched = df["pred"].notna().sum()
    print(f"[DEBUG] merged rows: {len(df)}, with pred: {matched}")
    print("[DEBUG] SIR head:", df["pred"].dropna().astype(str).str.upper().value_counts().head(5).to_dict())

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
            # proportion of R per ISO3
            agg_df = grp.agg(
                value=("sir", lambda s: (s == "R").mean()),
                n=("sir", "size"),
            )
            agg_label = "% Resistant (predicted)"

        elif metric == "median_mic":
            if value_mode != "mic":
                raise SystemExit("--metric median_mic requires MIC predictions.")
            agg_df = grp["mic"].median().reset_index(name="value")
            agg_label = "Median predicted MIC"

        elif metric == "count":
            agg_df = grp.size().reset_index(name="value")
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
        
    # make interactive map
    make_interactive_map(
        df_pts=df,
        value_mode=value_mode,
        title=args.title,
        agg_df=agg_df,
        agg_label=agg_label,
        world_geojson=world_geojson,
        out_html=args.out_html,
        log10=args.log10,
        cluster=args.cluster,
        iso3_key=args.geojson_iso3_key,
    )

    # static exports (PNG/SVG)
    if args.export_png or args.export_svg:
        if world_gdf is None:
            if gpd is None:
                print("[INFO] geopandas not available; cannot produce PNG/SVG without it.")
            else:
                world_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
                world_gdf = world_gdf[world_gdf["iso_a3"] != "-99"]
        if world_gdf is not None:
            static_export_png_svg(
                world_gdf=world_gdf,
                df_pts=df,
                agg_df=agg_df,
                agg_label=agg_label,
                value_mode=value_mode,
                out_png=args.export_png,
                out_svg=args.export_svg,
                log10=args.log10,
            )

if __name__ == "__main__":
    main()
