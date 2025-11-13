from __future__ import annotations
"""bamps_ml.plotting

Usage (trainer):
    from bamps_ml.plotting import PALETTE, confusion_plot, shap_beeswarm
"""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib import cm as _cm
from shap import TreeExplainer, summary_plot

# -----------------------------------------------------------------------------
# Global colour language (white = susceptible/low; red = resistant/high)
# -----------------------------------------------------------------------------
PALETTE = {
    "S": "#ffffff",   # white
    "I": "#ffc9c9",   # light rose (mid)
    "R": "#d7301f",   # strong red
}
# Confusion matrices should show magnitude, not class — use sequential Reds
_CMAP_CONF = plt.get_cmap("Reds")
# SHAP colormap: white → red
_CMAP_SHAP = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#d7301f"])
_NORM_01 = Normalize(vmin=0.0, vmax=1.0)

# -----------------------------------------------------------------------------
# Confusion‑matrix plotting
# -----------------------------------------------------------------------------

def confusion_plot(cm: np.ndarray, labels: Sequence[str], out_png: Path) -> None:
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square")
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap=_CMAP_CONF, vmin=0, vmax=max(1, cm.max()))

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    # legend note (global colour language)
    fig.text(
        0.5, -0.02,
        "Colour map: white = susceptible/absent/low; red = resistant/present/high",
        ha="center", va="top", fontsize=9
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# SHAP beeswarm with palette
# -----------------------------------------------------------------------------

def shap_beeswarm(model, X, out_png: Path) -> None:
    """Save a SHAP beeswarm/bar plot using palette colours."""
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # If multi-class: SHAP returns a list [class0, class1, ...] each (n_samples, n_features).
    # Collapse to a single importance matrix so we get ONE plot with ONE cmap (white→red).
    if isinstance(shap_values, list):
        import numpy as _np
        # Sum absolute contributions across classes ⇒ (n_samples, n_features)
        shap_global = _np.sum([_np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_global = shap_values

    try:
        # Beeswarm with value-based colouring:
        # white ~ low/absent (e.g., gene=0), red ~ high/present (gene=1)
        summary_plot(shap_global, X, show=False, cmap=_CMAP_SHAP)
    except Exception:
        # Bar fallback (no value colouring): still a single, global importance view
        summary_plot(shap_global, X, plot_type="bar", show=False)
    # legend note (global colour language)
    fig = plt.gcf()
    fig.text(
        0.5, 0.02,
        "Point colour encodes original feature value: white = low/absent, red = high/present",
        ha="center", va="bottom", fontsize=9
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_png, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# Horizontal bar chart (non-SHAP) with white→red colouring
# -----------------------------------------------------------------------------

def barh_importances(
    names: Sequence[str],
    scores: Sequence[float],
    out_png: Path,
    title: str | None = None,
    top_n: int | None = 20,
) -> None:
    """
    Plot horizontal bars where colour reflects magnitude (white→red).
    - names: feature names (iterable of str)
    - scores: importance scores (any non-negative scale)
    """
    names = list(names)
    scores = np.asarray(scores, dtype=float)
    if top_n is not None and len(scores) > top_n:
        idx = np.argsort(scores)[-top_n:]
        names = [names[i] for i in idx]
        scores = scores[idx]

    order = np.argsort(scores)
    names = [names[i] for i in order]
    scores = scores[order]

    # normalise 0..1 for colour mapping
    smax = scores.max() if len(scores) and np.isfinite(scores).any() else 1.0
    col = _cm.get_cmap(_CMAP_SHAP)( _NORM_01(scores / (smax if smax > 0 else 1.0)) )

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.25 * len(scores) + 1.5)))
    ax.barh(range(len(scores)), scores, color=col, edgecolor="#555555")
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    if title:
        ax.set_title(title, weight="bold")
    # legend note
    fig.text(0.5, 0.02,
             "Point colour encodes original feature value: white = low/absent, red = high/present",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)