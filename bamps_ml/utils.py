#!/usr/bin/env python3
"""bamps_ml.utils — config + breakpoint helpers"""
from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

###############################################################################
# Config loader
###############################################################################

def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load project‑wide YAML config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as fh:
        return yaml.safe_load(fh)

###############################################################################
# Breakpoint handling
###############################################################################
# Minimal built‑in EUCAST & CLSI tables (extend as needed)
EUCAST = {
    "ciprofloxacin": {"S": 0.25, "I": 0.5, "R": 1},
    "imipenem": {"S": 2, "I": 4, "R": 8},
    "meropenem": {"S": 2, "I": 4, "R": 8},
    "colistin": {"S": 2, "R": 4},
}
CLSI = {
    "ciprofloxacin": {"S": 1, "R": 4},
    "imipenem": {"S": 1, "R": 4},
    "meropenem": {"S": 1, "R": 4},
    "colistin": {"S": 2, "R": 4},
}

_DEF_TABLES = {"EUCAST": EUCAST, "CLSI": CLSI}

def resolve_breakpoints(standard: str, overrides: dict | None = None) -> dict:
    """Return merged breakpoint table for the chosen standard."""
    std = standard.upper()
    if std not in _DEF_TABLES:
        raise ValueError(f"Unknown breakpoint standard: {standard}")
    table = {**_DEF_TABLES[std]}  # shallow copy
    if overrides:
        table.update(overrides)
    return table

###############################################################################
# MIC numeric → category helper
###############################################################################

def mic_to_category(mic_val: float | int | str, bp: dict) -> str | float:
    """Convert numeric MIC to S/I/R. Returns NaN if unmappable."""
    try:
        mic = float(mic_val)
    except (ValueError, TypeError):
        return pd.NA
    if not bp:
        return pd.NA
    if "R" in bp and mic > bp["R"]:
        return "R"
    if "I" in bp and mic > bp.get("I", bp["S"]):
        return "I"
    return "S"
