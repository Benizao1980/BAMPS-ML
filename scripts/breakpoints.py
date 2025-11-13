import pandas as pd, numpy as np
from .breakpoints import get_table
from pathlib import Path
import yaml

def load_config(path="config/config.yaml"):
    with open(Path(path)) as fh:
        return yaml.safe_load(fh)

def resolve_breakpoints(standard: str, overrides: dict | None = None):
    """Return {antibiotic → {S,I,R}} for the chosen standard,
    merged with any per-antibiotic overrides."""
    table = get_table(standard)
    if overrides:
        table = {**table, **overrides}          # overrides win
    return table

def mic_to_category(mic_val: float, bp: dict):
    if pd.isna(mic_val):
        return np.nan
    if "R" in bp and mic_val > bp["R"]:
        return "R"
    if "I" in bp and mic_val > bp.get("I", bp["S"]):
        return "I"
    return "S"
