"""Shared state-subset utilities for replication scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd


def _clean_state_set(values: pd.Series) -> Set[str]:
    out: Set[str] = set()
    for raw in values.dropna().astype(str):
        s = raw.strip()
        if s and s.lower() != "nan":
            out.add(s)
    return out


def load_sales_volume_states(
    repo_root: Path,
    *,
    csv_relpath: str = "data/Raw Data/Full_Merged_Data_by_State.csv",
    gallons_col: str = "Gallon Sales (1,000 gallons)",
    state_col: str = "State",
) -> Set[str]:
    """
    Return states with positive observed sales volume in the merged raw-data file.
    """
    sales_path = repo_root / csv_relpath
    if not sales_path.exists():
        raise FileNotFoundError(f"Sales-volume file not found: {sales_path}")

    df = pd.read_csv(sales_path, usecols=[state_col, gallons_col])
    gallons = pd.to_numeric(df[gallons_col], errors="coerce")
    states = _clean_state_set(df.loc[gallons > 0, state_col])
    if not states:
        raise ValueError(f"No states with positive `{gallons_col}` found in {sales_path}.")
    return states


def resolve_state_subset(
    subset: str,
    *,
    repo_root: Path,
) -> Tuple[Optional[Set[str]], Dict[str, object]]:
    """
    Resolve subset selector into a concrete state set (or None for full sample).

    Returns
    -------
    include_states:
        None for full sample, otherwise a set of state names to keep.
    meta:
        Small metadata dict describing the resolved subset.
    """
    subset_norm = str(subset).strip().lower()
    if subset_norm == "all":
        return None, {
            "subset_id": "all",
            "subset_label": "Full AAA sample",
            "n_states_raw": None,
        }
    if subset_norm == "sales36":
        states = load_sales_volume_states(repo_root)
        return states, {
            "subset_id": "sales36",
            "subset_label": "States with positive observed sales volume",
            "n_states_raw": int(len(states)),
            "source": "data/Raw Data/Full_Merged_Data_by_State.csv",
        }
    raise ValueError(f"Unknown subset `{subset}`. Expected one of: all, sales36.")
