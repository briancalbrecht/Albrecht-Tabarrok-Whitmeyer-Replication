"""
Run and compare data-disciplined baseline-mass options for robust bounds.

This script keeps the robust program intact and varies only how baseline
state masses are constructed:
  - legacy station-count weights,
  - exact 1972 gallon masses where observed,
  - sales-based imputation for missing gallon states using a price band.

Outputs:
  - table_state_baseline_options.csv
  - state_baseline_options_meta.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from state_joint_robust_bounds import (
    REPO_ROOT,
    MarketSpec,
    STATE_ABBREVS,
    StateRow,
    build_state_markets,
    load_state_rationing,
    optimize_joint_bound_over_p0,
    solve_joint_bound,
)
from state_status_joint_robust_bounds import (
    build_state_status_markets,
    optimize_joint_bound_over_p0_grouped,
    solve_joint_bound_grouped,
)


@dataclass(frozen=True)
class BaselineScenario:
    scenario_id: str
    label: str
    mode: str  # stations | known_only | impute
    drop_indiana: bool = False
    band: Optional[str] = None  # full_no_ind | iqr_no_ind
    point: Optional[str] = None  # low | mid | high


def _load_baseline_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data/Raw Data/Full_Merged_Data_by_State.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline data: {path}")

    df = pd.read_csv(path, usecols=["State", "Gasoline Stations", "Sales ($1,000)", "Gallon Sales (1,000 gallons)"])
    df["State"] = df["State"].astype(str).str.strip()
    df["stations_1972"] = pd.to_numeric(df["Gasoline Stations"], errors="coerce")
    df["sales_1000"] = pd.to_numeric(df["Sales ($1,000)"], errors="coerce")
    gallons_raw = df["Gallon Sales (1,000 gallons)"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["gallons_1000"] = pd.to_numeric(gallons_raw, errors="coerce")
    df["implied_price"] = df["sales_1000"] / df["gallons_1000"]

    # Keep one row per state; dataset is already state-level.
    df = df.sort_values("State").drop_duplicates(subset=["State"], keep="first").reset_index(drop=True)
    return df[["State", "stations_1972", "sales_1000", "gallons_1000", "implied_price"]]


def _price_band(
    known: pd.DataFrame,
    *,
    band: str,
) -> Tuple[float, float]:
    if known.empty:
        raise ValueError("No known-gallon states available for price-band calibration.")
    if band == "full_no_ind":
        p_lo = float(known["implied_price"].min())
        p_hi = float(known["implied_price"].max())
    elif band == "iqr_no_ind":
        p_lo = float(known["implied_price"].quantile(0.25))
        p_hi = float(known["implied_price"].quantile(0.75))
    else:
        raise ValueError(f"Unknown band={band!r}.")
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_lo <= 0.0 or p_hi <= p_lo:
        raise ValueError(f"Invalid price band [{p_lo}, {p_hi}] for {band}.")
    return p_lo, p_hi


def _scenario_mass_map(
    *,
    rows: Sequence[StateRow],
    baseline_df: pd.DataFrame,
    scenario: BaselineScenario,
) -> Tuple[List[StateRow], Dict[str, object]]:
    state_order = [r.state for r in rows]
    state_set = set(state_order)

    b = baseline_df[baseline_df["State"].isin(state_set)].copy()
    # Track missing states in baseline file separately.
    missing_in_baseline_file = sorted(state_set - set(b["State"]))

    known = b[b["gallons_1000"].notna() & (b["gallons_1000"] > 0)].copy()
    if scenario.drop_indiana:
        known = known[known["State"] != "Indiana"].copy()
    known_states = set(known["State"])

    if scenario.mode == "stations":
        out = [StateRow(state=r.state, stations=float(r.stations), rationing=float(r.rationing)) for r in rows]
        meta = {
            "mode": scenario.mode,
            "known_states": int(len(known_states)),
            "missing_in_baseline_file": missing_in_baseline_file,
            "imputed_states": [],
        }
        return out, meta

    sales_map = dict(zip(b["State"], b["sales_1000"]))
    gallons_map = dict(zip(b["State"], b["gallons_1000"]))

    if scenario.mode == "known_only":
        include_states = set(known_states)
        mass_map: Dict[str, float] = {}
        for s in include_states:
            g = gallons_map.get(s)
            if g is None or not np.isfinite(g) or g <= 0:
                continue
            mass_map[s] = float(g)
        if not mass_map:
            raise ValueError(f"Scenario {scenario.scenario_id}: no states with known gallons after filtering.")
        out = [
            StateRow(state=r.state, stations=float(mass_map[r.state]), rationing=float(r.rationing))
            for r in rows
            if r.state in mass_map
        ]
        meta = {
            "mode": scenario.mode,
            "known_states": int(len(known_states)),
            "included_states": int(len(out)),
            "imputed_states": [],
            "missing_in_baseline_file": missing_in_baseline_file,
        }
        return out, meta

    if scenario.mode != "impute":
        raise ValueError(f"Unknown scenario mode={scenario.mode!r}.")

    if scenario.band is None or scenario.point is None:
        raise ValueError(f"Scenario {scenario.scenario_id} missing band/point for imputation.")

    known_for_band = known
    if scenario.band.endswith("_no_ind"):
        known_for_band = known_for_band[known_for_band["State"] != "Indiana"].copy()
    p_lo, p_hi = _price_band(known_for_band, band=scenario.band)
    if scenario.point == "low":
        # High price -> low inferred gallons.
        p_use = p_hi
    elif scenario.point == "high":
        # Low price -> high inferred gallons.
        p_use = p_lo
    elif scenario.point == "mid":
        p_use = 0.5 * (p_lo + p_hi)
    else:
        raise ValueError(f"Unknown point={scenario.point!r}.")

    mass_map: Dict[str, float] = {}
    imputed_states: List[str] = []
    for r in rows:
        s = r.state
        g = gallons_map.get(s)
        if g is not None and np.isfinite(g) and g > 0:
            mass_map[s] = float(g)
            continue
        sales = sales_map.get(s)
        if sales is None or not np.isfinite(sales) or sales <= 0:
            # If no sales to impute, drop state from this scenario.
            continue
        mass_map[s] = float(sales / p_use)
        imputed_states.append(s)

    out = [
        StateRow(state=r.state, stations=float(mass_map[r.state]), rationing=float(r.rationing))
        for r in rows
        if r.state in mass_map
    ]
    if len(out) < 2:
        raise ValueError(f"Scenario {scenario.scenario_id}: too few states after imputation.")

    meta = {
        "mode": scenario.mode,
        "known_states": int(len(known_states)),
        "included_states": int(len(out)),
        "imputed_states": sorted(imputed_states),
        "missing_in_baseline_file": missing_in_baseline_file,
        "price_band": scenario.band,
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "p_use": float(p_use),
    }
    return out, meta


def _run_state_bounds(
    *,
    rows: Sequence[StateRow],
    shortage: float,
    p_control: float,
    eps_open: float,
    p_base: float,
    eps_L: float,
    eps_U: float,
    q_max: float,
    outer_p0: str,
    p0_method: str,
    n_grid: int,
    outer_search_grid: int,
    outer_max_iters: int,
    outer_coord_grid: int,
    outer_starts: int,
    outer_seed: int,
    outer_tol: float,
) -> Dict[str, float]:
    markets, meta = build_state_markets(
        list(rows),
        shortage=shortage,
        p_control=p_control,
        eps_open=eps_open,
        p_base=p_base,
        eps_L=eps_L,
        eps_U=eps_U,
        q_max=q_max,
        p0_method=p0_method,
    )
    Qbar = float(meta["Qbar"])
    p_max_grid = max(10.0, p_control, p_base)
    g_steep = -p_base / eps_L
    psi_ref = 0.5 * abs(g_steep) * shortage**2

    if outer_p0 == "coordsrch":
        p0_lo = np.array(meta["p0_lo_vec"], dtype=float)
        p0_hi = np.array(meta["p0_hi_vec"], dtype=float)
        p0_init = np.array(meta["p0_init_vec"], dtype=float)
        search_n_grid = max(101, min(n_grid, outer_search_grid))
        upper = optimize_joint_bound_over_p0(
            markets,
            p0_lo=p0_lo,
            p0_hi=p0_hi,
            start_p0=p0_init,
            Qbar=Qbar,
            mode="upper",
            p_min=0.0,
            p_max=p_max_grid,
            n_grid=n_grid,
            search_n_grid=search_n_grid,
            max_iters=outer_max_iters,
            coord_grid_points=outer_coord_grid,
            n_starts=outer_starts,
            seed=outer_seed,
            tol=outer_tol,
        )
        lower = optimize_joint_bound_over_p0(
            markets,
            p0_lo=p0_lo,
            p0_hi=p0_hi,
            start_p0=p0_init,
            Qbar=Qbar,
            mode="lower",
            p_min=0.0,
            p_max=p_max_grid,
            n_grid=n_grid,
            search_n_grid=search_n_grid,
            max_iters=outer_max_iters,
            coord_grid_points=outer_coord_grid,
            n_starts=outer_starts,
            seed=outer_seed,
            tol=outer_tol,
        )
    else:
        upper = solve_joint_bound(markets, Qbar=Qbar, mode="upper", p_min=0.0, p_max=p_max_grid, n_grid=n_grid)
        lower = solve_joint_bound(markets, Qbar=Qbar, mode="lower", p_min=0.0, p_max=p_max_grid, n_grid=n_grid)

    out = {
        "n_states": int(meta["n_states"]),
        "Qbar": float(Qbar),
        "q_open": float(meta["q_open"]),
        "q_non_open": float(meta["q_non_open"]),
        "phi_lower_pct": 100.0 * float(lower["Phi"]),
        "phi_upper_pct": 100.0 * float(upper["Phi"]),
        "ratio_lower": float(lower["Phi"]) / psi_ref if psi_ref > 0 else float("nan"),
        "ratio_upper": float(upper["Phi"]) / psi_ref if psi_ref > 0 else float("nan"),
        "p_star_lower": float(lower["p_star"]),
        "p_star_upper": float(upper["p_star"]),
    }
    return out


def _run_state_status_bounds(
    *,
    rows: Sequence[StateRow],
    shortage: float,
    p_control: float,
    eps_open: float,
    p_base: float,
    eps_L: float,
    eps_U: float,
    q_max: float,
    outer_p0: str,
    p0_method: str,
    adding_up: str,
    n_grid: int,
    outer_search_grid: int,
    outer_max_iters: int,
    outer_coord_grid: int,
    outer_starts: int,
    outer_seed: int,
    outer_tol: float,
    include_state_shadow_map_data: bool = False,
) -> Dict[str, object]:
    markets, meta, shadow_df = build_state_status_markets(
        list(rows),
        shortage=shortage,
        p_control=p_control,
        eps_open=eps_open,
        p_base=p_base,
        eps_L=eps_L,
        eps_U=eps_U,
        q_max=q_max,
        p0_method=p0_method,
    )
    Qbar = float(meta["Qbar"])
    p_max_grid = max(10.0, p_control, p_base)
    g_steep = -p_base / eps_L
    psi_ref = 0.5 * abs(g_steep) * shortage**2

    if outer_p0 == "coordsrch":
        p0_lo = np.array(meta["p0_lo_vec"], dtype=float)
        p0_hi = np.array(meta["p0_hi_vec"], dtype=float)
        p0_init = np.array(meta["p0_init_vec"], dtype=float)
        search_n_grid = max(101, min(n_grid, outer_search_grid))
        if adding_up == "state":
            group_labels = [m.name.split("|", 1)[0] for m in markets]
            group_targets = {
                str(s): float(q)
                for s, q in shadow_df.groupby("state", sort=False)["q_obs"].sum().to_dict().items()
            }
            upper = optimize_joint_bound_over_p0_grouped(
                markets,
                p0_lo=p0_lo,
                p0_hi=p0_hi,
                start_p0=p0_init,
                group_labels=group_labels,
                group_targets=group_targets,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
                search_n_grid=search_n_grid,
                max_iters=outer_max_iters,
                coord_grid_points=outer_coord_grid,
                n_starts=outer_starts,
                seed=outer_seed,
                tol=outer_tol,
            )
            lower = optimize_joint_bound_over_p0_grouped(
                markets,
                p0_lo=p0_lo,
                p0_hi=p0_hi,
                start_p0=p0_init,
                group_labels=group_labels,
                group_targets=group_targets,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
                search_n_grid=search_n_grid,
                max_iters=outer_max_iters,
                coord_grid_points=outer_coord_grid,
                n_starts=outer_starts,
                seed=outer_seed,
                tol=outer_tol,
            )
        else:
            upper = optimize_joint_bound_over_p0(
                markets,
                p0_lo=p0_lo,
                p0_hi=p0_hi,
                start_p0=p0_init,
                Qbar=Qbar,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
                search_n_grid=search_n_grid,
                max_iters=outer_max_iters,
                coord_grid_points=outer_coord_grid,
                n_starts=outer_starts,
                seed=outer_seed,
                tol=outer_tol,
            )
            lower = optimize_joint_bound_over_p0(
                markets,
                p0_lo=p0_lo,
                p0_hi=p0_hi,
                start_p0=p0_init,
                Qbar=Qbar,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
                search_n_grid=search_n_grid,
                max_iters=outer_max_iters,
                coord_grid_points=outer_coord_grid,
                n_starts=outer_starts,
                seed=outer_seed,
                tol=outer_tol,
            )
    else:
        if adding_up == "state":
            group_labels = [m.name.split("|", 1)[0] for m in markets]
            group_targets = {
                str(s): float(q)
                for s, q in shadow_df.groupby("state", sort=False)["q_obs"].sum().to_dict().items()
            }
            upper = solve_joint_bound_grouped(
                markets,
                group_labels=group_labels,
                group_targets=group_targets,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
            )
            lower = solve_joint_bound_grouped(
                markets,
                group_labels=group_labels,
                group_targets=group_targets,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
            )
        else:
            upper = solve_joint_bound(markets, Qbar=Qbar, mode="upper", p_min=0.0, p_max=p_max_grid, n_grid=n_grid)
            lower = solve_joint_bound(markets, Qbar=Qbar, mode="lower", p_min=0.0, p_max=p_max_grid, n_grid=n_grid)

    out = {
        "n_states": int(meta["n_states"]),
        "n_markets_active": int(meta["n_markets_active"]),
        "Qbar": float(Qbar),
        "q_open": float(meta["q_open"]),
        "q_non_open": float(meta["q_non_open"]),
        "phi_lower_pct": 100.0 * float(lower["Phi"]),
        "phi_upper_pct": 100.0 * float(upper["Phi"]),
        "ratio_lower": float(lower["Phi"]) / psi_ref if psi_ref > 0 else float("nan"),
        "ratio_upper": float(upper["Phi"]) / psi_ref if psi_ref > 0 else float("nan"),
        "p_star_lower": float(lower["p_star"]),
        "p_star_upper": float(upper["p_star"]),
    }

    if include_state_shadow_map_data:
        if "p0_vector" in upper:
            p0_upper_vec = np.asarray(upper["p0_vector"], dtype=float)
        else:
            p0_upper_vec = np.asarray([m.p0 for m in markets], dtype=float)
        if "p0_vector" in lower:
            p0_lower_vec = np.asarray(lower["p0_vector"], dtype=float)
        else:
            p0_lower_vec = np.asarray([m.p0 for m in markets], dtype=float)

        if len(p0_upper_vec) != len(markets) or len(p0_lower_vec) != len(markets):
            raise ValueError("p0 vector length mismatch in state-status map data.")

        cell_df = shadow_df.copy()
        cell_df = cell_df.reset_index(drop=True)
        cell_df["p0_upper_cell"] = p0_upper_vec
        cell_df["p0_lower_cell"] = p0_lower_vec

        rows_map: List[Dict[str, object]] = []
        for state, grp in cell_df.groupby("state", sort=True):
            shares = grp["share_status"].to_numpy(dtype=float)
            share_sum = float(np.sum(shares))
            if share_sum <= 1e-12:
                continue
            w = shares / share_sum
            p_upper_avg = float(np.sum(w * grp["p0_upper_cell"].to_numpy(dtype=float)))
            p_lower_avg = float(np.sum(w * grp["p0_lower_cell"].to_numpy(dtype=float)))

            g_open = grp[grp["status"] == "open"]
            g_nonopen = grp[grp["status"] == "nonopen"]
            open_upper = float(g_open["p0_upper_cell"].iloc[0]) if not g_open.empty else float("nan")
            nonopen_upper = float(g_nonopen["p0_upper_cell"].iloc[0]) if not g_nonopen.empty else float("nan")
            open_lower = float(g_open["p0_lower_cell"].iloc[0]) if not g_open.empty else float("nan")
            nonopen_lower = float(g_nonopen["p0_lower_cell"].iloc[0]) if not g_nonopen.empty else float("nan")
            share_open = float(g_open["share_status"].iloc[0]) if not g_open.empty else 0.0
            share_nonopen = float(g_nonopen["share_status"].iloc[0]) if not g_nonopen.empty else 0.0
            rationing = float(grp["rationing"].iloc[0]) if "rationing" in grp.columns and not grp.empty else float("nan")

            rows_map.append(
                {
                    "state": str(state),
                    "abbrev": STATE_ABBREVS.get(str(state), str(state)[:2].upper()),
                    "p0_upper_avg": p_upper_avg,
                    "p0_lower_avg": p_lower_avg,
                    "p0_upper_open": open_upper,
                    "p0_upper_nonopen": nonopen_upper,
                    "p0_lower_open": open_lower,
                    "p0_lower_nonopen": nonopen_lower,
                    "share_open": share_open,
                    "share_nonopen": share_nonopen,
                    "rationing": rationing,
                }
            )
        out["state_shadow_map_df"] = pd.DataFrame(rows_map)
    return out


def _plot_state_avg_shadow_map(
    df: pd.DataFrame,
    *,
    p_base: float,
    out_dir: Path,
    filename_stem: str,
    title: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("Warning: plotly not available, skipping state-average shadow map.")
        return None, None

    if df is None or df.empty:
        print("Warning: no state-average shadow data available for map.")
        return None, None

    out_dir.mkdir(parents=True, exist_ok=True)
    # Enforce a consistent figure naming convention in exports.
    if not filename_stem.startswith("figure_"):
        filename_stem = f"figure_{filename_stem}"
    d = df.copy()
    d["upper_rel"] = d["p0_upper_avg"] / float(p_base)
    d["lower_rel"] = d["p0_lower_avg"] / float(p_base)
    all_abbrevs = {abbr for abbr in STATE_ABBREVS.values() if isinstance(abbr, str) and len(abbr) == 2}
    present_abbrevs = {str(a) for a in d["abbrev"].dropna().tolist()}
    missing_abbrevs = sorted(all_abbrevs - present_abbrevs)

    # Diverging palette with baseline (1.0) as the visual anchor.
    colors = ["#053061", "#2166ac", "#92c5de", "#fddbc7", "#d6604d", "#67001f"]

    def _shared_range(upper_values: pd.Series, lower_values: pd.Series) -> Tuple[float, float]:
        arr = np.concatenate(
            [
                upper_values.to_numpy(dtype=float),
                lower_values.to_numpy(dtype=float),
            ]
        )
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.6, 2.4
        lo_q = float(np.quantile(arr, 0.05))
        hi_q = float(np.quantile(arr, 0.95))
        # Shared symmetric range around baseline=1 for cross-panel comparability.
        half_span = max(abs(1.0 - lo_q), abs(hi_q - 1.0), 0.25)
        lo = max(0.0, 1.0 - half_span)
        hi = 1.0 + half_span
        if hi <= lo + 1e-9:
            hi = lo + 1e-3
        return lo, hi

    zmin, zmax = _shared_range(d["upper_rel"], d["lower_rel"])
    d["upper_rel_plot"] = d["upper_rel"].clip(lower=zmin, upper=zmax)
    d["lower_rel_plot"] = d["lower_rel"].clip(lower=zmin, upper=zmax)

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=(
            "A. Upper bound: State-average shadow price<br>at observed allocation",
            "B. Lower bound: State-average shadow price<br>at observed allocation",
        ),
        vertical_spacing=0.08,
    )
    # Plotly centers subplot titles on the subplot domain (excluding colorbars),
    # which appears visually offset; pin both panel headers to figure center.
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.update(x=0.5, xanchor="center", align="center")

    custom = d[
        [
            "state",
            "p0_upper_open",
            "p0_upper_nonopen",
            "p0_lower_open",
            "p0_lower_nonopen",
            "share_open",
            "share_nonopen",
            "rationing",
        ]
    ].to_numpy()

    if missing_abbrevs:
        # Render no-data states in white without changing non-U.S. background.
        fig.add_trace(
            go.Choropleth(
                locations=missing_abbrevs,
                z=[1.0] * len(missing_abbrevs),
                locationmode="USA-states",
                colorscale=[[0.0, "#ffffff"], [1.0, "#ffffff"]],
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                marker_line_color="#808080",
                marker_line_width=0.8,
                hovertemplate="%{location}<br>No data<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Choropleth(
                locations=missing_abbrevs,
                z=[1.0] * len(missing_abbrevs),
                locationmode="USA-states",
                colorscale=[[0.0, "#ffffff"], [1.0, "#ffffff"]],
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                marker_line_color="#808080",
                marker_line_width=0.8,
                hovertemplate="%{location}<br>No data<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Choropleth(
            locations=d["abbrev"],
            z=d["upper_rel_plot"],
            locationmode="USA-states",
            colorscale=[[i / 5, colors[i]] for i in range(6)],
            zmin=zmin,
            zmax=zmax,
            zmid=1.0,
            colorbar=dict(
                title="Upper bound state-average<br>shadow price / baseline",
                len=0.35,
                y=0.82,
                x=1.02,
            ),
            customdata=custom,
            hovertemplate=(
                "%{customdata[0]} (%{location})<br>"
                "State-average shadow price: %{z:.2f} x baseline<br>"
                "Open shadow price: %{customdata[1]:.2f}<br>"
                "Non-open shadow price: %{customdata[2]:.2f}<br>"
                "Open share: %{customdata[5]:.1%}, Non-open share: %{customdata[6]:.1%}<br>"
                "Rationing share: %{customdata[7]:.1%}<br>"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Choropleth(
            locations=d["abbrev"],
            z=d["lower_rel_plot"],
            locationmode="USA-states",
            colorscale=[[i / 5, colors[i]] for i in range(6)],
            zmin=zmin,
            zmax=zmax,
            zmid=1.0,
            colorbar=dict(
                title="Lower bound state-average<br>shadow price / baseline",
                len=0.35,
                y=0.18,
                x=1.02,
            ),
            customdata=custom,
            hovertemplate=(
                "%{customdata[0]} (%{location})<br>"
                "State-average shadow price: %{z:.2f} x baseline<br>"
                "Open shadow price: %{customdata[3]:.2f}<br>"
                "Non-open shadow price: %{customdata[4]:.2f}<br>"
                "Open share: %{customdata[5]:.1%}, Non-open share: %{customdata[6]:.1%}<br>"
                "Rationing share: %{customdata[7]:.1%}<br>"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="",
        title_font_size=16,
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#f0f0f0",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
        ),
        geo2=dict(
            scope="usa",
            showland=True,
            landcolor="#f0f0f0",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
        ),
        margin=dict(l=40, r=160, t=40, b=20),
        font=dict(size=11),
    )

    pdf_path = out_dir / f"{filename_stem}.pdf"
    png_path = out_dir / f"{filename_stem}.png"
    try:
        pio.write_image(fig, str(png_path), width=900, height=900, scale=2)
        pio.write_image(fig, str(pdf_path), width=900, height=900, scale=2)
        return pdf_path, png_path
    except Exception as e:
        print(f"Warning: could not save state-average shadow map: {e}")
        return None, None


def _default_scenarios() -> List[BaselineScenario]:
    return [
        BaselineScenario("stations48", "Legacy station-count baseline masses", mode="stations"),
        BaselineScenario("known_exact", "Only known-gallon states (exact baselines)", mode="known_only"),
        BaselineScenario(
            "known_exact_no_indiana",
            "Only known-gallon states (exact baselines), drop Indiana",
            mode="known_only",
            drop_indiana=True,
        ),
        BaselineScenario(
            "impute_full_low",
            "Known gallons + missing imputed (full band, low quantity)",
            mode="impute",
            band="full_no_ind",
            point="low",
        ),
        BaselineScenario(
            "impute_full_mid",
            "Known gallons + missing imputed (full band, midpoint)",
            mode="impute",
            band="full_no_ind",
            point="mid",
        ),
        BaselineScenario(
            "impute_full_high",
            "Known gallons + missing imputed (full band, high quantity)",
            mode="impute",
            band="full_no_ind",
            point="high",
        ),
        BaselineScenario(
            "impute_iqr_low",
            "Known gallons + missing imputed (IQR band, low quantity)",
            mode="impute",
            band="iqr_no_ind",
            point="low",
        ),
        BaselineScenario(
            "impute_iqr_mid",
            "Known gallons + missing imputed (IQR band, midpoint)",
            mode="impute",
            band="iqr_no_ind",
            point="mid",
        ),
        BaselineScenario(
            "impute_iqr_high",
            "Known gallons + missing imputed (IQR band, high quantity)",
            mode="impute",
            band="iqr_no_ind",
            point="high",
        ),
    ]


def _parse_scenario_filter(raw: str, all_scenarios: Sequence[BaselineScenario]) -> List[BaselineScenario]:
    wanted = {s.strip() for s in raw.split(",") if s.strip()}
    if not wanted:
        return list(all_scenarios)
    by_id = {s.scenario_id: s for s in all_scenarios}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise ValueError(f"Unknown scenario ids: {missing}.")
    return [by_id[w] for w in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare data-disciplined baseline-mass options for robust bounds.")
    parser.add_argument("--replication", action="store_true", help="Write outputs to disk.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: output/scenario_baseline_options).")
    parser.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario ids to run; default runs all.")
    parser.add_argument("--run-state", action="store_true", default=False, help="Run state-level robust bounds.")
    parser.add_argument("--run-status", action="store_true", default=False, help="Run state×status robust bounds.")
    parser.add_argument(
        "--adding-up",
        type=str,
        choices=["national", "state"],
        default="national",
        help=(
            "Constraint mode. Note: state adding-up is supported only with --outer-p0 fixed."
        ),
    )
    parser.add_argument("--shortage", type=float, default=0.09)
    parser.add_argument("--p-control", type=float, default=0.8)
    parser.add_argument("--eps-open", type=float, default=0.3)
    parser.add_argument("--eps-L", type=float, default=0.2)
    parser.add_argument("--eps-U", type=float, default=0.4)
    parser.add_argument("--p-base", type=float, default=1.0)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument("--p0-method", type=str, choices=["controlled", "baseline_low", "baseline_high", "baseline_mid"], default="baseline_mid")
    parser.add_argument("--outer-p0", type=str, choices=["fixed", "coordsrch"], default="coordsrch")
    parser.add_argument("--n-grid", type=int, default=2001)
    parser.add_argument("--outer-search-grid", type=int, default=1001)
    parser.add_argument("--outer-max-iters", type=int, default=3)
    parser.add_argument("--outer-starts", type=int, default=4)
    parser.add_argument("--outer-coord-grid", type=int, default=3)
    parser.add_argument("--outer-seed", type=int, default=1974)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument(
        "--plot-state-shadow-map",
        action="store_true",
        help="For one scenario, save stacked upper/lower state-average shadow-price maps.",
    )
    parser.add_argument(
        "--map-scenario-id",
        type=str,
        default="known_exact",
        help=(
            "Scenario id(s) used for --plot-state-shadow-map. "
            "Use comma-separated ids or '*' for all selected scenarios."
        ),
    )
    args = parser.parse_args()

    # Backward-compatible default: if neither switch is passed, run both.
    if not args.run_state and not args.run_status:
        args.run_state = True
        args.run_status = True

    if args.run_status and args.adding_up == "state" and args.outer_p0 == "coordsrch":
        raise ValueError(
            "Unsupported configuration: state-by-state adding-up with interval-anchor "
            "search (--outer-p0 coordsrch) has been removed. Use --adding-up national, "
            "or keep --adding-up state with --outer-p0 fixed."
        )

    if args.out_dir is None:
        out_dir = REPO_ROOT / "output/scenario_baseline_options"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_state_rationing(REPO_ROOT / "data")
    base_df = _load_baseline_data(REPO_ROOT)

    all_scenarios = _default_scenarios()
    scenarios = _parse_scenario_filter(args.scenarios, all_scenarios)

    station_total = float(np.sum([r.stations for r in base_rows]))
    records: List[Dict[str, object]] = []
    scenario_meta: Dict[str, object] = {
        "params": {
            "shortage": float(args.shortage),
            "p_control": float(args.p_control),
            "eps_open": float(args.eps_open),
            "eps_L": float(args.eps_L),
            "eps_U": float(args.eps_U),
            "p_base": float(args.p_base),
            "q_max": float(args.q_max),
            "outer_p0": str(args.outer_p0),
            "p0_method": str(args.p0_method),
            "adding_up": str(args.adding_up),
            "n_grid": int(args.n_grid),
            "outer_search_grid": int(args.outer_search_grid),
            "outer_max_iters": int(args.outer_max_iters),
            "outer_starts": int(args.outer_starts),
            "outer_coord_grid": int(args.outer_coord_grid),
            "outer_seed": int(args.outer_seed),
            "outer_tol": float(args.outer_tol),
        },
        "scenarios": {},
    }

    map_ids_raw = str(args.map_scenario_id).strip()
    if map_ids_raw == "*":
        map_ids = {s.scenario_id for s in scenarios}
    else:
        map_ids = {s.strip() for s in map_ids_raw.split(",") if s.strip()}

    for scen in scenarios:
        print(f"\n=== Scenario: {scen.scenario_id} ===")
        row: Dict[str, object] = {
            "scenario_id": scen.scenario_id,
            "scenario_label": scen.label,
            "mode": scen.mode,
            "drop_indiana": bool(scen.drop_indiana),
            "band": scen.band or "",
            "point": scen.point or "",
            "status": "ok",
            "error": "",
        }
        try:
            rows_s, mass_meta = _scenario_mass_map(rows=base_rows, baseline_df=base_df, scenario=scen)
            scenario_meta["scenarios"][scen.scenario_id] = mass_meta

            row["n_states"] = int(len(rows_s))
            row["imputed_states_n"] = int(len(mass_meta.get("imputed_states", [])))
            row["known_states_n"] = int(mass_meta.get("known_states", 0))
            row["p_lo"] = float(mass_meta["p_lo"]) if "p_lo" in mass_meta else np.nan
            row["p_hi"] = float(mass_meta["p_hi"]) if "p_hi" in mass_meta else np.nan
            row["p_use"] = float(mass_meta["p_use"]) if "p_use" in mass_meta else np.nan

            original_station_sum = float(np.sum([r.stations for r in base_rows if any(rr.state == r.state for rr in rows_s)]))
            row["station_share_of_48"] = original_station_sum / station_total if station_total > 0 else np.nan

            if args.run_state:
                st = _run_state_bounds(
                    rows=rows_s,
                    shortage=args.shortage,
                    p_control=args.p_control,
                    eps_open=args.eps_open,
                    p_base=args.p_base,
                    eps_L=args.eps_L,
                    eps_U=args.eps_U,
                    q_max=args.q_max,
                    outer_p0=args.outer_p0,
                    p0_method=args.p0_method,
                    n_grid=args.n_grid,
                    outer_search_grid=args.outer_search_grid,
                    outer_max_iters=args.outer_max_iters,
                    outer_coord_grid=args.outer_coord_grid,
                    outer_starts=args.outer_starts,
                    outer_seed=args.outer_seed,
                    outer_tol=args.outer_tol,
                )
                row.update(
                    {
                        "state_phi_lower_pct": st["phi_lower_pct"],
                        "state_phi_upper_pct": st["phi_upper_pct"],
                        "state_ratio_lower": st["ratio_lower"],
                        "state_ratio_upper": st["ratio_upper"],
                        "state_p_star_lower": st["p_star_lower"],
                        "state_p_star_upper": st["p_star_upper"],
                    }
                )

            if args.run_status:
                want_map = bool(args.plot_state_shadow_map and scen.scenario_id in map_ids)
                sx = _run_state_status_bounds(
                    rows=rows_s,
                    shortage=args.shortage,
                    p_control=args.p_control,
                    eps_open=args.eps_open,
                    p_base=args.p_base,
                    eps_L=args.eps_L,
                    eps_U=args.eps_U,
                    q_max=args.q_max,
                    outer_p0=args.outer_p0,
                    p0_method=args.p0_method,
                    adding_up=args.adding_up,
                    n_grid=args.n_grid,
                    outer_search_grid=args.outer_search_grid,
                    outer_max_iters=args.outer_max_iters,
                    outer_coord_grid=args.outer_coord_grid,
                    outer_starts=args.outer_starts,
                    outer_seed=args.outer_seed,
                    outer_tol=args.outer_tol,
                    include_state_shadow_map_data=want_map,
                )
                row.update(
                    {
                        "status_phi_lower_pct": sx["phi_lower_pct"],
                        "status_phi_upper_pct": sx["phi_upper_pct"],
                        "status_ratio_lower": sx["ratio_lower"],
                        "status_ratio_upper": sx["ratio_upper"],
                        "status_p_star_lower": sx["p_star_lower"],
                        "status_p_star_upper": sx["p_star_upper"],
                        "status_n_markets_active": int(sx["n_markets_active"]),
                    }
                )
                if want_map:
                    map_df = sx.get("state_shadow_map_df")
                    if isinstance(map_df, pd.DataFrame) and not map_df.empty:
                        map_stem = f"figure_state_avg_shadow_map_{scen.scenario_id}"
                        pdf_path, png_path = _plot_state_avg_shadow_map(
                            map_df,
                            p_base=args.p_base,
                            out_dir=out_dir,
                            filename_stem=map_stem,
                            title="State-Average Shadow Prices at Observed Allocation",
                        )
                        csv_path = out_dir / f"table_state_avg_shadow_map_{scen.scenario_id}.csv"
                        map_df.to_csv(csv_path, index=False)
                        row["status_map_csv"] = str(csv_path)
                        row["status_map_pdf"] = str(pdf_path) if pdf_path is not None else ""
                        row["status_map_png"] = str(png_path) if png_path is not None else ""
                        scenario_meta["scenarios"].setdefault(scen.scenario_id, {})
                        scenario_meta["scenarios"][scen.scenario_id]["state_avg_shadow_map_csv"] = str(csv_path)
                        scenario_meta["scenarios"][scen.scenario_id]["state_avg_shadow_map_pdf"] = (
                            str(pdf_path) if pdf_path is not None else None
                        )
                        scenario_meta["scenarios"][scen.scenario_id]["state_avg_shadow_map_png"] = (
                            str(png_path) if png_path is not None else None
                        )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"Scenario {scen.scenario_id} failed: {exc}")
        records.append(row)

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values(["status", "scenario_id"]).reset_index(drop=True)

    if args.replication:
        csv_path = out_dir / "table_state_baseline_options.csv"
        out_df.to_csv(csv_path, index=False)

        meta_path = out_dir / "state_baseline_options_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(scenario_meta, f, indent=2, sort_keys=True)

        print(f"\nWrote: {csv_path}")
        print(f"Wrote: {meta_path}")

    print("\nSummary:")
    show_cols = [c for c in [
        "scenario_id",
        "status",
        "n_states",
        "imputed_states_n",
        "state_ratio_lower",
        "state_ratio_upper",
        "status_ratio_lower",
        "status_ratio_upper",
        "error",
    ] if c in out_df.columns]
    print(out_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
