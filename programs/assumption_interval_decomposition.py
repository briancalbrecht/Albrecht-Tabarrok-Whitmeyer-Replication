"""
Assumption-to-interval decomposition for robust misallocation bounds.

This utility (no paper edits) reports how the welfare
bound interval changes as restrictions are layered:

1) Slope bounds with fixed anchors (midpoint p0 in each market),
2) + Anchor uncertainty (optimize over p0 intervals),
3) + Choke constraint (P(0) <= M) on top of (2).

Default data setup follows the current main empirical specification:
  - state-by-status markets,
  - 48-state sample with imputed missing gallon states at full-band midpoint
    (scenario id: impute_full_mid),
  - national adding-up.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["KMP_USE_SHM"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from run_state_baseline_options import (
    _default_scenarios,
    _load_baseline_data,
    _scenario_mass_map,
)
from state_joint_robust_bounds import (
    MarketSpec,
    REPO_ROOT,
    optimize_joint_bound_over_p0,
    solve_joint_bound,
)
from state_status_joint_robust_bounds import (
    build_state_status_markets,
    load_state_rationing,
    solve_joint_bound_grouped,
)


def _solve_fixed_slope_analytical(markets: Sequence[MarketSpec], Qbar: float) -> Dict[str, float]:
    """Exact Phi when each market has a fixed slope (g_L == g_U, possibly varying across markets).

    With fixed slopes g_i (no slope uncertainty), the LP has a unique solution.
    The efficient shadow price p* equalizes marginal values subject to adding-up:

        sum_i q_i*(p*) = Qbar  where  q_i*(p*) = q0_i + (p* - p0_i) / g_i

    Since sum_i q0_i = Qbar by construction, this reduces to sum_i (p* - p0_i)/g_i = 0,
    giving the weighted mean:

        p* = sum_i(p0_i / |g_i|) / sum_i(1 / |g_i|)

    Misallocation: Phi = sum_i (p0_i - p*)^2 / (2 * |g_i|)
    """
    p0 = np.array([m.p0 for m in markets], dtype=float)
    abs_g = np.array([abs(m.g_L) for m in markets], dtype=float)  # market-specific |g_i|

    weights = 1.0 / abs_g  # weight = 1/|g_i|
    p_star = float(np.dot(p0, weights) / np.sum(weights))
    Phi = float(np.sum((p0 - p_star) ** 2 * weights) / 2.0)
    return {"Phi": Phi, "p_star": p_star}


def _clone_markets_with_choke(markets: Sequence[MarketSpec], choke: float) -> List[MarketSpec]:
    return [
        MarketSpec(
            name=m.name,
            q0=m.q0,
            p0=m.p0,
            g_L=m.g_L,
            g_U=m.g_U,
            q_max=m.q_max,
            M=float(choke),
        )
        for m in markets
    ]


def _state_group_targets(markets: Sequence[MarketSpec]) -> Tuple[List[str], Dict[str, float]]:
    labels: List[str] = []
    targets: Dict[str, float] = {}
    for m in markets:
        state = str(m.name).split("|", 1)[0]
        labels.append(state)
        targets[state] = targets.get(state, 0.0) + float(m.q0)
    return labels, targets


def _solve_case(
    *,
    markets: Sequence[MarketSpec],
    meta: Dict,
    adding_up: str,
    anchor_mode: str,  # fixed | interval
    p_control: float,
    p_base: float,
    n_grid: int,
    outer_search_grid: int,
    outer_max_iters: int,
    outer_starts: int,
    outer_coord_grid: int,
    outer_seed: int,
    outer_tol: float,
) -> Dict[str, float]:
    p_max_grid = max(10.0, p_control, p_base, max((m.M for m in markets if np.isfinite(m.M)), default=0.0))
    Qbar = float(meta["Qbar"])

    if anchor_mode not in {"fixed", "interval"}:
        raise ValueError(f"Unknown anchor_mode={anchor_mode!r}.")

    if anchor_mode == "fixed":
        if adding_up == "state":
            group_labels, group_targets = _state_group_targets(markets)
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
            upper = solve_joint_bound(
                markets,
                Qbar=Qbar,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
            )
            lower = solve_joint_bound(
                markets,
                Qbar=Qbar,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=n_grid,
            )
    else:
        if adding_up == "state":
            raise ValueError(
                "Interval-anchor outer optimization is currently supported only with national adding-up."
            )
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

    return {
        "phi_lower": float(lower["Phi"]),
        "phi_upper": float(upper["Phi"]),
        "p_star_lower": float(lower["p_star"]),
        "p_star_upper": float(upper["p_star"]),
    }


def _format_range(lo: float, hi: float, ndigits: int = 2) -> str:
    return f"[{lo:.{ndigits}f}, {hi:.{ndigits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute an assumption-layer decomposition table for robust bounds.")
    parser.add_argument("--scenario-id", type=str, default="impute_full_mid", help="Baseline-mass scenario id.")
    parser.add_argument("--adding-up", type=str, choices=["national", "state"], default="national")
    parser.add_argument("--shortage", type=float, default=0.09)
    parser.add_argument("--p-control", type=float, default=0.8)
    parser.add_argument("--eps-open", type=float, default=0.3)
    parser.add_argument("--eps-L", type=float, default=0.2)
    parser.add_argument("--eps-U", type=float, default=0.4)
    parser.add_argument("--p-base", type=float, default=1.0)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument("--p0-method", type=str, choices=["controlled", "baseline_low", "baseline_high", "baseline_mid"], default="baseline_mid")
    parser.add_argument("--choke", type=float, default=4.0)
    parser.add_argument("--n-grid", type=int, default=8001)
    parser.add_argument("--outer-search-grid", type=int, default=4001)
    parser.add_argument("--outer-max-iters", type=int, default=10)
    parser.add_argument("--outer-starts", type=int, default=12)
    parser.add_argument("--outer-coord-grid", type=int, default=5)
    parser.add_argument("--outer-seed", type=int, default=1974)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (REPO_ROOT / "output/assumption_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _default_scenarios()
    by_id = {s.scenario_id: s for s in scenarios}
    if args.scenario_id not in by_id:
        raise ValueError(f"Unknown --scenario-id={args.scenario_id!r}. Known ids: {sorted(by_id)}")
    scenario = by_id[args.scenario_id]

    base_rows = load_state_rationing(REPO_ROOT / "data")
    baseline_df = _load_baseline_data(REPO_ROOT)
    rows_s, mass_meta = _scenario_mass_map(rows=base_rows, baseline_df=baseline_df, scenario=scenario)

    markets, meta, _shadow_df = build_state_status_markets(
        list(rows_s),
        shortage=args.shortage,
        p_control=args.p_control,
        eps_open=args.eps_open,
        p_base=args.p_base,
        eps_L=args.eps_L,
        eps_U=args.eps_U,
        q_max=args.q_max,
        p0_method=args.p0_method,
    )

    # Common Harberger benchmark used to compute R = L_Mis / L_Harb.
    g_steep = -args.p_base / args.eps_L
    psi_ref = 0.5 * abs(g_steep) * (float(meta["shortage"]) ** 2)

    shortage_sq = float(meta["shortage"]) ** 2

    rows_out: List[Dict[str, object]] = []

    # ---- Row 1: Common epsilon (matched Psi, R invariant to slope) ----
    # Rebuild markets at each epsilon endpoint with a common (shared) slope.
    # With eps_L = eps_U, bounds collapse to a single point; R = Phi/Psi(eps)
    # is invariant across epsilon values (slope cancels in the ratio).
    print("Solving case: common_epsilon | anchor=fixed | choke=none")
    # With a common epsilon (g_L=g_U), the LP has kappa=0 and a unique
    # solution — use the exact analytical formula instead of the grid solver.
    markets_elastic, meta_elastic, _ = build_state_status_markets(
        list(rows_s),
        shortage=args.shortage,
        p_control=args.p_control,
        eps_open=args.eps_open,
        p_base=args.p_base,
        eps_L=args.eps_U,
        eps_U=args.eps_U,
        q_max=args.q_max,
        p0_method=args.p0_method,
    )
    markets_inelastic, meta_inelastic, _ = build_state_status_markets(
        list(rows_s),
        shortage=args.shortage,
        p_control=args.p_control,
        eps_open=args.eps_open,
        p_base=args.p_base,
        eps_L=args.eps_L,
        eps_U=args.eps_L,
        q_max=args.q_max,
        p0_method=args.p0_method,
    )
    sol_elastic = _solve_fixed_slope_analytical(markets_elastic, float(meta_elastic["Qbar"]))
    sol_inelastic = _solve_fixed_slope_analytical(markets_inelastic, float(meta_inelastic["Qbar"]))
    # Phi_lower: elastic endpoint (eps_U, flattest slope → smallest Phi)
    # Phi_upper: inelastic endpoint (eps_L, steepest slope → largest Phi)
    ce_phi_lower = sol_elastic["Phi"]
    ce_phi_upper = sol_inelastic["Phi"]
    # Matched Psi: same eps as Phi
    psi_elastic = 0.5 * abs(-args.p_base / args.eps_U) * shortage_sq
    psi_inelastic = 0.5 * abs(-args.p_base / args.eps_L) * shortage_sq
    ce_r = ce_phi_lower / psi_elastic  # invariant; equals ce_phi_upper / psi_inelastic
    ce_phi_lower_pct = 100.0 * ce_phi_lower
    ce_phi_upper_pct = 100.0 * ce_phi_upper
    rows_out.append(
        {
            "case_id": "common_epsilon",
            "assumptions": f"Common epsilon in [{args.eps_L:g}, {args.eps_U:g}] (fixed anchors)",
            "anchor_mode": "fixed",
            "choke_M": float("inf"),
            "phi_lower_pct": ce_phi_lower_pct,
            "phi_upper_pct": ce_phi_upper_pct,
            "ratio_lower": ce_r,
            "ratio_upper": ce_r,
            "ratio_range": f"{ce_r:.2f}",
            "p_star_lower": sol_elastic["p_star"],
            "p_star_upper": sol_inelastic["p_star"],
        }
    )
    print(
        f"  done: Phi in [{ce_phi_lower_pct:.3f}%, {ce_phi_upper_pct:.3f}%], R = {ce_r:.3f}"
    )

    cases = [
        {
            "case_id": "slope_fixed_anchor",
            "assumptions": "Slope bounds only (fixed midpoint anchor)",
            "anchor_mode": "fixed",
            "choke": float("inf"),
        },
        {
            "case_id": "plus_anchor_uncertainty",
            "assumptions": "+ Anchor uncertainty (p0 intervals)",
            "anchor_mode": "interval",
            "choke": float("inf"),
        },
        {
            "case_id": "plus_choke",
            "assumptions": f"+ Choke constraint (M={args.choke:g})",
            "anchor_mode": "interval",
            "choke": float(args.choke),
        },
    ]

    for c in cases:
        choke_label = "none" if not np.isfinite(c["choke"]) else f"{c['choke']:.3g}"
        print(
            f"Solving case: {c['case_id']} | anchor={c['anchor_mode']} | choke={choke_label}"
        )
        case_markets = _clone_markets_with_choke(markets, c["choke"])
        sol = _solve_case(
            markets=case_markets,
            meta=meta,
            adding_up=args.adding_up,
            anchor_mode=c["anchor_mode"],
            p_control=args.p_control,
            p_base=args.p_base,
            n_grid=args.n_grid,
            outer_search_grid=args.outer_search_grid,
            outer_max_iters=args.outer_max_iters,
            outer_starts=args.outer_starts,
            outer_coord_grid=args.outer_coord_grid,
            outer_seed=args.outer_seed,
            outer_tol=args.outer_tol,
        )

        phi_lo_pct = 100.0 * sol["phi_lower"]
        phi_hi_pct = 100.0 * sol["phi_upper"]
        r_lo = sol["phi_lower"] / psi_ref if psi_ref > 0 else float("nan")
        r_hi = sol["phi_upper"] / psi_ref if psi_ref > 0 else float("nan")

        rows_out.append(
            {
                "case_id": c["case_id"],
                "assumptions": c["assumptions"],
                "anchor_mode": c["anchor_mode"],
                "choke_M": c["choke"],
                "phi_lower_pct": phi_lo_pct,
                "phi_upper_pct": phi_hi_pct,
                "ratio_lower": r_lo,
                "ratio_upper": r_hi,
                "ratio_range": _format_range(r_lo, r_hi, ndigits=2),
                "p_star_lower": sol["p_star_lower"],
                "p_star_upper": sol["p_star_upper"],
            }
        )
        print(
            f"  done: Phi in [{phi_lo_pct:.3f}%, {phi_hi_pct:.3f}%], "
            f"R in {_format_range(r_lo, r_hi, ndigits=3)}"
        )

    table = pd.DataFrame(rows_out)

    scenario_tag = args.scenario_id
    csv_path = out_dir / f"table_assumption_interval_decomposition_{scenario_tag}.csv"
    json_path = out_dir / f"assumption_interval_decomposition_{scenario_tag}_meta.json"
    table.to_csv(csv_path, index=False)

    try:
        table_csv_rel = str(csv_path.relative_to(REPO_ROOT))
    except ValueError:
        table_csv_rel = str(csv_path)

    meta_out = {
        "scenario_id": scenario_tag,
        "scenario_label": scenario.label,
        "scenario_mass_meta": mass_meta,
        "params": {
            "adding_up": args.adding_up,
            "shortage": float(args.shortage),
            "p_control": float(args.p_control),
            "eps_open": float(args.eps_open),
            "eps_L": float(args.eps_L),
            "eps_U": float(args.eps_U),
            "p_base": float(args.p_base),
            "q_max": float(args.q_max),
            "p0_method": args.p0_method,
            "choke": float(args.choke),
            "n_grid": int(args.n_grid),
            "outer_search_grid": int(args.outer_search_grid),
            "outer_max_iters": int(args.outer_max_iters),
            "outer_starts": int(args.outer_starts),
            "outer_coord_grid": int(args.outer_coord_grid),
            "outer_seed": int(args.outer_seed),
            "outer_tol": float(args.outer_tol),
        },
        "calibration": {
            "n_states": int(meta["n_states"]),
            "n_markets_active": int(meta["n_markets_active"]),
            "Qbar": float(meta["Qbar"]),
            "shortage": float(meta["shortage"]),
            "q_open": float(meta["q_open"]),
            "q_non_open": float(meta["q_non_open"]),
        },
        "harberger": {
            "psi_ref": float(psi_ref),
            "psi_ref_pct": float(100.0 * psi_ref),
            "g_steep": float(g_steep),
        },
        "outputs": {
            "table_csv": table_csv_rel,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print("\nAssumption-to-Interval Decomposition")
    print(table[["assumptions", "phi_lower_pct", "phi_upper_pct", "ratio_range"]].to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
