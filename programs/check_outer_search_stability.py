"""
Audit sensitivity of the interval-anchor outer search to budget and seed choices.

This script is not part of the paper pipeline. It is a numerical check for the
heuristic p0 search used in the state-by-status robust bounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from run_state_baseline_options import (
    SEARCH_PRESETS,
    _default_scenarios,
    _load_baseline_data,
    _run_state_status_bounds,
    _scenario_mass_map,
)
from state_joint_robust_bounds import REPO_ROOT, load_state_rationing


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_seed_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check stability of interval-anchor outer-search results.")
    parser.add_argument("--scenario-id", type=str, default="impute_full_mid")
    parser.add_argument("--search-presets", type=str, default="quick,paper")
    parser.add_argument("--seeds", type=str, default="1974,1975,1976")
    parser.add_argument("--shortage", type=float, default=0.09)
    parser.add_argument("--p-control", type=float, default=0.8)
    parser.add_argument("--eps-open", type=float, default=0.3)
    parser.add_argument("--eps-L", type=float, default=0.2)
    parser.add_argument("--eps-U", type=float, default=0.4)
    parser.add_argument("--p-base", type=float, default=1.0)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument("--p0-method", type=str, choices=["controlled", "baseline_low", "baseline_high", "baseline_mid"], default="baseline_mid")
    parser.add_argument("--adding-up", type=str, choices=["national", "state"], default="national")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    preset_ids = _parse_csv_list(args.search_presets)
    unknown = [p for p in preset_ids if p not in SEARCH_PRESETS]
    if unknown:
        raise ValueError(f"Unknown preset ids: {unknown}")

    seeds = _parse_seed_list(args.seeds)
    if not seeds:
        raise ValueError("Need at least one seed.")

    scenarios = {s.scenario_id: s for s in _default_scenarios()}
    if args.scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario_id={args.scenario_id!r}. Known ids: {sorted(scenarios)}")
    scenario = scenarios[args.scenario_id]

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (REPO_ROOT / "output/search_stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_state_rationing(REPO_ROOT / "data")
    baseline_df = _load_baseline_data(REPO_ROOT)
    rows_s, mass_meta = _scenario_mass_map(rows=base_rows, baseline_df=baseline_df, scenario=scenario)

    records: List[Dict[str, object]] = []
    for preset_id in preset_ids:
        budget = SEARCH_PRESETS[preset_id]
        for seed in seeds:
            out = _run_state_status_bounds(
                rows=rows_s,
                shortage=args.shortage,
                p_control=args.p_control,
                eps_open=args.eps_open,
                p_base=args.p_base,
                eps_L=args.eps_L,
                eps_U=args.eps_U,
                q_max=args.q_max,
                outer_p0="coordsrch",
                p0_method=args.p0_method,
                adding_up=args.adding_up,
                n_grid=budget["n_grid"],
                outer_search_grid=budget["outer_search_grid"],
                outer_max_iters=budget["outer_max_iters"],
                outer_coord_grid=budget["outer_coord_grid"],
                outer_starts=budget["outer_starts"],
                outer_seed=seed,
                outer_tol=1e-6,
                include_state_shadow_map_data=False,
            )
            records.append(
                {
                    "scenario_id": args.scenario_id,
                    "preset": preset_id,
                    "seed": seed,
                    "n_grid": budget["n_grid"],
                    "outer_search_grid": budget["outer_search_grid"],
                    "outer_max_iters": budget["outer_max_iters"],
                    "outer_starts": budget["outer_starts"],
                    "outer_coord_grid": budget["outer_coord_grid"],
                    "phi_lower_pct": float(out["phi_lower_pct"]),
                    "phi_upper_pct": float(out["phi_upper_pct"]),
                    "ratio_lower": float(out["ratio_lower"]),
                    "ratio_upper": float(out["ratio_upper"]),
                    "p_star_lower": float(out["p_star_lower"]),
                    "p_star_upper": float(out["p_star_upper"]),
                    "n_markets_active": int(out["n_markets_active"]),
                }
            )

    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / f"table_outer_search_stability_{args.scenario_id}.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("preset", as_index=False)
        .agg(
            runs=("seed", "count"),
            phi_lower_min=("phi_lower_pct", "min"),
            phi_lower_max=("phi_lower_pct", "max"),
            phi_upper_min=("phi_upper_pct", "min"),
            phi_upper_max=("phi_upper_pct", "max"),
            ratio_lower_min=("ratio_lower", "min"),
            ratio_lower_max=("ratio_lower", "max"),
            ratio_upper_min=("ratio_upper", "min"),
            ratio_upper_max=("ratio_upper", "max"),
        )
    )
    summary_path = out_dir / f"table_outer_search_stability_summary_{args.scenario_id}.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "scenario_id": args.scenario_id,
        "scenario_label": scenario.label,
        "scenario_mass_meta": mass_meta,
        "search_presets": preset_ids,
        "seeds": seeds,
        "adding_up": args.adding_up,
        "outputs": {
            "runs_csv": str(csv_path),
            "summary_csv": str(summary_path),
        },
    }
    meta_path = out_dir / f"outer_search_stability_{args.scenario_id}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
