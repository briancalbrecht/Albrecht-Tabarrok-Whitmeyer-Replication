"""
State x Status (48x2) joint robust bounds (Lemma 1 with adding-up).

This script splits each state into two markets:
  - open stations in state i
  - non-open (limiting + out-of-fuel) stations in state i

Per-station quantities are common across states:
  q_open: demand at controlled price from baseline-anchored linear demand
  q_non_open: residual from national adding-up to hit Qbar = 1 - shortage

State heterogeneity enters through:
  - baseline station share w_i
  - state rationing share r_i

Observed aggregate quantities by cell:
  q_obs_{i,open}    = w_i * (1-r_i) * q_open
  q_obs_{i,nonopen} = w_i * r_i     * q_non_open

Baseline aggregate anchors by cell:
  q_base_{i,open}    = w_i * (1-r_i)
  q_base_{i,nonopen} = w_i * r_i

Then we run the same Lemma-1 joint robust program with optional outer search
over unknown anchor prices p0_i in their admissible intervals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

os.environ["KMP_USE_SHM"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from state_joint_robust_bounds import (
    THIS_DIR,
    REPO_ROOT,
    MarketSpec,
    StateRow,
    _inner_x_many,
    _precompute_market,
    linear_open_quantity,
    load_state_rationing,
    optimize_joint_bound_over_p0,
    p0_bounds_from_baseline,
    solve_joint_bound,
)
from state_subset import resolve_state_subset

Mode = Literal["upper", "lower"]


def _markets_with_p0(markets: Sequence[MarketSpec], p0: Sequence[float]) -> List[MarketSpec]:
    return [
        MarketSpec(
            name=m.name,
            q0=m.q0,
            p0=float(p0i),
            g_L=m.g_L,
            g_U=m.g_U,
            q_max=m.q_max,
            M=m.M,
        )
        for m, p0i in zip(markets, p0)
    ]


def build_state_status_markets(
    rows: List[StateRow],
    *,
    shortage: float,
    p_control: float,
    eps_open: float,
    p_base: float,
    eps_L: float,
    eps_U: float,
    q_max: float,
    p0_method: str,
    min_mass: float = 1e-12,
) -> Tuple[List[MarketSpec], Dict, pd.DataFrame]:
    stations = np.array([r.stations for r in rows], dtype=float)
    weights = stations / float(np.sum(stations))
    rationing = np.array([r.rationing for r in rows], dtype=float)

    q_open = linear_open_quantity(eps_open=eps_open, p_control=p_control, p_base=1.0, q_base=1.0)
    Q_target = 1.0 - shortage
    rbar = float(np.sum(weights * rationing))
    if rbar <= 1e-12:
        q_non_open = Q_target
        q_open = Q_target
    else:
        q_non_open = (Q_target - (1.0 - rbar) * q_open) / rbar
        if q_non_open < 0.0:
            q_non_open = 0.0
            q_open = Q_target / (1.0 - rbar)

    records: List[Dict] = []
    markets: List[MarketSpec] = []
    p0_init_vec: List[float] = []
    p0_lo_vec: List[float] = []
    p0_hi_vec: List[float] = []
    n_zero_mass_cells = 0

    for i, row in enumerate(rows):
        wi = float(weights[i])
        ri = float(rationing[i])
        for status, share_status, q_status in (
            ("open", max(0.0, 1.0 - ri), q_open),
            ("nonopen", max(0.0, ri), q_non_open),
        ):
            q_base = wi * share_status
            if q_base <= min_mass:
                n_zero_mass_cells += 1
                continue

            q_obs = q_base * q_status
            g_L = -p_base / (eps_L * q_base)
            g_U = -p_base / (eps_U * q_base)

            p0_lo, p0_hi = p0_bounds_from_baseline(
                q_base=q_base,
                q_obs=q_obs,
                p_base=p_base,
                g_L=g_L,
                g_U=g_U,
            )

            if p0_method == "controlled":
                p0 = p_control
            elif p0_method == "baseline_low":
                p0 = p0_lo
            elif p0_method == "baseline_high":
                p0 = p0_hi
            elif p0_method == "baseline_mid":
                p0 = 0.5 * (p0_lo + p0_hi)
            else:
                raise ValueError(f"Unknown p0_method={p0_method!r}.")

            name = f"{row.state}|{status}"
            markets.append(
                MarketSpec(
                    name=name,
                    q0=float(q_obs),
                    p0=float(p0),
                    g_L=float(g_L),
                    g_U=float(g_U),
                    q_max=float(q_max),
                )
            )
            p0_init_vec.append(float(p0))
            p0_lo_vec.append(float(p0_lo))
            p0_hi_vec.append(float(p0_hi))

            records.append(
                {
                    "state": row.state,
                    "status": status,
                    "weight_state": wi,
                    "share_status": share_status,
                    "q_base": float(q_base),
                    "q_obs": float(q_obs),
                    "q_per_station": float(q_status),
                    "rationing": ri,
                    "p0_lo": float(p0_lo),
                    "p0_mid": float(0.5 * (p0_lo + p0_hi)),
                    "p0_hi": float(p0_hi),
                }
            )

    shadow_df = pd.DataFrame(records)
    if shadow_df.empty:
        raise ValueError("No state-status markets were created (all zero mass).")

    Qbar = float(shadow_df["q_obs"].sum())
    meta = {
        "n_states": int(len(rows)),
        "n_markets_active": int(len(markets)),
        "n_zero_mass_cells": int(n_zero_mass_cells),
        "Qbar": Qbar,
        "shortage": 1.0 - Qbar,
        "q_open": float(q_open),
        "q_non_open": float(q_non_open),
        "rbar": float(rbar),
        "p0_method": p0_method,
        "p0_min": float(np.min(p0_init_vec)),
        "p0_max": float(np.max(p0_init_vec)),
        "p0_bounds_min": float(np.min(p0_lo_vec)),
        "p0_bounds_max": float(np.max(p0_hi_vec)),
        "p0_lo_vec": [float(v) for v in p0_lo_vec],
        "p0_hi_vec": [float(v) for v in p0_hi_vec],
        "p0_init_vec": [float(v) for v in p0_init_vec],
    }
    return markets, meta, shadow_df


def _is_better(candidate: float, incumbent: Optional[float], mode: Mode, tol: float) -> bool:
    if incumbent is None:
        return True
    if mode == "upper":
        return candidate > incumbent + tol
    return candidate < incumbent - tol


def _coordinate_candidates(lo: float, hi: float, current: float, n_points: int) -> np.ndarray:
    if n_points <= 2:
        pts = np.array([lo, hi], dtype=float)
    else:
        pts = np.linspace(lo, hi, n_points)
    pts = np.append(pts, current)
    return np.unique(np.clip(pts, lo, hi))


def _build_group_structure(
    *,
    group_labels: Sequence[str],
    group_targets: Dict[str, float],
) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
    if len(group_labels) == 0:
        raise ValueError("No groups: empty market list.")
    idx_map: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, g in enumerate(group_labels):
        gg = str(g)
        if gg not in idx_map:
            idx_map[gg] = []
            order.append(gg)
        idx_map[gg].append(i)

    missing = [g for g in order if g not in group_targets]
    if missing:
        raise ValueError(f"Missing group targets for: {missing[:5]}")

    idx_arrays = [np.array(idx_map[g], dtype=int) for g in order]
    targets = np.array([float(group_targets[g]) for g in order], dtype=float)
    return order, idx_arrays, targets


def solve_joint_bound_grouped(
    markets: Sequence[MarketSpec],
    *,
    group_labels: Sequence[str],
    group_targets: Dict[str, float],
    mode: Mode,
    p_min: float = 0.0,
    p_max: Optional[float] = None,
    n_grid: int = 4001,
) -> Dict:
    """
    Joint robust bound with grouped adding-up constraints.

    Compared with the national solver (single Σ_i x_i = Qbar), this enforces
    Σ_{j in g} x_j = Q_g for each group g (here: each state). This is the
    "state aggregates binding" variant for state x status markets.
    """
    ms = list(markets)
    if len(ms) < 2:
        raise ValueError("Need at least 2 markets for a joint bound.")
    if len(group_labels) != len(ms):
        raise ValueError("group_labels length must match number of markets.")

    grp_names, grp_idx, grp_Q = _build_group_structure(
        group_labels=group_labels,
        group_targets=group_targets,
    )
    Qbar_total = float(np.sum(grp_Q))

    if p_max is None:
        finite_chokes = [m.M for m in ms if np.isfinite(m.M)]
        p_max = max([10.0, *(m.p0 for m in ms), *finite_chokes])
    p_max = float(p_max)

    if not (p_max > p_min):
        raise ValueError("Need p_max > p_min.")
    if n_grid < 101:
        raise ValueError("Need n_grid >= 101.")

    p_grid = np.linspace(p_min, p_max, n_grid)
    names = [m.name for m in ms]
    q0 = np.array([m.q0 for m in ms], dtype=float)
    p0 = np.array([m.p0 for m in ms], dtype=float)
    kappa = np.array([m.kappa for m in ms], dtype=float)
    q0p0_const = float(np.sum(q0 * p0))

    n_mk = len(ms)
    n_p = len(p_grid)
    ell_vals = np.empty((n_mk, n_p), dtype=float)
    u_vals = np.empty((n_mk, n_p), dtype=float)
    int_ell = np.empty((n_mk, n_p), dtype=float)
    int_u = np.empty((n_mk, n_p), dtype=float)

    for i, m in enumerate(ms):
        env = _precompute_market(m, p_grid)
        ell_vals[i, :] = env.ell_vals
        u_vals[i, :] = env.u_vals
        ell0 = float(np.interp(m.p0, p_grid, env.ell_cum))
        u0 = float(np.interp(m.p0, p_grid, env.u_cum))
        int_ell[i, :] = env.ell_cum - ell0
        int_u[i, :] = env.u_cum - u0

    feasible = np.ones(n_p, dtype=bool)
    for idx, qg in zip(grp_idx, grp_Q):
        Lg = np.sum(ell_vals[idx, :], axis=0)
        Ug = np.sum(u_vals[idx, :], axis=0)
        feasible &= (Lg <= qg + 1e-10) & (Ug >= qg - 1e-10)
    if not np.any(feasible):
        raise ValueError("No feasible p found under grouped adding-up constraints.")

    obj = np.full_like(p_grid, np.nan, dtype=float)
    x_grid = np.full((n_p, n_mk), np.nan, dtype=float)

    for j, p in enumerate(p_grid):
        if not feasible[j]:
            continue

        forward = p >= p0
        if mode == "upper":
            I = np.where(forward, int_ell[:, j], int_u[:, j])
            c = np.where(forward, ell_vals[:, j], u_vals[:, j])
            sign = -1.0
        else:
            I = np.where(forward, int_u[:, j], int_ell[:, j])
            c = np.where(forward, u_vals[:, j], ell_vals[:, j])
            sign = +1.0

        val = Qbar_total * float(p) - q0p0_const - float(np.sum(I))
        l = ell_vals[:, j]
        u = u_vals[:, j]

        x = np.full(n_mk, np.nan, dtype=float)
        pen = 0.0
        infeasible_group = False

        for idx, qg in zip(grp_idx, grp_Q):
            l_g = l[idx]
            u_g = u[idx]
            c_g = c[idx]
            k_g = kappa[idx]

            if qg < float(np.sum(l_g)) - 1e-10 or qg > float(np.sum(u_g)) + 1e-10:
                infeasible_group = True
                break

            x_g = _inner_x_many(float(qg), l_g, u_g, c_g, k_g)
            x[idx] = x_g

            ok = k_g > 0
            if np.any(ok):
                pen += float(np.sum(((x_g[ok] - c_g[ok]) ** 2) / (2.0 * k_g[ok])))

        if infeasible_group:
            continue

        val += sign * pen
        obj[j] = val
        x_grid[j, :] = x

    if mode == "upper":
        j_star = int(np.nanargmax(obj))
    else:
        j_star = int(np.nanargmin(obj))

    return {
        "mode": mode,
        "Phi": float(obj[j_star]),
        "p_star": float(p_grid[j_star]),
        "market_names": names,
        "x_star": [float(v) for v in x_grid[j_star, :]],
        "p_grid": p_grid,
        "objective_grid": obj,
        "x_grid": x_grid,
        "group_names": grp_names,
        "group_targets": [float(v) for v in grp_Q],
    }


def _solve_joint_bound_grouped_safe(
    markets: Sequence[MarketSpec],
    *,
    group_labels: Sequence[str],
    group_targets: Dict[str, float],
    mode: Mode,
    p_min: float,
    p_max: float,
    n_grid: int,
) -> Optional[Dict]:
    try:
        return solve_joint_bound_grouped(
            markets,
            group_labels=group_labels,
            group_targets=group_targets,
            mode=mode,
            p_min=p_min,
            p_max=p_max,
            n_grid=n_grid,
        )
    except ValueError:
        return None


def _coordinate_search_over_p0_grouped(
    markets_base: Sequence[MarketSpec],
    *,
    p0_lo: Sequence[float],
    p0_hi: Sequence[float],
    start_p0: Sequence[float],
    group_labels: Sequence[str],
    group_targets: Dict[str, float],
    mode: Mode,
    p_min: float,
    p_max: float,
    n_grid_search: int,
    n_grid_final: int,
    max_iters: int,
    coord_grid_points: int,
    tol: float,
) -> Dict:
    lo = np.asarray(p0_lo, dtype=float)
    hi = np.asarray(p0_hi, dtype=float)
    cur = np.asarray(start_p0, dtype=float)

    if not (len(lo) == len(hi) == len(cur) == len(markets_base)):
        raise ValueError("p0 search arrays must match number of markets.")
    if np.any(hi < lo - 1e-12):
        raise ValueError("Invalid p0 bounds: some hi < lo.")
    cur = np.clip(cur, lo, hi)

    n_eval = 0
    n_infeasible = 0

    def evaluate(p0_vec: np.ndarray, n_grid: int) -> Optional[Dict]:
        nonlocal n_eval, n_infeasible
        n_eval += 1
        markets = _markets_with_p0(markets_base, p0_vec)
        out = _solve_joint_bound_grouped_safe(
            markets,
            group_labels=group_labels,
            group_targets=group_targets,
            mode=mode,
            p_min=p_min,
            p_max=p_max,
            n_grid=n_grid,
        )
        if out is None:
            n_infeasible += 1
            return None
        out["p0_vector"] = [float(v) for v in p0_vec]
        return out

    best = evaluate(cur, n_grid_search)
    if best is None:
        for guess in (0.5 * (lo + hi), lo.copy(), hi.copy()):
            cand = evaluate(np.asarray(guess, dtype=float), n_grid_search)
            if cand is not None:
                cur = np.asarray(guess, dtype=float)
                best = cand
                break
    if best is None:
        raise ValueError("No feasible p0 vector found for grouped coordinate-search start.")

    n_iter_done = 0
    for it in range(max_iters):
        improved = False
        n_iter_done = it + 1

        for i in range(len(cur)):
            cands = _coordinate_candidates(lo[i], hi[i], cur[i], coord_grid_points)
            local_best = best
            local_p0 = cur.copy()
            local_val = float(best["Phi"])

            for p0_i in cands:
                if abs(float(p0_i) - float(cur[i])) <= 1e-14:
                    continue
                trial = cur.copy()
                trial[i] = float(p0_i)
                res = evaluate(trial, n_grid_search)
                if res is None:
                    continue
                trial_val = float(res["Phi"])
                if _is_better(trial_val, local_val, mode, tol):
                    local_val = trial_val
                    local_best = res
                    local_p0 = trial

            if _is_better(float(local_best["Phi"]), float(best["Phi"]), mode, tol):
                best = local_best
                cur = local_p0
                improved = True

        if not improved:
            break

    final_eval = evaluate(cur, n_grid_final)
    if final_eval is None:
        final_eval = best
        final_eval["outer_warning"] = "Final-grid grouped evaluation infeasible; using search-grid solution."

    return {
        "result": final_eval,
        "iterations": n_iter_done,
        "eval_count": n_eval,
        "infeasible_count": n_infeasible,
    }


def optimize_joint_bound_over_p0_grouped(
    markets_base: Sequence[MarketSpec],
    *,
    p0_lo: Sequence[float],
    p0_hi: Sequence[float],
    start_p0: Sequence[float],
    group_labels: Sequence[str],
    group_targets: Dict[str, float],
    mode: Mode,
    p_min: float,
    p_max: float,
    n_grid: int,
    search_n_grid: int,
    max_iters: int,
    coord_grid_points: int,
    n_starts: int,
    seed: int,
    tol: float = 1e-6,
) -> Dict:
    lo = np.asarray(p0_lo, dtype=float)
    hi = np.asarray(p0_hi, dtype=float)
    init = np.asarray(start_p0, dtype=float)

    if np.any(hi < lo - 1e-12):
        raise ValueError("Invalid p0 bounds: some hi < lo.")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1.")
    if coord_grid_points < 2:
        raise ValueError("coord_grid_points must be >= 2.")
    if search_n_grid < 101:
        raise ValueError("search_n_grid must be >= 101.")

    mid = 0.5 * (lo + hi)
    deterministic = [
        np.clip(init, lo, hi),
        mid,
        lo.copy(),
        hi.copy(),
    ]

    starts: List[np.ndarray] = []
    seen: set[Tuple[float, ...]] = set()

    def add_start(v: np.ndarray) -> None:
        vv = np.clip(np.asarray(v, dtype=float), lo, hi)
        key = tuple(np.round(vv, 12))
        if key in seen:
            return
        starts.append(vv)
        seen.add(key)

    for v in deterministic:
        add_start(v)
        if len(starts) >= n_starts:
            break

    rng = np.random.default_rng(seed)
    draw_tries = 0
    max_draw_tries = max(20, 20 * n_starts)
    while len(starts) < n_starts and draw_tries < max_draw_tries:
        draw = lo + (hi - lo) * rng.random(len(lo))
        add_start(draw)
        draw_tries += 1

    if len(starts) < n_starts:
        n_starts = len(starts)

    best_outer: Optional[Dict] = None
    all_attempts = 0
    total_evals = 0
    total_infeasible = 0

    for s in starts:
        all_attempts += 1
        try:
            local = _coordinate_search_over_p0_grouped(
                markets_base,
                p0_lo=lo,
                p0_hi=hi,
                start_p0=s,
                group_labels=group_labels,
                group_targets=group_targets,
                mode=mode,
                p_min=p_min,
                p_max=p_max,
                n_grid_search=search_n_grid,
                n_grid_final=n_grid,
                max_iters=max_iters,
                coord_grid_points=coord_grid_points,
                tol=tol,
            )
        except ValueError:
            continue

        total_evals += int(local["eval_count"])
        total_infeasible += int(local["infeasible_count"])

        cand = local["result"]
        if best_outer is None or _is_better(float(cand["Phi"]), float(best_outer["result"]["Phi"]), mode, tol):
            best_outer = local

    if best_outer is None:
        raise ValueError("Grouped outer p0 optimization failed: no feasible start.")

    result = dict(best_outer["result"])
    result["outer_search"] = {
        "mode": mode,
        "n_starts_requested": int(n_starts),
        "n_starts_attempted": int(all_attempts),
        "iterations": int(best_outer["iterations"]),
        "coord_grid_points": int(coord_grid_points),
        "max_iters": int(max_iters),
        "search_n_grid": int(search_n_grid),
        "final_n_grid": int(n_grid),
        "total_evaluations": int(total_evals),
        "total_infeasible": int(total_infeasible),
    }
    return result


def _write_outputs(
    *,
    out_dir: Path,
    bounds_df: pd.DataFrame,
    shadow_df: pd.DataFrame,
    meta: Dict,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    bounds_path = out_dir / "table_state_status_joint_bounds.csv"
    bounds_df.to_csv(bounds_path, index=False)
    paths["bounds_csv"] = str(bounds_path)

    shadow_path = out_dir / "table_state_status_shadow_price_bounds.csv"
    shadow_df.to_csv(shadow_path, index=False)
    paths["shadow_csv"] = str(shadow_path)

    meta_path = out_dir / "state_status_joint_bounds_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    paths["meta_json"] = str(meta_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="State x status (48x2) joint robust bounds with adding-up."
    )
    parser.add_argument("--replication", action="store_true", help="Write outputs to disk.")
    parser.add_argument(
        "--shortage",
        type=float,
        default=0.09,
        help="Aggregate shortage share delta (default 0.09 from Yergin 1991).",
    )
    parser.add_argument("--p-control", type=float, default=0.8)
    parser.add_argument("--eps-open", type=float, default=0.3)
    parser.add_argument("--eps-L", type=float, default=0.2)
    parser.add_argument("--eps-U", type=float, default=0.4)
    parser.add_argument("--p-base", type=float, default=1.0)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument(
        "--adding-up",
        type=str,
        choices=["national", "state"],
        default="national",
        help=(
            "Constraint mode: one national adding-up equation or one per-state equation. "
            "Note: state adding-up is supported only with --outer-p0 fixed."
        ),
    )
    parser.add_argument(
        "--p0-method",
        type=str,
        choices=["controlled", "baseline_low", "baseline_high", "baseline_mid"],
        default="baseline_mid",
    )
    parser.add_argument("--outer-p0", type=str, choices=["fixed", "coordsrch"], default="coordsrch")
    parser.add_argument("--outer-max-iters", type=int, default=10)
    parser.add_argument("--outer-starts", type=int, default=12)
    parser.add_argument("--outer-coord-grid", type=int, default=5)
    parser.add_argument("--outer-search-grid", type=int, default=4001)
    parser.add_argument("--outer-seed", type=int, default=1974)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--n-grid", type=int, default=8001)
    parser.add_argument("--state-subset", type=str, choices=["all", "sales36"], default="all")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.adding_up == "state" and args.outer_p0 == "coordsrch":
        raise ValueError(
            "Unsupported configuration: state-by-state adding-up with interval-anchor "
            "search (--outer-p0 coordsrch) has been removed. Use --adding-up national, "
            "or keep --adding-up state with --outer-p0 fixed."
        )

    include_states, subset_meta = resolve_state_subset(args.state_subset, repo_root=REPO_ROOT)
    rows = load_state_rationing(THIS_DIR.parent / "data", include_states=include_states)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        base_out = THIS_DIR.parent / "output"
        suffix = "state48x2" if args.state_subset == "all" else f"state{args.state_subset}x2"
        if args.adding_up == "state":
            suffix = f"{suffix}_statebind"
        out_dir = base_out / f"scenario_{suffix}"

    markets, meta, shadow_df = build_state_status_markets(
        rows,
        shortage=args.shortage,
        p_control=args.p_control,
        eps_open=args.eps_open,
        p_base=args.p_base,
        eps_L=args.eps_L,
        eps_U=args.eps_U,
        q_max=args.q_max,
        p0_method=args.p0_method,
    )
    Qbar = float(meta["Qbar"])
    p_max_grid = max(10.0, args.p_control, args.p_base)
    market_states = [m.name.split("|", 1)[0] for m in markets]
    state_targets = {
        str(s): float(q)
        for s, q in shadow_df.groupby("state", sort=False)["q_obs"].sum().to_dict().items()
    }

    print(
        f"State subset: {subset_meta['subset_id']} ({subset_meta['subset_label']}); "
        f"states={meta['n_states']}, active markets={meta['n_markets_active']} "
        f"(zero-mass cells dropped={meta['n_zero_mass_cells']})"
    )
    print(f"Adding-up mode: {args.adding_up}")
    print(
        f"Calibration: q_open={meta['q_open']:.3f}, q_non_open={meta['q_non_open']:.3f}, "
        f"Qbar={Qbar:.6f}, shortage={meta['shortage']:.3%}"
    )

    if args.outer_p0 == "coordsrch":
        p0_lo = np.array(meta["p0_lo_vec"], dtype=float)
        p0_hi = np.array(meta["p0_hi_vec"], dtype=float)
        p0_init = np.array(meta["p0_init_vec"], dtype=float)
        search_n_grid = max(101, min(args.n_grid, args.outer_search_grid))

        if args.adding_up == "state":
            upper = optimize_joint_bound_over_p0_grouped(
                markets,
                p0_lo=p0_lo,
                p0_hi=p0_hi,
                start_p0=p0_init,
                group_labels=market_states,
                group_targets=state_targets,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=args.n_grid,
                search_n_grid=search_n_grid,
                max_iters=args.outer_max_iters,
                coord_grid_points=args.outer_coord_grid,
                n_starts=args.outer_starts,
                seed=args.outer_seed,
                tol=args.outer_tol,
            )
            lower = optimize_joint_bound_over_p0_grouped(
                markets,
                p0_lo=p0_lo,
                p0_hi=p0_hi,
                start_p0=p0_init,
                group_labels=market_states,
                group_targets=state_targets,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=args.n_grid,
                search_n_grid=search_n_grid,
                max_iters=args.outer_max_iters,
                coord_grid_points=args.outer_coord_grid,
                n_starts=args.outer_starts,
                seed=args.outer_seed,
                tol=args.outer_tol,
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
                n_grid=args.n_grid,
                search_n_grid=search_n_grid,
                max_iters=args.outer_max_iters,
                coord_grid_points=args.outer_coord_grid,
                n_starts=args.outer_starts,
                seed=args.outer_seed,
                tol=args.outer_tol,
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
                n_grid=args.n_grid,
                search_n_grid=search_n_grid,
                max_iters=args.outer_max_iters,
                coord_grid_points=args.outer_coord_grid,
                n_starts=args.outer_starts,
                seed=args.outer_seed,
                tol=args.outer_tol,
            )
    else:
        if args.adding_up == "state":
            upper = solve_joint_bound_grouped(
                markets,
                group_labels=market_states,
                group_targets=state_targets,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=args.n_grid,
            )
            lower = solve_joint_bound_grouped(
                markets,
                group_labels=market_states,
                group_targets=state_targets,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=args.n_grid,
            )
        else:
            upper = solve_joint_bound(
                markets,
                Qbar=Qbar,
                mode="upper",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=args.n_grid,
            )
            lower = solve_joint_bound(
                markets,
                Qbar=Qbar,
                mode="lower",
                p_min=0.0,
                p_max=p_max_grid,
                n_grid=args.n_grid,
            )

    g_steep = -args.p_base / args.eps_L
    Psi_ref = 0.5 * abs(g_steep) * (meta["shortage"] ** 2)
    ratio_upper = upper["Phi"] / Psi_ref if Psi_ref > 0 else float("nan")
    ratio_lower = lower["Phi"] / Psi_ref if Psi_ref > 0 else float("nan")

    print(
        f"Upper Φ={upper['Phi']*100:.3f}% at p*={upper['p_star']:.3f}; "
        f"ratio Φ/Ψ={ratio_upper:.3f}"
    )
    print(
        f"Lower Φ={lower['Phi']*100:.3f}% at p*={lower['p_star']:.3f}; "
        f"ratio Φ/Ψ={ratio_lower:.3f}"
    )

    bounds_df = pd.DataFrame(
        [
            {
                "adding_up": args.adding_up,
                "outer_p0": args.outer_p0,
                "p0_method": args.p0_method,
                "n_states": meta["n_states"],
                "n_markets_active": meta["n_markets_active"],
                "Phi_upper_pct": upper["Phi"] * 100.0,
                "Phi_lower_pct": lower["Phi"] * 100.0,
                "Phi_upper_over_Psi_pct": ratio_upper * 100.0,
                "Phi_lower_over_Psi_pct": ratio_lower * 100.0,
                "p_star_upper": upper["p_star"],
                "p_star_lower": lower["p_star"],
                "q_open": meta["q_open"],
                "q_non_open": meta["q_non_open"],
            }
        ]
    )

    meta_out = {
        "params": {
            "shortage": float(args.shortage),
            "p_control": float(args.p_control),
            "eps_open": float(args.eps_open),
            "eps_L": float(args.eps_L),
            "eps_U": float(args.eps_U),
            "p_base": float(args.p_base),
            "q_max": float(args.q_max),
            "adding_up": str(args.adding_up),
            "n_grid": int(args.n_grid),
            "p0_method": str(args.p0_method),
            "outer_p0": str(args.outer_p0),
            "outer_max_iters": int(args.outer_max_iters),
            "outer_starts": int(args.outer_starts),
            "outer_coord_grid": int(args.outer_coord_grid),
            "outer_search_grid": int(args.outer_search_grid),
            "outer_seed": int(args.outer_seed),
            "outer_tol": float(args.outer_tol),
            "state_subset": str(args.state_subset),
        },
        "subset": subset_meta,
        "calibration": {
            "n_states": int(meta["n_states"]),
            "n_markets_active": int(meta["n_markets_active"]),
            "n_zero_mass_cells": int(meta["n_zero_mass_cells"]),
            "Qbar": float(meta["Qbar"]),
            "shortage": float(meta["shortage"]),
            "q_open": float(meta["q_open"]),
            "q_non_open": float(meta["q_non_open"]),
            "mean_rationing": float(meta["rbar"]),
        },
        "harberger": {
            "Psi_ref": float(Psi_ref),
            "Psi_ref_pct": float(Psi_ref * 100.0),
            "g_steep": float(g_steep),
        },
    }
    if "outer_search" in upper:
        meta_out["outer_search_upper"] = upper["outer_search"]
    if "outer_search" in lower:
        meta_out["outer_search_lower"] = lower["outer_search"]

    if args.replication:
        written = _write_outputs(
            out_dir=out_dir,
            bounds_df=bounds_df,
            shadow_df=shadow_df,
            meta=meta_out,
        )
        print("\nWrote outputs:")
        for k, v in written.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
