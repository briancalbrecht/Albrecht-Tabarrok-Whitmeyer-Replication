"""
State-level joint robust bounds (Lemma 1) with binding adding-up constraint.

This script treats each U.S. state as a segmented market i and computes the
Lemma-1 robust bounds on misallocation Φ with Σ_i q_i = Q̄ binding.

Data:
  - `data/AAA Fuel Report 1974 w State Names and total stations simplified.xlsx`
    (AAA state survey: % limiting purchases, % out of fuel, and baseline stations)

Key modeling step (AAA gives station *status* shares, not gallons):
  1) Calibrate open-station demand at the controlled price via Hill demand
     (so open clears at q_open > 1 when p < 1).
  2) Back out q_non_open so the station-weighted average quantity equals Q̄ = 1 - shortage.
  3) Map each state's rationing share r_i into a per-station quantity multiplier:
        q_rel_i = (1 - r_i) * q_open + r_i * q_non_open
     and aggregate quantity q0_i = weight_i * q_rel_i.

Anchors:
  Lemma 1 requires an anchor point (q0_i, p0_i) on each demand curve. Since we
  observe quantities but not shadow prices, we first compute admissible bounds
  p0_i in [p0_lo_i, p0_hi_i] from baseline linear extrapolation (Proposition 2 style).
  This script supports either:
    - fixed p0 selection (`--outer-p0 fixed`), or
    - coordinate search over the full hyper-rectangle Π_i [p0_lo_i, p0_hi_i]
      (`--outer-p0 coordsrch`, default).

Usage:
  python state_joint_robust_bounds.py --replication

Recommended high-precision run settings under `--outer-p0 coordsrch`:
  use a stronger outer search, e.g.:
    --outer-starts 12 --outer-max-iters 10 --outer-coord-grid 5
    --outer-search-grid 4001 --n-grid 8001
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

# Sandbox-safe OpenMP defaults: avoid SHM-backed runtime initialization failures.
os.environ["KMP_USE_SHM"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from state_subset import resolve_state_subset

THIS_DIR = Path(__file__).resolve().parent
# Replication package root (paper_figure_replication/).
REPO_ROOT = THIS_DIR.parent


# =============================================================================
# Lemma 1 Solver (inlined from joint_robust_bounds.py)
# =============================================================================

Mode = Literal["upper", "lower"]


@dataclass(frozen=True)
class MarketSpec:
    """
    Primitives for market i in the Lemma 1 robust bounds problem.

    Economic interpretation:
    - Each market has an inverse demand curve P_i(q) passing through anchor (q0, p0).
    - We observe q0 (quantity at controlled price) but not the full demand curve.
    - Slope bounds [g_L, g_U] come from elasticity estimates: steeper = more inelastic.
    - The choke price M bounds P_i(0): no one pays more than M for the first unit.

    The envelopes ell(p) and u(p) are the min/max quantities market i could demand
    at shadow price p, given slope constraints. These define the "reachability set"
    from the anchor point.

    Parameters
    ----------
    name : str
        Market label (e.g., state name).
    q0 : float
        Observed quantity at controlled price (normalized by baseline).
    p0 : float
        Shadow price at q0. In the paper: extrapolated from baseline via slope bounds.
    g_L : float
        Steepest allowable slope (most negative). From elasticity lower bound.
    g_U : float
        Flattest allowable slope (least negative). From elasticity upper bound.
    q_max : float
        Capacity bound on quantity (default: inf = no bound).
    M : float
        Choke price bound P(0) <= M (default: inf = no bound).
    """

    name: str
    q0: float
    p0: float
    g_L: float
    g_U: float
    q_max: float = float("inf")
    M: float = float("inf")

    def __post_init__(self) -> None:
        if not (self.g_L <= self.g_U < 0):
            raise ValueError(
                f"{self.name}: need g_L <= g_U < 0, got g_L={self.g_L}, g_U={self.g_U}"
            )
        if not (0 <= self.q0 <= self.q_max):
            raise ValueError(
                f"{self.name}: need 0 <= q0 <= q_max, got q0={self.q0}, q_max={self.q_max}"
            )
        if self.p0 < 0:
            raise ValueError(f"{self.name}: need p0 >= 0, got p0={self.p0}")

    @property
    def kappa(self) -> float:
        """kappa = 1/g_L - 1/g_U (>0 when g_L < g_U)."""
        return (1.0 / self.g_L) - (1.0 / self.g_U)

    def ell(self, p: float) -> float:
        """Lower quantity envelope ell_i(p)."""
        if np.isfinite(self.M) and p >= self.M:
            return 0.0
        q1 = self.q0 + (p - self.p0) / self.g_U
        q2 = self.q0 + (p - self.p0) / self.g_L
        q = min(q1, q2)
        q = float(np.clip(q, 0.0, self.q_max))
        return q

    def u(self, p: float) -> float:
        """Upper quantity envelope u_i(p), including optional choke cap."""
        if np.isfinite(self.M) and p >= self.M:
            return 0.0

        q1 = self.q0 + (p - self.p0) / self.g_U
        q2 = self.q0 + (p - self.p0) / self.g_L
        q_slope = max(q1, q2, 0.0)

        if np.isfinite(self.M):
            q_choke = max(0.0, (p - self.M) / self.g_U)
        else:
            q_choke = float("inf")

        q = min(self.q_max, q_slope, q_choke)
        return float(max(0.0, q))


def _cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral with y[0] -> 0."""
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty_like(x, dtype=float)
    out[0] = 0.0
    out[1:] = np.cumsum(area)
    return out


@dataclass(frozen=True)
class _PrecomputedEnvelopes:
    p_grid: np.ndarray
    ell_vals: np.ndarray
    u_vals: np.ndarray
    ell_cum: np.ndarray
    u_cum: np.ndarray

    def ell(self, p: float) -> float:
        return float(np.interp(p, self.p_grid, self.ell_vals))

    def u(self, p: float) -> float:
        return float(np.interp(p, self.p_grid, self.u_vals))

    def int_ell(self, a: float, b: float) -> float:
        return self._int(a, b, self.ell_cum)

    def int_u(self, a: float, b: float) -> float:
        return self._int(a, b, self.u_cum)

    def _int(self, a: float, b: float, cum: np.ndarray) -> float:
        if a == b:
            return 0.0
        if a > b:
            a, b = b, a
            sign = -1.0
        else:
            sign = 1.0
        ca = float(np.interp(a, self.p_grid, cum))
        cb = float(np.interp(b, self.p_grid, cum))
        return sign * (cb - ca)


def _precompute_market(m: MarketSpec, p_grid: np.ndarray) -> _PrecomputedEnvelopes:
    ell_vals = np.array([m.ell(float(p)) for p in p_grid], dtype=float)
    u_vals = np.array([m.u(float(p)) for p in p_grid], dtype=float)
    ell_cum = _cumtrapz(p_grid, ell_vals)
    u_cum = _cumtrapz(p_grid, u_vals)
    return _PrecomputedEnvelopes(
        p_grid=p_grid, ell_vals=ell_vals, u_vals=u_vals, ell_cum=ell_cum, u_cum=u_cum
    )


def _inner_x_many(
    Qbar: float,
    l: np.ndarray,
    u: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Solve the inner allocation problem given a candidate shadow price p*.

    Economic interpretation:
        Given p*, each market i has a "target" quantity c_i (from envelope)
        and bounds [l_i, u_i] (from reachability). The adding-up constraint
        Σ x_i = Q̄ may force some markets away from their targets.

        This QP finds the allocation that deviates minimally from targets
        while satisfying adding-up. The kappa_i weights reflect demand
        curvature: markets with flatter demand (larger kappa) absorb more
        of the reallocation.

    Mathematical problem:
        minimize  Σ_i (x_i - c_i)² / (2κ_i)
        s.t.      Σ_i x_i = Q̄
                  l_i ≤ x_i ≤ u_i

    Solution via KKT: x_i(λ) = clip(c_i - λκ_i, [l_i, u_i]) where λ is the
    shadow price of the adding-up constraint, found by bisection.
    """
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    c = np.asarray(c, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    if not (l.shape == u.shape == c.shape == kappa.shape):
        raise ValueError("l, u, c, kappa must have identical shapes.")

    if np.any(u < l - 1e-12):
        raise ValueError("Invalid bounds: some u < l.")

    pinned = kappa <= tol
    x = np.empty_like(c, dtype=float)
    if np.any(pinned):
        x[pinned] = np.clip(c[pinned], l[pinned], u[pinned])
        Qbar = float(Qbar - np.sum(x[pinned]))
        if Qbar < -1e-10:
            raise ValueError("Pinned markets exceed total quantity.")

    flex = ~pinned
    if not np.any(flex):
        if abs(Qbar) > 1e-10:
            raise ValueError("No flexible markets but Qbar not matched.")
        return x

    lf = l[flex]
    uf = u[flex]
    cf = c[flex]
    kf = kappa[flex]

    L = float(np.sum(lf))
    U = float(np.sum(uf))
    if Qbar < L - 1e-10 or Qbar > U + 1e-10:
        raise ValueError("Infeasible X(p): Qbar outside [sum l, sum u].")

    lam_low = float(np.min((cf - uf) / kf))
    lam_high = float(np.max((cf - lf) / kf))

    if abs(Qbar - U) <= 1e-12:
        x[flex] = uf
        return x
    if abs(Qbar - L) <= 1e-12:
        x[flex] = lf
        return x

    for _ in range(max_iter):
        lam = 0.5 * (lam_low + lam_high)
        xf = np.clip(cf - lam * kf, lf, uf)
        s = float(np.sum(xf))
        if abs(s - Qbar) <= 1e-12:
            break
        if s > Qbar:
            lam_low = lam
        else:
            lam_high = lam

    x[flex] = np.clip(cf - lam * kf, lf, uf)
    return x


def solve_joint_bound(
    markets: Sequence[MarketSpec],
    Qbar: Optional[float] = None,
    *,
    mode: Mode,
    p_min: float = 0.0,
    p_max: Optional[float] = None,
    n_grid: int = 4001,
) -> Dict:
    """
    Solve the Lemma 1 joint robust-bounds problem for n markets.

    Economic problem (Lemma 1 in the paper):
        Find the allocation x = (x_1, ..., x_n) that maximizes (mode="upper")
        or minimizes (mode="lower") the misallocation DWL Φ, subject to:
          (1) adding-up: Σ_i x_i = Q̄  (total quantity is fixed)
          (2) reachability: x_i in the envelope set for market i

    The envelope set captures what quantities are consistent with some
    demand curve passing through anchor (q0_i, p0_i) with slope in [g_L, g_U].

    Solution approach:
        Outer loop: search over common shadow price p* on a grid.
        Inner loop: for each p*, solve a strictly convex QP to find the
            worst-case (or best-case) allocation x(p*) satisfying adding-up
            and envelope constraints.

    The key economic insight: the adding-up constraint with 48 markets
    disciplines the bounds tightly. Reallocating to one state means taking
    from another, which limits how bad misallocation can get.

    Uses grid search for outer p-optimization and KKT/bisection for inner x.
    """
    ms = list(markets)
    if len(ms) < 2:
        raise ValueError("Need at least 2 markets for a joint bound.")

    if Qbar is None:
        Qbar = float(sum(m.q0 for m in ms))
    Qbar = float(Qbar)

    if p_max is None:
        finite_chokes = [m.M for m in ms if np.isfinite(m.M)]
        p_max = max([10.0, *(m.p0 for m in ms), *finite_chokes])
    p_max = float(p_max)

    if not (p_max > p_min):
        raise ValueError("Need p_max > p_min.")
    if n_grid < 101:
        raise ValueError("Need a reasonably fine grid (n_grid >= 101).")

    p_grid = np.linspace(p_min, p_max, n_grid)

    names = [m.name for m in ms]
    q0 = np.array([m.q0 for m in ms], dtype=float)
    p0 = np.array([m.p0 for m in ms], dtype=float)
    kappa = np.array([m.kappa for m in ms], dtype=float)

    q0p0_const = float(np.sum(q0 * p0))

    ell_vals = np.empty((len(ms), len(p_grid)), dtype=float)
    u_vals = np.empty((len(ms), len(p_grid)), dtype=float)
    int_ell = np.empty_like(ell_vals)
    int_u = np.empty_like(u_vals)

    for i, m in enumerate(ms):
        env = _precompute_market(m, p_grid)
        ell_vals[i, :] = env.ell_vals
        u_vals[i, :] = env.u_vals
        ell0 = float(np.interp(m.p0, p_grid, env.ell_cum))
        u0 = float(np.interp(m.p0, p_grid, env.u_cum))
        int_ell[i, :] = env.ell_cum - ell0
        int_u[i, :] = env.u_cum - u0

    L_grid = np.sum(ell_vals, axis=0)
    U_grid = np.sum(u_vals, axis=0)
    feasible = (L_grid <= Qbar + 1e-10) & (U_grid >= Qbar - 1e-10)

    if not np.any(feasible):
        raise ValueError("No feasible p found (I is empty). Check anchors/slope bounds/Qbar.")

    obj = np.full_like(p_grid, fill_value=np.nan, dtype=float)
    x_grid = np.full((len(p_grid), len(ms)), fill_value=np.nan, dtype=float)

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

        val = Qbar * float(p) - q0p0_const - float(np.sum(I))

        l = ell_vals[:, j]
        u = u_vals[:, j]

        x = _inner_x_many(Qbar, l, u, c, kappa)

        pen = 0.0
        ok = kappa > 0
        if np.any(ok):
            pen = float(np.sum(((x[ok] - c[ok]) ** 2) / (2.0 * kappa[ok])))

        val += sign * pen

        obj[j] = val
        x_grid[j, :] = x

    if mode == "upper":
        j_star = int(np.nanargmax(obj))
    else:
        j_star = int(np.nanargmin(obj))

    p_star = float(p_grid[j_star])
    Phi_star = float(obj[j_star])
    x_star = [float(v) for v in x_grid[j_star, :]]

    return {
        "mode": mode,
        "Phi": Phi_star,
        "p_star": p_star,
        "market_names": names,
        "x_star": x_star,
        "p_grid": p_grid,
        "objective_grid": obj,
        "x_grid": x_grid,
        "L_grid": L_grid,
        "U_grid": U_grid,
    }


def _markets_with_p0(markets: Sequence[MarketSpec], p0: Sequence[float]) -> List[MarketSpec]:
    """Clone markets with a replacement p0 vector."""
    if len(markets) != len(p0):
        raise ValueError("Length mismatch between markets and p0 vector.")
    out: List[MarketSpec] = []
    for m, p in zip(markets, p0):
        out.append(
            MarketSpec(
                name=m.name,
                q0=m.q0,
                p0=float(p),
                g_L=m.g_L,
                g_U=m.g_U,
                q_max=m.q_max,
                M=m.M,
            )
        )
    return out


def _is_better(candidate: float, incumbent: Optional[float], mode: Mode, tol: float) -> bool:
    if incumbent is None:
        return True
    if mode == "upper":
        return candidate > incumbent + tol
    return candidate < incumbent - tol


def _solve_joint_bound_safe(
    markets: Sequence[MarketSpec],
    *,
    Qbar: float,
    mode: Mode,
    p_min: float,
    p_max: float,
    n_grid: int,
) -> Optional[Dict]:
    try:
        return solve_joint_bound(
            markets,
            Qbar=Qbar,
            mode=mode,
            p_min=p_min,
            p_max=p_max,
            n_grid=n_grid,
        )
    except ValueError:
        return None


def _coordinate_candidates(lo: float, hi: float, current: float, n_points: int) -> np.ndarray:
    if n_points <= 2:
        pts = np.array([lo, hi], dtype=float)
    else:
        pts = np.linspace(lo, hi, n_points)
    pts = np.append(pts, current)
    return np.unique(np.clip(pts, lo, hi))


def _coordinate_search_over_p0(
    markets_base: Sequence[MarketSpec],
    *,
    p0_lo: Sequence[float],
    p0_hi: Sequence[float],
    start_p0: Sequence[float],
    Qbar: float,
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
        out = _solve_joint_bound_safe(
            markets,
            Qbar=Qbar,
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
        fallback_points = [
            0.5 * (lo + hi),
            lo.copy(),
            hi.copy(),
        ]
        for guess in fallback_points:
            guess = np.asarray(guess, dtype=float)
            cand = evaluate(guess, n_grid_search)
            if cand is not None:
                cur = guess
                best = cand
                break
    if best is None:
        raise ValueError("No feasible p0 vector found for coordinate search start.")

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
        final_eval["outer_warning"] = "Final-grid evaluation infeasible; using search-grid solution."

    return {
        "result": final_eval,
        "iterations": n_iter_done,
        "eval_count": n_eval,
        "infeasible_count": n_infeasible,
    }


def optimize_joint_bound_over_p0(
    markets_base: Sequence[MarketSpec],
    *,
    p0_lo: Sequence[float],
    p0_hi: Sequence[float],
    start_p0: Sequence[float],
    Qbar: float,
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
    """
    Outer optimization over p0_i in the hyper-rectangle Π_i [p0_lo_i, p0_hi_i].
    """
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
        # Bounds can collapse so fewer than n_starts unique initializations may exist.
        n_starts = len(starts)

    best_outer: Optional[Dict] = None
    all_attempts = 0
    total_evals = 0
    total_infeasible = 0

    for s in starts:
        all_attempts += 1
        try:
            local = _coordinate_search_over_p0(
                markets_base,
                p0_lo=lo,
                p0_hi=hi,
                start_p0=s,
                Qbar=Qbar,
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
        raise ValueError("Outer p0 optimization failed: no feasible start produced a solution.")

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


# =============================================================================
# Hill demand calibration
# =============================================================================
#
# Hill demand: P(q) = P_max / (1 + (q/κ)^(1/ε))
#
# Inverting: Q(p) = κ × (P_max/p - 1)^ε
#
# Key property: elasticity at baseline (q=1, p=1) equals ε × (P_max - 1)/P_max.
# With P_max = 5 and ε_econ = 0.2, elasticity at baseline ≈ 0.16.
#
# We calibrate so Q(p_base) = Q_base, which pins κ.
# =============================================================================


def calibrate_hill_epsilon(epsilon_econ: float, P_max: float, p_base: float = 1.0) -> float:
    """Convert economic elasticity at baseline to Hill ε parameter."""
    ratio = P_max / p_base
    return epsilon_econ * (ratio - 1.0) / ratio


def calibrate_hill_kappa(Q_base: float, P_max: float, p_base: float, epsilon_hill: float) -> float:
    """Calibrate Hill κ so that Q(p_base) = Q_base."""
    ratio = P_max / p_base
    return Q_base * ((ratio - 1.0) ** (-epsilon_hill))


def hill_quantity(p: float, P_max: float, kappa: float, epsilon_hill: float) -> float:
    """Hill demand: Q(p) = κ × (P_max/p - 1)^ε. Returns 0 at choke price."""
    if p >= P_max:
        return 0.0
    if p <= 0:
        return float("inf")
    return kappa * ((P_max / p - 1.0) ** epsilon_hill)


def linear_open_quantity(*, eps_open: float, p_control: float, p_base: float = 1.0, q_base: float = 1.0) -> float:
    """
    Open-station quantity from baseline-anchored linear demand with elasticity eps_open.

    With slope g = -p_base / (eps_open * q_base), demand at p_control is:
        q_open = q_base + (p_control - p_base) / g
    """
    if eps_open <= 0:
        raise ValueError(f"Need eps_open > 0, got {eps_open}.")
    if p_base <= 0:
        raise ValueError(f"Need p_base > 0, got {p_base}.")
    if q_base <= 0:
        raise ValueError(f"Need q_base > 0, got {q_base}.")
    g_open = -p_base / (eps_open * q_base)
    return float(q_base + (p_control - p_base) / g_open)


# =============================================================================
# State data loading and market construction
# =============================================================================

STATE_ABBREVS = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}


@dataclass(frozen=True)
class StateRow:
    state: str
    stations: float
    rationing: float  # limiting + out-of-fuel share in [0,1]


def load_state_rationing(
    data_root: Path,
    *,
    include_states: Optional[Sequence[str]] = None,
) -> List[StateRow]:
    """
    Load AAA survey shares for states with coverage.

    We drop DC and keep the 48 states in the AAA file (it excludes Alabama/Alaska).
    """
    aaa_file = data_root / "AAA Fuel Report 1974 w State Names and total stations simplified.xlsx"
    df = pd.read_excel(
        aaa_file,
        usecols=["State", "Gasoline Stations 1972", "% Limiting Purchases", "%  Out of Fuel"],
    )

    df["State"] = df["State"].astype(str).str.strip()
    df = df[df["State"] != "District of Columbia"].copy()

    if include_states is not None:
        keep = {str(s).strip() for s in include_states if str(s).strip()}
        df = df[df["State"].isin(keep)].copy()

    df["Gasoline Stations 1972"] = pd.to_numeric(df["Gasoline Stations 1972"], errors="coerce")
    df["% Limiting Purchases"] = pd.to_numeric(df["% Limiting Purchases"], errors="coerce")
    df["%  Out of Fuel"] = pd.to_numeric(df["%  Out of Fuel"], errors="coerce")

    df = df[df["Gasoline Stations 1972"].notna() & (df["Gasoline Stations 1972"] > 0)].copy()

    rationing_pct = df["% Limiting Purchases"].fillna(0.0) + df["%  Out of Fuel"].fillna(0.0)
    rationing = (rationing_pct.clip(lower=0.0, upper=100.0)) / 100.0

    out: List[StateRow] = []
    for state, stations, r in zip(df["State"].astype(str), df["Gasoline Stations 1972"], rationing):
        out.append(StateRow(state=state, stations=float(stations), rationing=float(r)))

    if len(out) < 2:
        raise ValueError("AAA file produced too few states (check input file).")
    return out


def p0_bounds_from_baseline(
    *,
    q_base: float,
    q_obs: float,
    p_base: float,
    g_L: float,
    g_U: float,
) -> Tuple[float, float]:
    """
    Compute admissible shadow price bounds at observed quantity q_obs.

    We know the demand curve passes through baseline (q_base, p_base) with
    slope in [g_L, g_U]. Extrapolating to q_obs gives:
        p = p_base + g × (q_obs - q_base)

    If q_obs < q_base (market is rationed), p > p_base: shortage raises shadow price.
    If q_obs > q_base (market is oversupplied), p < p_base: glut lowers shadow price.
    """
    # Extrapolate with steepest and flattest slopes
    p_steep = p_base + g_L * (q_obs - q_base)
    p_flat = p_base + g_U * (q_obs - q_base)
    lo = float(max(0.0, min(p_steep, p_flat)))
    hi = float(max(p_steep, p_flat))
    return lo, hi


def build_state_markets(
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
) -> Tuple[List[MarketSpec], Dict]:
    """
    Construct MarketSpec objects from AAA survey data.

    Economic calibration (three steps):

    Step 1: Station status → per-station quantity
        The AAA survey gives station *status* shares (open vs limiting/closed),
        not gallons. We convert using a baseline-anchored linear demand to get:
          - q_open: quantity per open station (clears at controlled price)
          - q_non_open: quantity per non-open station (rationed)
        The national shortage pins q_non_open given q_open.

    Step 2: State rationing → state quantity
        Each state's observed quantity is:
          q_obs_i = weight_i × [(1 - r_i) × q_open + r_i × q_non_open]
        where r_i is the rationing share and weight_i is the station share.
        High-rationing states like Connecticut get less per station than
        low-rationing states like Idaho.

    Step 3: Slope bounds → shadow price bounds
        We don't observe shadow prices. But from the baseline (q_base, p_base)
        and slope bounds [g_L, g_U], we extrapolate admissible p0_i:
          p0 = p_base + g × (q_obs - q_base)
        Taking extreme slopes gives [p0_lo, p0_hi]. Rationed states have
        higher shadow prices; oversupplied states have lower.

    Note: No choke price constraint. State-level shadow prices stay well below
    any reasonable choke (max ~2.6 vs choke ~5), so the constraint never binds.
    """
    # Market weights: each state's share of total stations
    stations = np.array([r.stations for r in rows], dtype=float)
    weights = stations / float(np.sum(stations))
    rationing = np.array([r.rationing for r in rows], dtype=float)

    # -------------------------------------------------------------------------
    # Step 1: Station status → per-station quantity
    # Baseline-anchored linear demand at open stations.
    # For default eps_open=0.25 and p_control=0.8, this gives q_open=1.05.
    # The national shortage pins q_non_open: solve for the rationed quantity
    # such that weighted average equals national supply Q = 1 - shortage.
    # -------------------------------------------------------------------------
    q_open = linear_open_quantity(eps_open=eps_open, p_control=p_control, p_base=1.0, q_base=1.0)

    Q_target = 1.0 - shortage
    rbar = float(np.sum(weights * rationing))  # national average rationing share
    if rbar <= 1e-12:
        # Edge case: no rationing anywhere
        q_non_open = Q_target
        q_open = Q_target
    else:
        # Back out q_non_open from: (1-rbar)*q_open + rbar*q_non_open = Q_target
        q_non_open = (Q_target - (1.0 - rbar) * q_open) / rbar
        if q_non_open < 0.0:
            # Shortage too severe: all rationed stations get zero
            q_non_open = 0.0
            q_open = Q_target / (1.0 - rbar)

    # -------------------------------------------------------------------------
    # Step 2: State rationing share → state quantity
    # High-rationing states (Connecticut) have more non-open stations, so they
    # get less per station. Low-rationing states (Idaho) get more.
    # -------------------------------------------------------------------------
    q_rel = (1.0 - rationing) * q_open + rationing * q_non_open  # per-station
    q0 = weights * q_rel  # state quantity = weight × per-station
    Qbar = float(np.sum(q0))  # should equal Q_target by construction

    # -------------------------------------------------------------------------
    # Step 3: Slope bounds → shadow price bounds
    # For each state, extrapolate from baseline (q_base, p_base) to get
    # admissible shadow prices at observed quantity q_obs.
    # -------------------------------------------------------------------------
    markets: List[MarketSpec] = []
    p0_used: List[float] = []
    p0_lo_vec: List[float] = []
    p0_hi_vec: List[float] = []

    for i, r in enumerate(rows):
        q_base = float(weights[i])  # baseline quantity = state's station share
        q_obs = float(q0[i])        # observed quantity under price control

        # Slope bounds from elasticity: g = -p/(ε*q), steeper for lower ε
        g_L = -p_base / (eps_L * q_base)  # steepest (most inelastic)
        g_U = -p_base / (eps_U * q_base)  # flattest (most elastic)

        # Extrapolate shadow price: p0 = p_base + g*(q_obs - q_base)
        # If q_obs < q_base (rationed), shadow price > p_base.
        # If q_obs > q_base (oversupplied), shadow price < p_base.
        lo, hi = p0_bounds_from_baseline(
            q_base=q_base, q_obs=q_obs, p_base=p_base, g_L=g_L, g_U=g_U
        )

        # Select anchor p0 based on method
        if p0_method == "controlled":
            p0 = p_control
        elif p0_method == "baseline_low":
            p0 = lo
        elif p0_method == "baseline_high":
            p0 = hi
        elif p0_method == "baseline_mid":
            p0 = 0.5 * (lo + hi)
        else:
            raise ValueError(f"Unknown p0_method={p0_method!r}.")

        markets.append(
            MarketSpec(
                name=r.state,
                q0=q_obs,
                p0=float(p0),
                g_L=float(g_L),
                g_U=float(g_U),
                q_max=float(q_max),
                # No choke constraint: state-level shadow prices never approach choke
            )
        )
        p0_used.append(float(p0))
        p0_lo_vec.append(float(lo))
        p0_hi_vec.append(float(hi))

    meta = {
        "n_states": len(rows),
        "Qbar": Qbar,
        "shortage": 1.0 - Qbar,
        "q_open": q_open,
        "q_non_open": q_non_open,
        "rbar": float(rbar),
        "p0_method": p0_method,
        "p0_min": float(np.min(p0_used)),
        "p0_max": float(np.max(p0_used)),
        "p0_bounds_min": float(np.min(p0_lo_vec)),
        "p0_bounds_max": float(np.max(p0_hi_vec)),
        "p0_lo_vec": [float(v) for v in p0_lo_vec],
        "p0_hi_vec": [float(v) for v in p0_hi_vec],
        "p0_init_vec": [float(v) for v in p0_used],
    }
    return markets, meta


def build_state_shadow_price_bounds_table(
    rows: List[StateRow],
    *,
    shortage: float,
    p_control: float,
    eps_open: float,
    p_base: float,
    eps_L: float,
    eps_U: float,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a per-state table of shadow-price bounds at the observed allocation.
    """
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

    q_rel = (1.0 - rationing) * q_open + rationing * q_non_open

    records = []
    for i, r in enumerate(rows):
        q_base = float(weights[i])
        q_obs = float(weights[i] * q_rel[i])

        g_L = -p_base / (eps_L * q_base)
        g_U = -p_base / (eps_U * q_base)

        p0_lo, p0_hi = p0_bounds_from_baseline(
            q_base=q_base, q_obs=q_obs, p_base=p_base, g_L=g_L, g_U=g_U
        )
        records.append(
            {
                "state": r.state,
                "abbrev": STATE_ABBREVS.get(r.state, ""),
                "stations": float(r.stations),
                "weight": float(weights[i]),
                "rationing": float(r.rationing),
                "q_rel": float(q_rel[i]),
                "q_obs": float(q_obs),
                "p0_lo": float(p0_lo),
                "p0_mid": float(0.5 * (p0_lo + p0_hi)),
                "p0_hi": float(p0_hi),
            }
        )

    meta = {
        "n_states": len(rows),
        "q_open": q_open,
        "q_non_open": q_non_open,
        "mean_rationing": float(rbar),
    }
    return pd.DataFrame.from_records(records), meta


# =============================================================================
# Plotting functions
# =============================================================================


def _plot_state_shadow_price_bounds(
    df: pd.DataFrame,
    *,
    p_control: float,
    p_base: float,
    p_star_upper: Optional[float],
    p_star_lower: Optional[float],
    out_dir: Path,
    filename_stem: str = "figure_state_shadow_price_bounds",
) -> Tuple[Path, Path]:
    """
    Caterpillar plot: for each state, show [p0_lo, p0_hi] with p0_mid marker.
    """
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort high-to-low so the y-axis direction matches the colorbar direction.
    d = df.sort_values(["p0_mid", "state"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(d), dtype=float)

    fig_h = max(10.0, 0.25 * len(d) + 1.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    ax.hlines(y, d["p0_lo"], d["p0_hi"], color="lightgray", linewidth=2.5, zorder=1)

    # Match the map's blue->red shadow-price color gradation.
    map_palette = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]
    map_cmap = LinearSegmentedColormap.from_list("shadow_map_scale", map_palette, N=256)

    sc = ax.scatter(
        d["p0_mid"],
        y,
        c=d["rationing"],
        cmap=map_cmap,
        vmin=0.0,
        vmax=1.0,
        s=26,
        edgecolors="black",
        linewidths=0.25,
        zorder=2,
        label="midpoint",
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Rationing share (limiting + out-of-fuel)")

    ax.axvline(x=p_control, color="gray", linestyle="--", linewidth=1.5, label=r"$\bar p$")
    ax.axvline(x=p_base, color="black", linestyle=":", linewidth=1.5, label=r"$p^{base}$")

    if p_star_upper is not None:
        ax.axvline(
            x=float(p_star_upper),
            color="tab:red",
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            label=r"$p^*$ (upper)",
        )
    if p_star_lower is not None and (p_star_upper is None or abs(p_star_lower - p_star_upper) > 1e-10):
        ax.axvline(
            x=float(p_star_lower),
            color="tab:blue",
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            label=r"$p^*$ (lower)",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(d["state"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Shadow-price bounds at observed allocation (lower to upper bound per state)")
    ax.set_title("State Shadow-Price Bounds (from baseline + elasticity bounds)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.tight_layout()
    pdf_path = out_dir / f"{filename_stem}.pdf"
    png_path = out_dir / f"{filename_stem}.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def _plot_shadow_price_map(
    df: pd.DataFrame,
    *,
    p_base: float,
    out_dir: Path,
    filename_stem: str = "figure_state_shadow_price_map",
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Choropleth map: two stacked panels showing min and max shadow-price bounds.
    Style matches Figure 1 (RationingTotal1974.pdf).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("Warning: plotly not available, skipping choropleth map.")
        return None, None

    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["p0_lo_rel"] = df["p0_lo"] / p_base
    df["p0_hi_rel"] = df["p0_hi"] / p_base
    all_abbrevs = {abbr for abbr in STATE_ABBREVS.values() if isinstance(abbr, str) and len(abbr) == 2}
    present_abbrevs = {str(a) for a in df["abbrev"].dropna().tolist()}
    missing_abbrevs = sorted(all_abbrevs - present_abbrevs)

    # Diverging scale anchored at baseline=1.0 (white center).
    tick_vals = [0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
    tick_txt = ["0.7", "0.9", "1.0", "1.2", "1.5", "2.0", "2.5"]
    colors = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"]
    colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]

    # Two-panel stacked layout: upper bound on top, lower bound on bottom.
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=(
            "A. Upper Bound: Highest inferred shadow price (relative to baseline)",
            "B. Lower Bound: Lowest inferred shadow price (relative to baseline)",
        ),
        vertical_spacing=0.08,
    )

    if missing_abbrevs:
        # Render no-data states in black while keeping background neutral.
        missing_trace = go.Choropleth(
            locations=missing_abbrevs,
            z=[1.0] * len(missing_abbrevs),
            locationmode="USA-states",
            colorscale=[[0.0, "#000000"], [1.0, "#000000"]],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            hovertemplate="%{location}<br>No data<extra></extra>",
        )
        fig.add_trace(missing_trace, row=1, col=1)
        fig.add_trace(missing_trace, row=2, col=1)

    # Panel A (top): Upper bound (p0_hi)
    fig.add_trace(
        go.Choropleth(
            locations=df["abbrev"],
            z=df["p0_hi_rel"],
            locationmode="USA-states",
            colorscale=colorscale,
            zmin=0.6,
            zmax=3.0,
            zmid=1.0,
            colorbar=dict(
                title="Upper bound shadow price<br>/ baseline price",
                tickvals=tick_vals,
                ticktext=tick_txt,
                len=0.35,
                y=0.82,
                x=1.02,
            ),
            hovertemplate=(
                "%{location}<br>"
                "Upper-bound shadow price: %{z:.2f} x baseline<br>"
                "Rationing: %{customdata[0]:.0%}<br>"
                "<extra></extra>"
            ),
            customdata=df[["rationing"]].values,
        ),
        row=1, col=1,
    )

    # Panel B (bottom): Lower bound (p0_lo)
    fig.add_trace(
        go.Choropleth(
            locations=df["abbrev"],
            z=df["p0_lo_rel"],
            locationmode="USA-states",
            colorscale=colorscale,
            zmin=0.6,
            zmax=3.0,
            zmid=1.0,
            colorbar=dict(
                title="Lower bound shadow price<br>/ baseline price",
                tickvals=tick_vals,
                ticktext=tick_txt,
                len=0.35,
                y=0.18,
                x=1.02,
            ),
            hovertemplate=(
                "%{location}<br>"
                "Lower-bound shadow price: %{z:.2f} x baseline<br>"
                "Rationing: %{customdata[0]:.0%}<br>"
                "<extra></extra>"
            ),
            customdata=df[["rationing"]].values,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title_text="Inferred State Shadow-Price Bounds Relative to Baseline (February 1974)",
        title_x=0.5,
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
        margin=dict(l=40, r=160, t=70, b=20),
        font=dict(size=11),
    )

    pdf_path = out_dir / f"{filename_stem}.pdf"
    png_path = out_dir / f"{filename_stem}.png"

    try:
        pio.write_image(fig, str(png_path), width=900, height=900, scale=2)
        pio.write_image(fig, str(pdf_path), width=900, height=900, scale=2)
        return pdf_path, png_path
    except Exception as e:
        print(f"Warning: Could not save choropleth map: {e}")
        return None, None


def _build_state_extremal_snapshot(
    result: Dict,
    markets: Sequence[MarketSpec],
    *,
    mode: Mode,
) -> pd.DataFrame:
    """
    Build per-state diagnostics at the bound-attaining p*:
      - feasible interval [ell_i(p*), u_i(p*)]
      - unconstrained endpoint choice c_i(p*)
      - optimizer allocation x_i*
      - adjustment Delta_i = x_i* - c_i(p*) from adding-up
    """
    p_star = float(result["p_star"])
    x_star = np.asarray(result["x_star"], dtype=float)

    if len(markets) != len(x_star):
        raise ValueError("Length mismatch between markets and x_star.")

    rows: List[Dict[str, float | str | bool]] = []
    for m, x_i in zip(markets, x_star):
        ell_i = float(m.ell(p_star))
        u_i = float(m.u(p_star))
        forward = bool(p_star >= m.p0)

        if mode == "upper":
            c_i = ell_i if forward else u_i
        else:
            c_i = u_i if forward else ell_i

        rows.append(
            {
                "state": str(m.name),
                "q0": float(m.q0),
                "p0": float(m.p0),
                "p_star": p_star,
                "ell": ell_i,
                "u": u_i,
                "c": float(c_i),
                "x": float(x_i),
                "delta": float(x_i - c_i),
                "forward": forward,
            }
        )

    return pd.DataFrame.from_records(rows)


def _plot_state_adding_up_extremals(
    *,
    upper_result: Dict,
    lower_result: Dict,
    markets_upper: Sequence[MarketSpec],
    markets_lower: Sequence[MarketSpec],
    out_dir: Path,
    filename_stem: str = "figure_state_adding_up_extremals",
) -> Tuple[Path, Path]:
    """
    Visualize adding-up-induced extremals at p* for upper/lower bound cases.
    """
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)

    d_up = _build_state_extremal_snapshot(upper_result, markets_upper, mode="upper")
    d_lo = _build_state_extremal_snapshot(lower_result, markets_lower, mode="lower")

    n_states = max(len(d_up), len(d_lo))
    fig_h = max(10.0, 0.24 * n_states + 1.8)
    fig, (ax_u, ax_l) = plt.subplots(1, 2, figsize=(16.0, fig_h), sharey=False)

    all_vals = np.concatenate(
        [
            d_up[["ell", "u", "c", "x"]].to_numpy().ravel(),
            d_lo[["ell", "u", "c", "x"]].to_numpy().ravel(),
        ]
    )
    x_min = float(np.nanmin(all_vals))
    x_max = float(np.nanmax(all_vals))
    x_pad = 0.06 * (x_max - x_min + 1e-12)
    x_lo = max(0.0, x_min - x_pad)
    x_hi = x_max + x_pad

    def _draw_panel(ax, d: pd.DataFrame, title: str) -> None:
        d = d.sort_values(["delta", "state"]).reset_index(drop=True)
        y = np.arange(len(d), dtype=float)

        ax.hlines(y, d["ell"], d["u"], color="lightgray", linewidth=2.2, alpha=0.95, zorder=1)

        d_pos = d["delta"] > 1e-10
        d_neg = d["delta"] < -1e-10
        d_zero = ~(d_pos | d_neg)

        if np.any(d_pos):
            ax.hlines(
                y[d_pos],
                d.loc[d_pos, "c"],
                d.loc[d_pos, "x"],
                color="tab:blue",
                linewidth=1.8,
                alpha=0.9,
                zorder=2,
            )
        if np.any(d_neg):
            ax.hlines(
                y[d_neg],
                d.loc[d_neg, "x"],
                d.loc[d_neg, "c"],
                color="tab:red",
                linewidth=1.8,
                alpha=0.9,
                zorder=2,
            )
        if np.any(d_zero):
            ax.scatter(
                d.loc[d_zero, "x"],
                y[d_zero],
                marker="o",
                s=14,
                color="gray",
                alpha=0.7,
                zorder=2,
            )

        ax.scatter(
            d["c"],
            y,
            marker="x",
            s=34,
            color="black",
            linewidths=1.0,
            zorder=3,
        )
        ax.scatter(
            d["x"],
            y,
            marker="o",
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
            zorder=4,
        )

        labels = [STATE_ABBREVS.get(str(s), str(s)[:3].upper()) for s in d["state"]]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("State quantity $q_i$")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.22)

        n_move = int(np.sum(np.abs(d["delta"].to_numpy()) > 1e-9))
        ax.text(
            0.01,
            0.02,
            f"States moved by adding-up: {n_move}/{len(d)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
        )

    _draw_panel(
        ax_u,
        d_up,
        "A. Higher-Loss Case\n"
        f"Estimated loss = {upper_result['Phi']*100:.2f}% of baseline spending, "
        f"common shadow price = {upper_result['p_star']:.2f}",
    )
    _draw_panel(
        ax_l,
        d_lo,
        "B. Lower-Loss Case\n"
        f"Estimated loss = {lower_result['Phi']*100:.2f}% of baseline spending, "
        f"common shadow price = {lower_result['p_star']:.2f}",
    )
    ax_u.set_ylabel("State (sorted by adjustment $\\Delta_i$)")

    handles = [
        Line2D([0], [0], color="lightgray", lw=2.2, label="Feasible interval $[\\ell_i(p^*), u_i(p^*)]$"),
        Line2D([0], [0], color="black", marker="x", lw=0, markersize=6, label="Unconstrained corner $c_i(p^*)$"),
        Line2D([0], [0], color="black", marker="o", markerfacecolor="white", lw=0, markersize=6, label="Optimizer $x_i^*$"),
        Line2D([0], [0], color="tab:blue", lw=1.8, label="$\\Delta_i > 0$ (adding-up raises $x_i$)"),
        Line2D([0], [0], color="tab:red", lw=1.8, label="$\\Delta_i < 0$ (adding-up lowers $x_i$)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "State-Level Reallocation Needed to Satisfy Adding-Up at Bound-Attaining Shadow Prices",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    pdf_path = out_dir / f"{filename_stem}.pdf"
    png_path = out_dir / f"{filename_stem}.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _write_tables(
    *,
    out_dir: Path,
    bounds_df: Optional[pd.DataFrame],
    shadow_df: Optional[pd.DataFrame],
    meta: Dict,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    if bounds_df is not None:
        bounds_path = out_dir / "table_state_joint_bounds.csv"
        bounds_df.to_csv(bounds_path, index=False)
        written["bounds_csv"] = str(bounds_path)

    if shadow_df is not None:
        shadow_path = out_dir / "table_state_shadow_price_bounds.csv"
        shadow_df.to_csv(shadow_path, index=False)
        written["shadow_csv"] = str(shadow_path)

    meta_path = out_dir / "state_joint_bounds_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    written["meta_json"] = str(meta_path)

    return written


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="State-level joint robust bounds (Lemma 1) with binding adding-up constraint."
    )
    parser.add_argument(
        "--replication",
        action="store_true",
        help="Run the full replication bundle: write tables + figures (and p0 sweep if --outer-p0=fixed).",
    )
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
    # Note: No --M (choke price) argument. State-level shadow prices stay well
    # below any reasonable choke, so the constraint never binds.
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument(
        "--p0-method",
        type=str,
        choices=["controlled", "baseline_low", "baseline_high", "baseline_mid"],
        default="baseline_mid",
    )
    parser.add_argument(
        "--p0-sweep",
        action="store_true",
        help="Run baseline_low/mid/high and report a sensitivity range (only with --outer-p0=fixed).",
    )
    parser.add_argument(
        "--outer-p0",
        type=str,
        choices=["fixed", "coordsrch"],
        default="coordsrch",
        help="How to handle unknown p0_i: fixed choice vs coordinate search over [p0_lo_i, p0_hi_i].",
    )
    parser.add_argument(
        "--outer-max-iters",
        type=int,
        default=2,
        help="Coordinate-search passes over states for the outer p0 optimizer.",
    )
    parser.add_argument(
        "--outer-starts",
        type=int,
        default=4,
        help="Number of multi-start initializations for outer p0 optimization.",
    )
    parser.add_argument(
        "--outer-coord-grid",
        type=int,
        default=2,
        help="Candidate points per coordinate in [p0_lo_i, p0_hi_i] (2=endpoints only).",
    )
    parser.add_argument(
        "--outer-search-grid",
        type=int,
        default=2001,
        help="Inner p-grid used during outer p0 search; final value is recomputed at --n-grid.",
    )
    parser.add_argument(
        "--outer-seed",
        type=int,
        default=1974,
        help="Random seed for outer p0 multi-start initialization.",
    )
    parser.add_argument(
        "--outer-tol",
        type=float,
        default=1e-6,
        help="Improvement tolerance for outer p0 coordinate updates.",
    )
    parser.add_argument("--n-grid", type=int, default=8001)
    parser.add_argument(
        "--state-subset",
        type=str,
        choices=["all", "sales36"],
        default="all",
        help=(
            "State sample to use: `all` keeps the standard AAA sample; "
            "`sales36` keeps only states with positive observed sales volume."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for tables/figures (default: output/).",
    )
    parser.add_argument(
        "--write-tables",
        action="store_true",
        help="Write CSV tables and JSON meta file to the output directory.",
    )
    parser.add_argument(
        "--plot-shadow-prices",
        action="store_true",
        help="Save caterpillar plot of per-state shadow-price bounds.",
    )
    parser.add_argument(
        "--plot-map",
        action="store_true",
        help="Save choropleth map of shadow-price bounds.",
    )
    parser.add_argument(
        "--plot-adding-up",
        action="store_true",
        help="Save state-level adding-up extremal figure at the bound-attaining p* values.",
    )
    args = parser.parse_args()

    if args.replication:
        args.write_tables = True
        args.plot_shadow_prices = True
        args.plot_map = True
        args.plot_adding_up = True
        if args.outer_p0 == "fixed":
            args.p0_sweep = True

    if args.outer_p0 != "fixed" and args.p0_sweep:
        print("Note: ignoring --p0-sweep because --outer-p0=coordsrch.")
        args.p0_sweep = False

    if args.outer_p0 == "coordsrch":
        coarse_outer = (
            args.outer_starts < 12
            or args.outer_max_iters < 10
            or args.outer_coord_grid < 5
            or args.outer_search_grid < 4001
            or args.n_grid < 8001
        )
        if coarse_outer:
            print(
                "Note (final-final runs): consider stronger outer search "
                "(--outer-starts 12 --outer-max-iters 10 --outer-coord-grid 5 "
                "--outer-search-grid 4001 --n-grid 8001)."
            )

    include_states, subset_meta = resolve_state_subset(args.state_subset, repo_root=REPO_ROOT)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        base_out = THIS_DIR.parent / "output"
        out_dir = base_out if args.state_subset == "all" else (base_out / f"scenario_{args.state_subset}")

    rows = load_state_rationing(THIS_DIR.parent / "data", include_states=include_states)
    print(
        f"State subset: {subset_meta['subset_id']} "
        f"({subset_meta['subset_label']}); states used in AAA run = {len(rows)}"
    )

    sweep_rows = []
    p0_ranges: Dict[str, Dict[str, float]] = {}
    bounds_df: Optional[pd.DataFrame] = None

    upper: Dict
    lower: Dict
    meta: Dict
    diag_upper: Optional[Dict] = None
    diag_lower: Optional[Dict] = None
    diag_markets_upper: Optional[List[MarketSpec]] = None
    diag_markets_lower: Optional[List[MarketSpec]] = None

    if args.outer_p0 == "fixed":
        methods = (
            ["baseline_low", "baseline_mid", "baseline_high"] if args.p0_sweep else [args.p0_method]
        )
        method_solutions: Dict[str, Dict[str, object]] = {}
        upper_max = -float("inf")
        lower_min = float("inf")

        for method in methods:
            markets, meta = build_state_markets(
                rows,
                shortage=args.shortage,
                p_control=args.p_control,
                eps_open=args.eps_open,
                p_base=args.p_base,
                eps_L=args.eps_L,
                eps_U=args.eps_U,
                q_max=args.q_max,
                p0_method=method,
            )

            Qbar = float(meta["Qbar"])
            shortage = float(meta["shortage"])

            if method == methods[0]:
                print(f"States used in joint program: {meta['n_states']}")
                print(f"Aggregate Qbar: {Qbar:.6f}  (shortage={shortage:.3%})")
                print(
                    f"Status->quantity calibration: q_open={meta['q_open']:.3f}, q_non_open={meta['q_non_open']:.3f}, "
                    f"mean rationing={meta['rbar']:.3f}"
                )

            if method != "controlled":
                print(f"Anchor prices p0_i via {method}: range [{meta['p0_min']:.3f}, {meta['p0_max']:.3f}]")
            else:
                print(f"Anchor prices p0_i: all set to p_control={args.p_control:.3f}")
            p0_ranges[method] = {"p0_min": float(meta["p0_min"]), "p0_max": float(meta["p0_max"])}

            p_max_grid = max(10.0, args.p_control, args.p_base)
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
            method_solutions[method] = {
                "upper": upper,
                "lower": lower,
                "markets_upper": markets,
                "markets_lower": markets,
            }

            g_steep = -args.p_base / args.eps_L
            Psi_ref = 0.5 * abs(g_steep) * shortage**2

            ratio_upper = upper["Phi"] / Psi_ref if Psi_ref > 0 else float("nan")
            ratio_lower = lower["Phi"] / Psi_ref if Psi_ref > 0 else float("nan")

            sweep_rows.append(
                {
                    "outer_p0": "fixed",
                    "p0_method": method,
                    "Phi_upper_pct": upper["Phi"] * 100.0,
                    "Phi_lower_pct": lower["Phi"] * 100.0,
                    "Phi_upper_over_Psi_pct": ratio_upper * 100.0,
                    "Phi_lower_over_Psi_pct": ratio_lower * 100.0,
                    "p_star_upper": upper["p_star"],
                    "p_star_lower": lower["p_star"],
                }
            )
            upper_max = max(upper_max, float(upper["Phi"]))
            lower_min = min(lower_min, float(lower["Phi"]))

            if not args.p0_sweep:
                names = upper["market_names"]
                x_star = np.array(upper["x_star"], dtype=float)
                q0 = np.array([m.q0 for m in markets], dtype=float)
                delta = x_star - q0
                idx = np.argsort(np.abs(delta))[::-1][:10]
                print("\nTop 10 |x_i* - q_i^obs| (upper bound solution):")
                print(f"{'State':<15} {'q_obs':>10} {'x*':>10} {'Δ':>10}")
                for k in idx:
                    print(f"{names[k]:<15} {q0[k]:>10.6f} {x_star[k]:>10.6f} {delta[k]:>10.6f}")

        bounds_df = pd.DataFrame(sweep_rows)
        if args.p0_sweep:
            print("\nLemma 1 joint robust bounds (adding-up binds):")
            print(f"  Harberger Ψ (ε={args.eps_L:.2f}): {Psi_ref*100:.3f}%")
            print(bounds_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(
                f"\nEnvelope over p0 choices: Φ_lower min = {lower_min*100:.3f}%, Φ_upper max = {upper_max*100:.3f}%"
            )

        diag_method = "baseline_mid" if (args.p0_sweep and "baseline_mid" in method_solutions) else methods[0]
        diag_upper = method_solutions[diag_method]["upper"]  # type: ignore[index]
        diag_lower = method_solutions[diag_method]["lower"]  # type: ignore[index]
        diag_markets_upper = method_solutions[diag_method]["markets_upper"]  # type: ignore[index]
        diag_markets_lower = method_solutions[diag_method]["markets_lower"]  # type: ignore[index]
    else:
        markets, meta = build_state_markets(
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
        shortage = float(meta["shortage"])
        p_max_grid = max(10.0, args.p_control, args.p_base)

        print(f"States used in joint program: {meta['n_states']}")
        print(f"Aggregate Qbar: {Qbar:.6f}  (shortage={shortage:.3%})")
        print(
            f"Status->quantity calibration: q_open={meta['q_open']:.3f}, q_non_open={meta['q_non_open']:.3f}, "
            f"mean rationing={meta['rbar']:.3f}"
        )
        print(
            f"Admissible p0_i bounds from baseline: [{meta['p0_bounds_min']:.3f}, {meta['p0_bounds_max']:.3f}]"
        )

        p0_lo = np.array(meta["p0_lo_vec"], dtype=float)
        p0_hi = np.array(meta["p0_hi_vec"], dtype=float)
        p0_init = np.array(meta["p0_init_vec"], dtype=float)
        p0_ranges["admissible"] = {
            "p0_min": float(np.min(p0_lo)),
            "p0_max": float(np.max(p0_hi)),
        }

        search_n_grid = max(101, min(args.n_grid, args.outer_search_grid))

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

        g_steep = -args.p_base / args.eps_L
        Psi_ref = 0.5 * abs(g_steep) * shortage**2

        ratio_upper = upper["Phi"] / Psi_ref if Psi_ref > 0 else float("nan")
        ratio_lower = lower["Phi"] / Psi_ref if Psi_ref > 0 else float("nan")

        p0_upper = np.array(upper["p0_vector"], dtype=float)
        p0_lower = np.array(lower["p0_vector"], dtype=float)
        p0_ranges["upper_opt"] = {"p0_min": float(np.min(p0_upper)), "p0_max": float(np.max(p0_upper))}
        p0_ranges["lower_opt"] = {"p0_min": float(np.min(p0_lower)), "p0_max": float(np.max(p0_lower))}

        print(
            "Outer p0 search (upper): "
            f"Φ={upper['Phi']*100:.3f}%, p*={upper['p_star']:.3f}, "
            f"p0 range [{np.min(p0_upper):.3f}, {np.max(p0_upper):.3f}], "
            f"evals={upper['outer_search']['total_evaluations']}, infeasible={upper['outer_search']['total_infeasible']}"
        )
        print(
            "Outer p0 search (lower): "
            f"Φ={lower['Phi']*100:.3f}%, p*={lower['p_star']:.3f}, "
            f"p0 range [{np.min(p0_lower):.3f}, {np.max(p0_lower):.3f}], "
            f"evals={lower['outer_search']['total_evaluations']}, infeasible={lower['outer_search']['total_infeasible']}"
        )

        sweep_rows.append(
            {
                "outer_p0": "coordsrch",
                "p0_method": args.p0_method,
                "Phi_upper_pct": upper["Phi"] * 100.0,
                "Phi_lower_pct": lower["Phi"] * 100.0,
                "Phi_upper_over_Psi_pct": ratio_upper * 100.0,
                "Phi_lower_over_Psi_pct": ratio_lower * 100.0,
                "p_star_upper": upper["p_star"],
                "p_star_lower": lower["p_star"],
                "p0_upper_min": float(np.min(p0_upper)),
                "p0_upper_max": float(np.max(p0_upper)),
                "p0_lower_min": float(np.min(p0_lower)),
                "p0_lower_max": float(np.max(p0_lower)),
                "outer_starts": int(args.outer_starts),
                "outer_max_iters": int(args.outer_max_iters),
                "outer_coord_grid": int(args.outer_coord_grid),
                "outer_search_grid": int(search_n_grid),
            }
        )
        bounds_df = pd.DataFrame(sweep_rows)

        names = upper["market_names"]
        x_star = np.array(upper["x_star"], dtype=float)
        q0 = np.array([m.q0 for m in markets], dtype=float)
        delta = x_star - q0
        idx = np.argsort(np.abs(delta))[::-1][:10]
        print("\nTop 10 |x_i* - q_i^obs| (upper bound solution):")
        print(f"{'State':<15} {'q_obs':>10} {'x*':>10} {'Δ':>10}")
        for k in idx:
            print(f"{names[k]:<15} {q0[k]:>10.6f} {x_star[k]:>10.6f} {delta[k]:>10.6f}")

        diag_upper = upper
        diag_lower = lower
        if "p0_vector" in upper:
            diag_markets_upper = _markets_with_p0(markets, upper["p0_vector"])
        else:
            diag_markets_upper = markets
        if "p0_vector" in lower:
            diag_markets_lower = _markets_with_p0(markets, lower["p0_vector"])
        else:
            diag_markets_lower = markets

    shadow_df: Optional[pd.DataFrame] = None
    need_shadow = args.plot_shadow_prices or args.plot_map or args.write_tables
    if need_shadow:
        shadow_df, shadow_meta = build_state_shadow_price_bounds_table(
            rows,
            shortage=args.shortage,
            p_control=args.p_control,
            eps_open=args.eps_open,
            p_base=args.p_base,
            eps_L=args.eps_L,
            eps_U=args.eps_U,
        )

    if args.plot_shadow_prices and shadow_df is not None:
        p_star_upper = None
        p_star_lower = None
        if args.p0_sweep and bounds_df is not None and "p0_method" in bounds_df.columns:
            ref = bounds_df[bounds_df["p0_method"] == "baseline_mid"]
            if not ref.empty:
                p_star_upper = float(ref["p_star_upper"].iloc[0])
                p_star_lower = float(ref["p_star_lower"].iloc[0])
        elif not args.p0_sweep:
            p_star_upper = float(upper["p_star"])
            p_star_lower = float(lower["p_star"])

        pdf_path, png_path = _plot_state_shadow_price_bounds(
            shadow_df,
            p_control=args.p_control,
            p_base=args.p_base,
            p_star_upper=p_star_upper,
            p_star_lower=p_star_lower,
            out_dir=out_dir,
        )
        print(f"\nSaved shadow-price bounds figure:\n  - {pdf_path}\n  - {png_path}")

    if args.plot_map and shadow_df is not None:
        pdf_path, png_path = _plot_shadow_price_map(
            shadow_df,
            p_base=args.p_base,
            out_dir=out_dir,
        )
        if pdf_path is not None:
            print(f"\nSaved shadow-price map:\n  - {pdf_path}\n  - {png_path}")

    if (
        args.plot_adding_up
        and diag_upper is not None
        and diag_lower is not None
        and diag_markets_upper is not None
        and diag_markets_lower is not None
    ):
        pdf_path, png_path = _plot_state_adding_up_extremals(
            upper_result=diag_upper,
            lower_result=diag_lower,
            markets_upper=diag_markets_upper,
            markets_lower=diag_markets_lower,
            out_dir=out_dir,
        )
        print(f"\nSaved adding-up extremal figure:\n  - {pdf_path}\n  - {png_path}")

    if args.write_tables:
        meta_out = {
            "params": {
                "shortage": float(args.shortage),
                "p_control": float(args.p_control),
                "eps_open": float(args.eps_open),
                "eps_L": float(args.eps_L),
                "eps_U": float(args.eps_U),
                "p_base": float(args.p_base),
                "q_max": float(args.q_max),
                "n_grid": int(args.n_grid),
                "p0_method": str(args.p0_method),
                "p0_sweep": bool(args.p0_sweep),
                "outer_p0": str(args.outer_p0),
                "outer_max_iters": int(args.outer_max_iters),
                "outer_starts": int(args.outer_starts),
                "outer_coord_grid": int(args.outer_coord_grid),
                "outer_search_grid": int(args.outer_search_grid),
                "outer_seed": int(args.outer_seed),
                "outer_tol": float(args.outer_tol),
            },
            "data_coverage": {
                "n_states": int(len(rows)),
                "subset_id": str(subset_meta["subset_id"]),
                "subset_label": str(subset_meta["subset_label"]),
                "subset_states_raw_count": (
                    int(subset_meta["n_states_raw"])
                    if subset_meta.get("n_states_raw") is not None
                    else None
                ),
                "subset_states_used": sorted([r.state for r in rows]),
                "note": (
                    "Uses AAA simplified file with DC dropped; optionally filtered "
                    "to the requested state subset."
                ),
            },
            "calibration": {
                "Qbar": float(meta["Qbar"]),
                "shortage": float(meta["shortage"]),
                "q_open": float(meta["q_open"]),
                "q_non_open": float(meta["q_non_open"]),
                "mean_rationing": float(meta["rbar"]),
            },
            "p0_ranges": p0_ranges,
            "harberger": {
                "Psi_ref": float(Psi_ref),
                "Psi_ref_pct": float(Psi_ref * 100.0),
                "g_steep": float(-args.p_base / args.eps_L),
            },
        }

        written = _write_tables(out_dir=out_dir, bounds_df=bounds_df, shadow_df=shadow_df, meta=meta_out)
        print("\nWrote replication tables/meta:")
        for k, v in written.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
