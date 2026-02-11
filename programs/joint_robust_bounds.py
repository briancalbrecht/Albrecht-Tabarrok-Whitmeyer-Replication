"""
Joint robust bounds with a binding adding-up constraint.

This implements the "reduction to (p, x)" program described in the paper's
Lemma 1, including adding-up penalty terms.

Core idea:
  - Each market i has an unknown inverse demand P_i(q) with bounded slopes
    g_{i,L} <= P_i'(q) <= g_{i,U} < 0 and (optionally) a choke cap P_i(0) <= M_i.
  - We observe an anchor point (q0_i, p0_i) on each demand curve.
  - For a candidate common shadow price p, market i's feasible quantity is
    interval-bounded by envelopes ell_i(p) <= q_i(p) <= u_i(p).
  - The adding-up constraint binds through x in X(p): sum_i x_i = Qbar with
    x_i in [ell_i(p), u_i(p)].
  - When slopes are truly bounded (g_{i,L} < g_{i,U}), the inverse-quantity
    path cannot instantaneously jump to hit the aggregate constraint at p.
    The resulting "cannot have it all" correction is the quadratic penalty
    term with kappa_i = 1/g_{i,L} - 1/g_{i,U}.

This module focuses on the 2-market case (open vs closed/limiting), where the
inner x-optimization is a 1D quadratic problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np


Mode = Literal["upper", "lower"]


@dataclass(frozen=True)
class MarketSpec:
    """
    Market i primitives expressed in the quantity units used by the adding-up
    constraint (i.e., the q_i that sum to Qbar).

    Parameters
    ----------
    name:
        Label for diagnostics.
    q0, p0:
        Anchor point on inverse demand: P_i(q0) = p0.
        In the lemma, this is the observed allocation point.
    g_L, g_U:
        Slope bounds on inverse demand: g_L <= P'(q) <= g_U < 0.
        g_L is "steeper" (more negative).
    q_max:
        Exogenous capacity bound (optional). Use a large number if irrelevant.
    M:
        Choke cap P(0) <= M (optional). Use np.inf to disable.
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
        """Upper quantity envelope u_i(p), including an optional choke cap."""
        if np.isfinite(self.M) and p >= self.M:
            return 0.0

        q1 = self.q0 + (p - self.p0) / self.g_U
        q2 = self.q0 + (p - self.p0) / self.g_L
        q_slope = max(q1, q2, 0.0)

        if np.isfinite(self.M):
            # From P(0) <= M and P'(q) <= g_U: P(q) <= M + g_U*q, so
            # q(p) <= (p - M)/g_U for p < M (and =0 for p >= M).
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


def _inner_x_two_market(
    Qbar: float,
    bounds1: Tuple[float, float],
    bounds2: Tuple[float, float],
    c1: float,
    c2: float,
    kappa1: float,
    kappa2: float,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """
    Solve the 2-market inner problem:
      minimize/maximize sum_i (x_i - c_i)^2/(2*kappa_i)
      s.t. x1 + x2 = Qbar, x_i in [l_i, u_i].

    For kappa_i > 0, the KKT solution is unique; we clamp to the feasible
    interval implied by the box constraints.
    """
    l1, u1 = bounds1
    l2, u2 = bounds2

    # Feasible x1 interval implied by x2 = Qbar - x1.
    lo = max(l1, Qbar - u2)
    hi = min(u1, Qbar - l2)
    if lo > hi + tol:
        raise ValueError(f"Infeasible X(p): x1 in [{lo}, {hi}] is empty.")

    # Degenerate cases: if a market has no slope uncertainty (kappa ~ 0),
    # its terminal value is effectively pinned down (no "peel-off" interval).
    if kappa1 <= tol and kappa2 <= tol:
        x1 = float(np.clip(c1, lo, hi))
        x2 = Qbar - x1
        return x1, x2
    if kappa1 <= tol:
        x1 = float(np.clip(c1, lo, hi))
        x2 = Qbar - x1
        return x1, x2
    if kappa2 <= tol:
        # Pin x2 near c2 => x1 = Qbar - x2
        x1 = float(np.clip(Qbar - c2, lo, hi))
        x2 = Qbar - x1
        return x1, x2

    x1_star = (kappa1 * Qbar + kappa2 * c1 - kappa1 * c2) / (kappa1 + kappa2)
    x1 = float(np.clip(x1_star, lo, hi))
    x2 = Qbar - x1
    return x1, x2


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
    Solve the inner "peel-off" problem for n markets:

        minimize  sum_i (x_i - c_i)^2 / (2*kappa_i)
        s.t.      sum_i x_i = Qbar
                  l_i <= x_i <= u_i

    With kappa_i > 0 this is a strictly convex QP with a unique KKT solution:
        x_i(λ) = clip(c_i - λ*kappa_i, [l_i, u_i])
    where λ is chosen so sum_i x_i(λ) = Qbar.
    """
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    c = np.asarray(c, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    if not (l.shape == u.shape == c.shape == kappa.shape):
        raise ValueError("l, u, c, kappa must have identical shapes.")

    if np.any(u < l - 1e-12):
        raise ValueError("Invalid bounds: some u < l.")

    # Handle (rare) degenerate markets with ~zero kappa by pinning them at c.
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

    # Bracket λ so that x(λ_low)=u and x(λ_high)=l.
    lam_low = float(np.min((cf - uf) / kf))
    lam_high = float(np.max((cf - lf) / kf))

    # If already at a bound, return it.
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
            # Need smaller total => increase λ.
            lam_low = lam
        else:
            lam_high = lam

    x[flex] = np.clip(cf - lam * kf, lf, uf)
    return x


def _objective_at_p_two_market(
    p: float,
    Qbar: float,
    m1: MarketSpec,
    m2: MarketSpec,
    env1: _PrecomputedEnvelopes,
    env2: _PrecomputedEnvelopes,
    mode: Mode,
) -> Tuple[float, Tuple[float, float]]:
    """
    Evaluate the lemma objective at (p, x*(p)) for a fixed (m1, m2).

    Returns
    -------
    value, (x1, x2)
    """
    # Feasibility boxes at p
    l1, u1 = env1.ell(p), env1.u(p)
    l2, u2 = env2.ell(p), env2.u(p)
    if (l1 + l2) > Qbar + 1e-10 or (u1 + u2) < Qbar - 1e-10:
        raise ValueError("p not feasible (outside I).")

    forward1 = p >= m1.p0
    forward2 = p >= m2.p0

    # Envelope integrals (depend on direction and bound type)
    val = Qbar * p - (m1.q0 * m1.p0 + m2.q0 * m2.p0)

    def _int(env: _PrecomputedEnvelopes, a: float, b: float, which: str) -> float:
        return env.int_ell(a, b) if which == "ell" else env.int_u(a, b)

    # Market 1 integral contribution
    if mode == "upper":
        if forward1:
            val -= _int(env1, m1.p0, p, "ell")
        else:
            val += _int(env1, p, m1.p0, "u")
    else:  # mode == "lower"
        if forward1:
            val -= _int(env1, m1.p0, p, "u")
        else:
            val += _int(env1, p, m1.p0, "ell")

    # Market 2 integral contribution
    if mode == "upper":
        if forward2:
            val -= _int(env2, m2.p0, p, "ell")
        else:
            val += _int(env2, p, m2.p0, "u")
    else:
        if forward2:
            val -= _int(env2, m2.p0, p, "u")
        else:
            val += _int(env2, p, m2.p0, "ell")

    # Quadratic "peel-off" penalty (sign depends on bound).
    # Centers swap across upper/lower and forward/backward cases.
    if mode == "upper":
        c1 = l1 if forward1 else u1
        c2 = l2 if forward2 else u2
        sign = -1.0
    else:
        c1 = u1 if forward1 else l1
        c2 = u2 if forward2 else l2
        sign = +1.0

    x1, x2 = _inner_x_two_market(
        Qbar=Qbar,
        bounds1=(l1, u1),
        bounds2=(l2, u2),
        c1=c1,
        c2=c2,
        kappa1=m1.kappa,
        kappa2=m2.kappa,
    )

    pen = 0.0
    if m1.kappa > 0:
        pen += (x1 - c1) ** 2 / (2.0 * m1.kappa)
    if m2.kappa > 0:
        pen += (x2 - c2) ** 2 / (2.0 * m2.kappa)

    val += sign * pen
    return float(val), (float(x1), float(x2))


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
    Solve the Lemma-1 joint robust-bounds problem for n markets.

    This is the general-n version of `solve_joint_bound_two_market`, using the
    same grid-search strategy for the outer p-optimization and a KKT/bisection
    solver for the inner x-optimization.
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

    # Precompute envelopes and signed integrals int_{p0}^{p} ell/u for each market.
    ell_vals = np.empty((len(ms), len(p_grid)), dtype=float)
    u_vals = np.empty((len(ms), len(p_grid)), dtype=float)
    int_ell = np.empty_like(ell_vals)
    int_u = np.empty_like(u_vals)

    for i, m in enumerate(ms):
        env = _precompute_market(m, p_grid)
        ell_vals[i, :] = env.ell_vals
        u_vals[i, :] = env.u_vals
        # Signed integrals from p0_i to each p_grid point.
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

        # Envelope integrals (the signed `int_*` arrays make forward/backward automatic).
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


def solve_joint_bound_two_market(
    m1: MarketSpec,
    m2: MarketSpec,
    Qbar: Optional[float] = None,
    *,
    mode: Mode,
    p_min: float = 0.0,
    p_max: Optional[float] = None,
    n_grid: int = 4001,
) -> Dict:
    """
    Solve the (p, x) program for n=2 markets.

    Parameters
    ----------
    Qbar:
        Total quantity constraint. If omitted, uses m1.q0 + m2.q0.
    mode:
        "upper" for overline Phi, "lower" for underline Phi.
    p_min, p_max:
        Price search window. If p_max omitted, uses max finite choke or 10.
    n_grid:
        Grid points for precomputation and coarse maximization/minimization.
        Larger -> more accurate but slower.
    """
    if Qbar is None:
        Qbar = m1.q0 + m2.q0
    Qbar = float(Qbar)

    if p_max is None:
        finite_chokes = [m for m in [m1.M, m2.M] if np.isfinite(m)]
        p_max = max([10.0, m1.p0, m2.p0, *finite_chokes])
    p_max = float(p_max)

    if not (p_max > p_min):
        raise ValueError("Need p_max > p_min.")
    if n_grid < 101:
        raise ValueError("Need a reasonably fine grid (n_grid >= 101).")

    p_grid = np.linspace(p_min, p_max, n_grid)
    env1 = _precompute_market(m1, p_grid)
    env2 = _precompute_market(m2, p_grid)

    L = env1.ell_vals + env2.ell_vals
    U = env1.u_vals + env2.u_vals
    feasible = (L <= Qbar + 1e-10) & (U >= Qbar - 1e-10)

    if not np.any(feasible):
        raise ValueError(
            "No feasible p found (I is empty). Check anchors/slope bounds/Qbar."
        )

    obj = np.full_like(p_grid, fill_value=np.nan, dtype=float)
    xs = np.full((len(p_grid), 2), fill_value=np.nan, dtype=float)

    for i, p in enumerate(p_grid):
        if not feasible[i]:
            continue
        val, (x1, x2) = _objective_at_p_two_market(
            float(p), Qbar, m1, m2, env1, env2, mode
        )
        obj[i] = val
        xs[i, 0] = x1
        xs[i, 1] = x2

    if mode == "upper":
        i_star = int(np.nanargmax(obj))
    else:
        i_star = int(np.nanargmin(obj))

    p_star = float(p_grid[i_star])
    Phi_star = float(obj[i_star])
    x_star = (float(xs[i_star, 0]), float(xs[i_star, 1]))

    return {
        "mode": mode,
        "Phi": Phi_star,
        "p_star": p_star,
        "x_star": x_star,
        "p_grid": p_grid,
        "objective_grid": obj,
        "x_grid": xs,
        "L_grid": L,
        "U_grid": U,
    }
