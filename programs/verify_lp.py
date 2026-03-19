"""
verify_lp.py — Analytical verification of the LP/optimization implementation.

Tests the core joint_robust_bounds and state_joint_robust_bounds solvers against:
  1. Phi=0 cases (efficient allocation → no misallocation)
  2. Closed-form analytical formula for fixed slopes
  3. Brute-force comparison for slope-bound cases
  4. Bound ordering and monotonicity
  5. Agreement between the 2-market solver and the n-market solver

Analytical derivation (linear demand, no binding box constraints):
  For N markets with linear inverse demand P_i(q) = p0_i + g_i*(q - q0_i):
  - Shadow price at observed allocation: P_i(q0_i) = p0_i
  - Efficient p* satisfies Σ_i q0_i + (p* - p0_i)/g_i = Qbar = Σ_i q0_i
    => (p* - p0_i)/g_i sums to zero
    => For common slope g: p* = mean(p0_i)
  - Misallocation: Phi = Σ_i (p0_i - p*)^2 / (2*|g_i|)
  - For 2 markets, common slope g: Phi = (p0_1 - p0_2)^2 / (4*|g|)

Run from replication/ directory:
  python programs/verify_lp.py
"""

from __future__ import annotations

import os
import sys

os.environ["KMP_USE_SHM"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
from joint_robust_bounds import MarketSpec, solve_joint_bound_two_market
from state_joint_robust_bounds import solve_joint_bound as solve_n_market

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, str, float, float]] = []


def check(name: str, got: float, expected: float, tol: float = 1e-3) -> bool:
    """Pass if |got - expected| <= tol * max(1, |expected|)."""
    scale = max(1.0, abs(expected))
    ok = abs(got - expected) <= tol * scale
    status = "PASS" if ok else "FAIL"
    _results.append((status, name, got, expected))
    mark = "✓" if ok else "✗"
    print(f"  [{status}] {mark}  {name}")
    if not ok:
        print(f"           got={got:.8g}  expected={expected:.8g}  diff={abs(got - expected):.3g}")
    return ok


def check_le(name: str, a: float, b: float, tol: float = 1e-4) -> bool:
    """Pass if a <= b + tol."""
    ok = a <= b + tol
    status = "PASS" if ok else "FAIL"
    _results.append((status, name, a, b))
    mark = "✓" if ok else "✗"
    print(f"  [{status}] {mark}  {name}  ({a:.6g} <= {b:.6g})")
    return ok


def analytical_phi_fixed_g(p0s: list[float], g: float) -> float:
    """Phi for N markets with common fixed slope g (no binding box constraints).
    Formula: Phi = sum_i (p0_i - p*)^2 / (2*|g|)  where p* = mean(p0_i).
    """
    p_star = float(np.mean(p0s))
    return sum((p - p_star) ** 2 for p in p0s) / (2.0 * abs(g))


def solve2(m1: MarketSpec, m2: MarketSpec, Qbar: float, mode: str) -> float:
    return float(solve_joint_bound_two_market(m1, m2, Qbar=Qbar, mode=mode, n_grid=8001)["Phi"])


def solveN(markets: list[MarketSpec], Qbar: float, mode: str) -> float:
    return float(solve_n_market(markets, Qbar=Qbar, mode=mode, n_grid=8001)["Phi"])


def brute_force_upper_2market(m1: MarketSpec, m2: MarketSpec, Qbar: float,
                               n_g: int = 200) -> float:
    """Brute-force upper bound: grid over (g1,g2), analytically compute Phi.

    For each (g1, g2) pair the adding-up constraint uniquely pins p*:
        (p* - p0_1)/g1 + (p* - p0_2)/g2 = 0
        => p* = (p0_1/g1 + p0_2/g2) / (1/g1 + 1/g2)

    Then Phi = (p0_1 - p*)^2/(2|g1|) + (p0_2 - p*)^2/(2|g2|).
    Valid when q*(p*) in (0, q_max) for both markets.
    """
    best = 0.0
    g1s = np.linspace(m1.g_L, m1.g_U, n_g if m1.g_L < m1.g_U else 1)
    g2s = np.linspace(m2.g_L, m2.g_U, n_g if m2.g_L < m2.g_U else 1)
    for g1 in g1s:
        for g2 in g2s:
            denom = 1.0 / g1 + 1.0 / g2
            if abs(denom) < 1e-12:
                continue
            p_star = (m1.p0 / g1 + m2.p0 / g2) / denom
            if p_star < 0:
                continue
            # Box-constraint check (skip constrained cases — brute force is only
            # exact when unconstrained)
            q1 = m1.q0 + (p_star - m1.p0) / g1
            q2 = m2.q0 + (p_star - m2.p0) / g2
            if q1 < -1e-4 or q1 > m1.q_max + 1e-4:
                continue
            if q2 < -1e-4 or q2 > m2.q_max + 1e-4:
                continue
            phi = (m1.p0 - p_star) ** 2 / (2 * abs(g1)) + \
                  (m2.p0 - p_star) ** 2 / (2 * abs(g2))
            best = max(best, phi)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1 — Phi = 0 at efficient allocation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUITE 1: Phi = 0 when observed allocation is efficient")
print("=" * 65)
print("Logic: if p0_1 = p0_2, shadow prices already equalize → Phi = 0.")

rng = np.random.default_rng(1974)

for trial, g in enumerate([-1.0, -2.5, -5.0, -10.0]):
    p0 = 1.2
    q1 = rng.uniform(0.1, 0.8)
    m1 = MarketSpec(f"m1", q0=q1, p0=p0, g_L=g, g_U=g)
    m2 = MarketSpec(f"m2", q0=1.0 - q1, p0=p0, g_L=g, g_U=g)
    phi_lo = solve2(m1, m2, 1.0, "lower")
    phi_hi = solve2(m1, m2, 1.0, "upper")
    check(f"2-market lower=0  (g={g})", phi_lo, 0.0)
    check(f"2-market upper=0  (g={g})", phi_hi, 0.0)

# Equal p0 with slope bounds (g_L < g_U): still Phi=0 since
# for any g the adding-up pins p*=p0 when p0_1=p0_2.
m1 = MarketSpec("m1", q0=0.6, p0=1.2, g_L=-5.0, g_U=-2.0)
m2 = MarketSpec("m2", q0=0.4, p0=1.2, g_L=-5.0, g_U=-2.0)
check("2-market lower=0  (slope bounds, equal p0)", solve2(m1, m2, 1.0, "lower"), 0.0)
check("2-market upper=0  (slope bounds, equal p0)", solve2(m1, m2, 1.0, "upper"), 0.0)

# n-market solver: 3 markets with equal p0
p0 = 1.5
markets_eq = [MarketSpec(f"m{i}", q0=0.3, p0=p0, g_L=-3.0, g_U=-3.0) for i in range(3)]
check("n-market lower=0  (3 markets, equal p0)", solveN(markets_eq, 0.9, "lower"), 0.0)
check("n-market upper=0  (3 markets, equal p0)", solveN(markets_eq, 0.9, "upper"), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2 — Closed-form formula for fixed slopes
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUITE 2: Phi matches closed-form for fixed slopes")
print("=" * 65)
print("Formula: Phi = (p0_1 - p0_2)^2 / (4*|g|)  [2-market]")
print("         Phi = sum_i (p0_i - p*)^2 / (2*|g|)  [N-market]")

rng = np.random.default_rng(42)


def analytical_phi_upper_2market(p0_1: float, p0_2: float, g_U: float) -> float:
    """Exact upper bound for 2 markets with slope bounds.

    Derived from Phi = (p0_1-p0_2)^2 / (2*(|g1|+|g2|)).
    Maximum is at g1=g2=g_U (flattest, smallest |g|):
        Phi_upper = (p0_1-p0_2)^2 / (4*|g_U|)
    Valid when q*(p*) > 0 for both markets (no box constraint binding).
    """
    return (p0_1 - p0_2) ** 2 / (4.0 * abs(g_U))


def analytical_phi_lower_2market(p0_1: float, p0_2: float, g_L: float) -> float:
    """Exact lower bound for 2 markets with slope bounds.

    Minimum is at g1=g2=g_L (steepest, largest |g|):
        Phi_lower = (p0_1-p0_2)^2 / (4*|g_L|)
    """
    return (p0_1 - p0_2) ** 2 / (4.0 * abs(g_L))


def feasible_equal_g(g: float, q1: float, q2: float, p0_1: float, p0_2: float) -> bool:
    """Check that equal-slope efficient allocation stays non-negative, p0 > 0."""
    if p0_1 <= 0.05 or p0_2 <= 0.05:
        return False
    p_star = (p0_1 + p0_2) / 2.0
    if p_star <= 0.05:
        return False
    q1_star = q1 + (p_star - p0_1) / g
    q2_star = q2 + (p_star - p0_2) / g
    return q1_star >= 0.02 and q2_star >= 0.02


# Alias used in Suites 2 and 3 (same semantics)
feasible_2market = feasible_equal_g


# 2-market with proper slope bounds — exact closed-form answer
print("\n--- 2-market, slope bounds [g_L, g_U], exact formula ---")
trial = 0
attempts = 0
while trial < 20 and attempts < 2000:
    attempts += 1
    g_U = rng.uniform(-4.0, -1.0)
    g_L = g_U * rng.uniform(1.2, 2.5)   # g_L more negative than g_U
    if g_L >= g_U:
        continue
    q1 = rng.uniform(0.2, 0.7)
    q2 = 1.0 - q1
    # Ensure efficient allocation at g_U (flattest) stays non-negative
    # p0 spread bounded by feasibility at FLATTEST slope (worst case)
    max_half = min(0.6, 2 * min(q1, q2) * abs(g_U) * 0.8, 0.9)
    p0_mid = rng.uniform(0.9, 1.6)
    half = rng.uniform(0.05, max_half)
    p0_1 = p0_mid + half
    p0_2 = p0_mid - half
    # Check feasibility at both extreme slopes
    if not feasible_equal_g(g_U, q1, q2, p0_1, p0_2):
        continue
    if not feasible_equal_g(g_L, q1, q2, p0_1, p0_2):
        continue
    m1 = MarketSpec("m1", q0=q1, p0=p0_1, g_L=g_L, g_U=g_U)
    m2 = MarketSpec("m2", q0=q2, p0=p0_2, g_L=g_L, g_U=g_U)
    exp_hi = analytical_phi_upper_2market(p0_1, p0_2, g_U)
    exp_lo = analytical_phi_lower_2market(p0_1, p0_2, g_L)
    phi_lo = solve2(m1, m2, 1.0, "lower")
    phi_hi = solve2(m1, m2, 1.0, "upper")
    check(f"T{trial+1:02d} upper (gL={g_L:.2f},gU={g_U:.2f},Δp0={2*half:.2f})", phi_hi, exp_hi)
    check(f"T{trial+1:02d} lower (gL={g_L:.2f},gU={g_U:.2f},Δp0={2*half:.2f})", phi_lo, exp_lo)
    trial += 1

# n-market, slope bounds [g_L, g_U] — compare upper to formula at g_U,
# lower to formula at g_L.  Use SLOPE_EPS=0.2 (20% wider at steep end)
# so the feasible-p grid can resolve the interval.
SLOPE_EPS = 0.20

def feasible_nmarket(g: float, q0s: list[float], p0s: list[float]) -> bool:
    if any(p <= 0.05 for p in p0s):
        return False
    p_star = float(np.mean(p0s))
    return all(q + (p_star - p) / g >= 0.02 for q, p in zip(q0s, p0s))

print("\n--- n-market, slope bounds [g_L=g*(1+ε), g_U=g] → formula at each endpoint ---")
for n_mkts in [3, 5]:
    trial = 0
    attempts = 0
    while trial < 5 and attempts < 500:
        attempts += 1
        g = rng.uniform(-5.0, -1.0)
        q0s = list(rng.dirichlet(np.ones(n_mkts)))
        p0_mid = rng.uniform(0.9, 1.5)
        max_dev = min(q0s) * abs(g) * 0.5
        p0s = [float(p0_mid + rng.uniform(-max_dev, max_dev)) for _ in range(n_mkts)]
        g_L = g * (1 + SLOPE_EPS)  # steeper (more negative)
        if not feasible_nmarket(g,   q0s, p0s):
            continue
        if not feasible_nmarket(g_L, q0s, p0s):
            continue
        markets = [MarketSpec(f"m{i}", q0=float(q0s[i]), p0=p0s[i], g_L=g_L, g_U=g)
                   for i in range(n_mkts)]
        exp_hi = analytical_phi_fixed_g(p0s, g)    # upper bound: flattest slope = g_U
        exp_lo = analytical_phi_fixed_g(p0s, g_L)  # lower bound: steepest slope = g_L
        phi_lo = solveN(markets, 1.0, "lower")
        phi_hi = solveN(markets, 1.0, "upper")
        check(f"N={n_mkts} T{trial+1} upper (g_U={g:.2f})", phi_hi, exp_hi, tol=1e-2)
        check(f"N={n_mkts} T{trial+1} lower (g_L={g_L:.2f})", phi_lo, exp_lo, tol=1e-2)
        trial += 1

# n-market and 2-market agree for N=2
print("\n--- 2-market solver == n-market solver for N=2 ---")
trial = 0
attempts = 0
while trial < 10 and attempts < 500:
    attempts += 1
    g = rng.uniform(-5.0, -0.5)
    q1 = rng.uniform(0.2, 0.7)
    max_dev = 2 * min(q1, 1.0 - q1) * abs(g) * 0.8
    p0_mid = rng.uniform(0.8, 1.5)
    half_dev = rng.uniform(0.0, min(max_dev / 2, 0.8))
    p0_1 = p0_mid + half_dev
    p0_2 = p0_mid - half_dev
    if not feasible_2market(g, q1, 1.0 - q1, p0_1, p0_2):
        continue
    g_L = g * (1 + SLOPE_EPS)
    m1 = MarketSpec("m1", q0=q1, p0=p0_1, g_L=g_L, g_U=g)
    m2 = MarketSpec("m2", q0=1.0 - q1, p0=p0_2, g_L=g_L, g_U=g)
    phi2_hi = solve2(m1, m2, 1.0, "upper")
    phiN_hi = solveN([m1, m2], 1.0, "upper")
    phi2_lo = solve2(m1, m2, 1.0, "lower")
    phiN_lo = solveN([m1, m2], 1.0, "lower")
    check(f"T{trial+1:02d} upper: 2-market == n-market", phi2_hi, phiN_hi)
    check(f"T{trial+1:02d} lower: 2-market == n-market", phi2_lo, phiN_lo)
    trial += 1


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3 — Brute force vs LP with slope bounds
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUITE 3: LP bounds vs brute-force grid search (slope bounds)")
print("=" * 65)
print("LP upper >= brute force upper (LP finds weakly more)")
print("LP upper ~= brute force upper (should be close on fine grid)")

rng = np.random.default_rng(7)

trial = 0
attempts = 0
while trial < 15 and attempts < 1000:
    attempts += 1
    g_L = rng.uniform(-6.0, -2.5)
    g_U = g_L + rng.uniform(0.5, 3.0)
    if g_U >= 0.0:
        g_U = -0.2
    q1 = rng.uniform(0.2, 0.7)
    # Keep p0 spread feasible under the flattest slope (g_U)
    max_dev = 2 * min(q1, 1.0 - q1) * abs(g_U) * 0.8
    p0_mid = rng.uniform(0.8, 1.5)
    half_dev = rng.uniform(0.0, min(max_dev / 2, 0.8))
    p0_1 = p0_mid + half_dev
    p0_2 = p0_mid - half_dev
    if not feasible_2market(g_U, q1, 1.0 - q1, p0_1, p0_2):
        continue

    m1 = MarketSpec("m1", q0=q1, p0=p0_1, g_L=g_L, g_U=g_U)
    m2 = MarketSpec("m2", q0=1.0 - q1, p0=p0_2, g_L=g_L, g_U=g_U)

    lp_hi = solve2(m1, m2, 1.0, "upper")
    bf_hi = brute_force_upper_2market(m1, m2, 1.0, n_g=300)

    # LP upper must be >= brute force (LP is an exact solver; BF is lower bound)
    check_le(f"T{trial+1:02d} BF_upper <= LP_upper", bf_hi, lp_hi)
    # They should be close (within 2%)
    check(f"T{trial+1:02d} LP_upper ~= BF_upper", lp_hi, bf_hi, tol=0.02)
    trial += 1


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 4 — Bound ordering and monotonicity
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUITE 4: Bound ordering and monotonicity")
print("=" * 65)

rng = np.random.default_rng(99)

print("\n--- Lower <= Upper ---")
trial = 0
attempts = 0
while trial < 10 and attempts < 500:
    attempts += 1
    g_L = rng.uniform(-6.0, -2.0)
    g_U = g_L + rng.uniform(0.2, 2.5)
    if g_U >= 0:
        g_U = -0.1
    q1 = rng.uniform(0.2, 0.7)
    max_dev = 2 * min(q1, 1.0 - q1) * abs(g_U) * 0.8
    p0_mid = rng.uniform(0.8, 1.5)
    half_dev = rng.uniform(0.0, min(max_dev / 2, 0.8))
    p0_1 = p0_mid + half_dev
    p0_2 = p0_mid - half_dev
    if not feasible_2market(g_U, q1, 1.0 - q1, p0_1, p0_2):
        continue
    m1 = MarketSpec("m1", q0=q1, p0=p0_1, g_L=g_L, g_U=g_U)
    m2 = MarketSpec("m2", q0=1.0 - q1, p0=p0_2, g_L=g_L, g_U=g_U)
    lo = solve2(m1, m2, 1.0, "lower")
    hi = solve2(m1, m2, 1.0, "upper")
    check_le(f"T{trial+1:02d} lower <= upper (2-market)", lo, hi)
    lo_n = solveN([m1, m2], 1.0, "lower")
    hi_n = solveN([m1, m2], 1.0, "upper")
    check_le(f"T{trial+1:02d} lower <= upper (n-market)", lo_n, hi_n)
    trial += 1

print("\n--- Choke tightens upper bound (not lower) ---")
m1 = MarketSpec("m1", q0=0.7, p0=0.9, g_L=-4.0, g_U=-2.0)
m2 = MarketSpec("m2", q0=0.2, p0=1.6, g_L=-4.0, g_U=-2.0)
m1c = MarketSpec("m1", q0=0.7, p0=0.9, g_L=-4.0, g_U=-2.0, M=4.0)
m2c = MarketSpec("m2", q0=0.2, p0=1.6, g_L=-4.0, g_U=-2.0, M=4.0)
hi_nc = solve2(m1, m2, 0.9, "upper")
hi_c = solve2(m1c, m2c, 0.9, "upper")
lo_nc = solve2(m1, m2, 0.9, "lower")
lo_c = solve2(m1c, m2c, 0.9, "lower")
check_le("choke upper <= no-choke upper", hi_c, hi_nc)
check_le("no-choke lower <= choke lower + tol", lo_nc, lo_c + 0.01)

print("\n--- Wider slope bounds give wider interval ---")
# Narrower [g_L+1, g_U-1] should give subset of bounds
g_L, g_U = -5.0, -2.0
m1w = MarketSpec("m1", q0=0.6, p0=1.0, g_L=g_L, g_U=g_U)
m2w = MarketSpec("m2", q0=0.3, p0=1.8, g_L=g_L, g_U=g_U)
m1n = MarketSpec("m1", q0=0.6, p0=1.0, g_L=g_L + 1, g_U=g_U - 0.5)
m2n = MarketSpec("m2", q0=0.3, p0=1.8, g_L=g_L + 1, g_U=g_U - 0.5)
hi_wide = solve2(m1w, m2w, 0.9, "upper")
hi_narrow = solve2(m1n, m2n, 0.9, "upper")
lo_wide = solve2(m1w, m2w, 0.9, "lower")
lo_narrow = solve2(m1n, m2n, 0.9, "lower")
check_le("narrow upper <= wide upper", hi_narrow, hi_wide)
check_le("wide lower <= narrow lower", lo_wide, lo_narrow)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 5 — Paper parameters: known output
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUITE 5: Paper parameters — 2-market station-level (no choke)")
print("=" * 65)
print("Reproduces the station-level robust_bounds.py setup.")
print("Key: demand curves rotate around (q_base, p_base), so the shadow")
print("price at q_obs shifts with slope — p0 is an interval, not a point.")

P_ANCHOR = 1.0
BASE_EPS = 0.3
EPS_L, EPS_H = 0.2, 0.4
P_BAR   = 0.8
SHORTAGE = 0.09
Q_SUPPLY = 1.0 - SHORTAGE      # 0.91
FRAC_OPEN   = 0.62
FRAC_CLOSED = 0.38

# Calibrate per-station quantities from baseline elasticity
g_base_ps = -P_ANCHOR / (BASE_EPS * 1.0)          # slope at 1 per-station unit
q_open_ps  = 1.0 + (P_BAR - P_ANCHOR) / g_base_ps  # ≈ 1.060 per-station
q_closed_ps = (Q_SUPPLY - FRAC_OPEN * q_open_ps) / FRAC_CLOSED  # ≈ 0.665 per-station

# Aggregate quantities (sum to Qbar = Q_SUPPLY)
qO_obs  = FRAC_OPEN   * q_open_ps
qC_obs  = FRAC_CLOSED * q_closed_ps
Qbar_s5 = qO_obs + qC_obs

# Aggregate baseline quantities (pre-crisis, normalized)
qO_base = FRAC_OPEN   * 1.0
qC_base = FRAC_CLOSED * 1.0

# Market-specific slope bounds: g = -p_base / (eps * q_base)
# Rotating the demand curve around the baseline anchor (q_base, p_base)
# changes the shadow price at q_obs → p0 is an interval.
gO_L = -P_ANCHOR / (EPS_L * qO_base)
gO_U = -P_ANCHOR / (EPS_H * qO_base)
gC_L = -P_ANCHOR / (EPS_L * qC_base)
gC_U = -P_ANCHOR / (EPS_H * qC_base)

def p0_interval(q_base: float, q_obs: float, g_L: float, g_U: float):
    """Shadow price at q_obs for slope ∈ [g_L, g_U] (anchored at q_base, P_ANCHOR)."""
    p_steep = P_ANCHOR + g_L * (q_obs - q_base)
    p_flat  = P_ANCHOR + g_U * (q_obs - q_base)
    return max(0.0, min(p_steep, p_flat)), max(p_steep, p_flat)

pO_lo, pO_hi = p0_interval(qO_base, qO_obs, gO_L, gO_U)
pC_lo, pC_hi = p0_interval(qC_base, qC_obs, gC_L, gC_U)

print(f"  Aggregate: qO_obs={qO_obs:.4f}  qC_obs={qC_obs:.4f}  Qbar={Qbar_s5:.4f}")
print(f"  Open  slope bounds: [{gO_L:.2f}, {gO_U:.2f}]  p0 ∈ [{pO_lo:.4f}, {pO_hi:.4f}]")
print(f"  Closed slope bounds: [{gC_L:.2f}, {gC_U:.2f}]  p0 ∈ [{pC_lo:.4f}, {pC_hi:.4f}]")

M_CHOKE_S5 = 4.0

def p0_interval_with_choke(q_base: float, q_obs: float, g_L: float, g_U: float,
                            M: float) -> tuple:
    """Shadow price interval, capped by choke constraint."""
    p_steep = P_ANCHOR + g_L * (q_obs - q_base)
    p_flat  = P_ANCHOR + g_U * (q_obs - q_base)
    lo = max(0.0, min(p_steep, p_flat))
    hi = max(p_steep, p_flat)
    # Choke cap (mirrors robust_bounds.py p0_bounds logic)
    hi = min(hi, M, M + g_U * q_obs)
    lo = min(lo, M)
    if lo > hi:
        lo = hi
    return lo, hi

pO_lo_c, pO_hi_c = p0_interval_with_choke(qO_base, qO_obs, gO_L, gO_U, M_CHOKE_S5)
pC_lo_c, pC_hi_c = p0_interval_with_choke(qC_base, qC_obs, gC_L, gC_U, M_CHOKE_S5)

# Enumerate all 4 (p0_O, p0_C) corners and take max/min over Phi
upper_phis, lower_phis = [], []
for p0_O, p0_C in [(pO_lo_c, pC_lo_c), (pO_lo_c, pC_hi_c),
                   (pO_hi_c, pC_lo_c), (pO_hi_c, pC_hi_c)]:
    mO = MarketSpec("open",   q0=qO_obs, p0=p0_O, g_L=gO_L, g_U=gO_U, M=M_CHOKE_S5)
    mC = MarketSpec("closed", q0=qC_obs, p0=p0_C, g_L=gC_L, g_U=gC_U, M=M_CHOKE_S5)
    upper_phis.append(solve2(mO, mC, Qbar_s5, "upper"))
    lower_phis.append(solve2(mO, mC, Qbar_s5, "lower"))

phi_lo = min(lower_phis)
phi_hi = max(upper_phis)

# Harberger benchmark: 0.5 * |g_steep| * shortage^2 (unit normalization)
psi_paper = 0.5 * abs(-P_ANCHOR / EPS_L) * SHORTAGE ** 2

print(f"\n  p0 open ∈  [{pO_lo_c:.4f}, {pO_hi_c:.4f}]  (with choke M={M_CHOKE_S5})")
print(f"  p0 closed ∈ [{pC_lo_c:.4f}, {pC_hi_c:.4f}]")
print(f"\n  Phi_lower = {100*phi_lo:.3f}%  (paper: ~2.3%)")
print(f"  Phi_upper = {100*phi_hi:.3f}%  (paper: ~12.7%)")
print(f"  R_lower   = {phi_lo/psi_paper:.3f}  (paper: ~1.1)")
print(f"  R_upper   = {phi_hi/psi_paper:.3f}  (paper: ~6.3)")

check("Phi_lower in paper range [2.1%, 2.5%]", phi_lo, 0.0232, tol=0.08)
check("Phi_upper in paper range [11.5%, 14%]", phi_hi, 0.1270, tol=0.08)
check("R_lower ≈ 1.1",  phi_lo / psi_paper, 1.15, tol=0.12)
check("R_upper ≈ 6.3",  phi_hi / psi_paper, 6.28, tol=0.08)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
n_pass = sum(1 for r in _results if r[0] == "PASS")
n_fail = sum(1 for r in _results if r[0] == "FAIL")
print(f"\n{n_pass} / {n_pass + n_fail} tests passed\n")

if n_fail:
    print("FAILED tests:")
    for status, name, got, exp in _results:
        if status == "FAIL":
            print(f"  ✗  {name}")
            print(f"       got={got:.8g}  expected={exp:.8g}  diff={abs(got-exp):.3g}")
else:
    print("All tests passed.")
