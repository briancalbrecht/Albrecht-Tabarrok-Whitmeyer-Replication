"""Figure 5–6 + Table 1: robust bounds with rationing allocation."""

import argparse
import json
import os

# Sandbox-safe OpenMP defaults: avoid SHM-backed runtime initialization failures.
os.environ["KMP_USE_SHM"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set, Tuple
from pathlib import Path

import pandas as pd

from joint_robust_bounds import MarketSpec, solve_joint_bound_two_market
from state_subset import resolve_state_subset


# Parameters

THIS_DIR = Path(__file__).resolve().parent
# Standalone replication package root (paper_figure_replication/).
REPO_ROOT = THIS_DIR.parent

# Anchor price (normalized baseline)
P_ANCHOR = 1.0

# Baseline elasticity for single-point calibration
BASE_EPSILON = 0.3

# Elasticity bounds
EPSILON_L = 0.2   # Low elasticity bound -> steeper inverse demand
EPSILON_H = 0.4   # High elasticity bound -> flatter inverse demand

# Slope bounds: g = -1/epsilon
G_STEEP = -P_ANCHOR / EPSILON_L  # = -5.00
G_FLAT = -P_ANCHOR / EPSILON_H   # = -2.50

# Choke price
M_CHOKE = 4.0

# Rationing parameters
P_BAR = 0.8  # Controlled price
FRACTION_OPEN = 0.62
FRACTION_CLOSED = 0.38  # Combined closed + limiting
# Baseline aggregate shortage calibration from Yergin (1991).
SHORTAGE_RATE = 0.09
Q_SUPPLY = 1.0 - SHORTAGE_RATE  # = 0.91


def _clean_state_set(values: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for raw in values:
        s = str(raw).strip()
        if s and s.lower() != "nan":
            out.add(s)
    return out


def derive_station_fractions_from_aaa(
    *,
    include_states: Optional[Set[str]] = None,
) -> Dict[str, object]:
    """
    Compute open/closed station fractions from AAA station counts.

    If include_states is provided, keep only those states before aggregation.
    """
    aaa_path = THIS_DIR.parent / "data" / "AAA Fuel Report 1974 w State Names and total stations simplified.xlsx"
    df = pd.read_excel(
        aaa_path,
        usecols=["State", "Stations surveyed", "Limiting Purchases", "Out of Fuel"],
    )
    df["State"] = df["State"].astype(str).str.strip()
    df = df[df["State"] != "District of Columbia"].copy()

    if include_states is not None:
        keep = _clean_state_set(include_states)
        df = df[df["State"].isin(keep)].copy()

    for col in ["Stations surveyed", "Limiting Purchases", "Out of Fuel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Stations surveyed"].notna() & (df["Stations surveyed"] > 0)].copy()
    if df.empty:
        raise ValueError("No AAA states left after applying subset filter.")

    limiting = df["Limiting Purchases"].fillna(0.0).clip(lower=0.0)
    out_of_fuel = df["Out of Fuel"].fillna(0.0).clip(lower=0.0)
    closed_count = (limiting + out_of_fuel).sum()
    surveyed = float(df["Stations surveyed"].sum())
    open_count = surveyed - float(closed_count)

    if surveyed <= 0:
        raise ValueError("AAA surveyed station total is non-positive.")
    open_frac = float(open_count / surveyed)
    closed_frac = float(closed_count / surveyed)
    if open_frac <= 0 or closed_frac <= 0:
        raise ValueError(
            f"Derived invalid station shares: open={open_frac:.6f}, closed={closed_frac:.6f}."
        )

    return {
        "open_frac": open_frac,
        "closed_frac": closed_frac,
        "n_states": int(df["State"].nunique()),
        "states": sorted(df["State"].astype(str).tolist()),
        "source": str(aaa_path),
    }


# Baseline-anchored linear open-quantity helper

def linear_open_quantity(*, epsilon: float, p_bar: float, p_base: float = 1.0, q_base: float = 1.0) -> float:
    """Open-station quantity from linear inverse demand anchored at (q_base, p_base)."""
    if epsilon <= 0:
        raise ValueError(f"Need epsilon > 0, got {epsilon}.")
    if p_base <= 0 or q_base <= 0:
        raise ValueError(f"Need positive p_base and q_base, got p_base={p_base}, q_base={q_base}.")
    g_open = -p_base / (epsilon * q_base)
    return float(q_base + (p_bar - p_base) / g_open)


# Rationing allocation

@dataclass
class RationedData:
    """Rationing allocation: open stations take demand, closed get residual."""
    # Station fractions
    open_frac: float = FRACTION_OPEN
    closed_frac: float = FRACTION_CLOSED  # Combined closed + limiting

    # Controlled price
    p_bar: float = P_BAR

    # Elasticity for demand calculation (use mid-range)
    epsilon: float = BASE_EPSILON

    def __post_init__(self):
        # Demand at the ceiling price from linear, baseline-anchored calibration.
        self.demand_at_pbar = linear_open_quantity(
            epsilon=self.epsilon,
            p_bar=self.p_bar,
            p_base=P_ANCHOR,
            q_base=1.0,
        )

        # Open stations satisfy their demand
        self.q_open = self.demand_at_pbar

        # Closed/limiting stations get the residual supply
        total_open = self.open_frac * self.q_open
        total_closed = Q_SUPPLY - total_open

        if total_closed < 0:
            raise ValueError(f"Open demand exceeds supply! Open: {total_open:.3f}, Supply: {Q_SUPPLY:.3f}")

        self.q_closed = total_closed / self.closed_frac

    @property
    def total_quantity(self) -> float:
        return self.open_frac * self.q_open + self.closed_frac * self.q_closed

    @property
    def shortage(self) -> float:
        return 1.0 - self.total_quantity


# Shadow price calculations

def shadow_price_linear(q: float, g: float) -> float:
    """Linear inverse demand anchored at (q=1, p=1)."""
    return P_ANCHOR + g * (q - 1.0)


def shadow_price_with_choke(q: float, g_steep: float, g_flat: float, M: float) -> float:
    """Shadow price with piecewise demand respecting choke constraint."""
    p_zero_steep = shadow_price_linear(0, g_steep)

    if p_zero_steep <= M:
        return shadow_price_linear(q, g_steep)

    # Kink point
    q_kink = (M - P_ANCHOR + g_steep) / (g_steep - g_flat)
    p_kink = shadow_price_linear(q_kink, g_steep)

    if q >= q_kink:
        return shadow_price_linear(q, g_steep)
    else:
        return p_kink + g_flat * (q - q_kink)


def compute_shadow_price_bounds(data: RationedData, with_choke: bool = False, M: float = None):
    """Compute shadow price bounds for open and closed/limiting stations."""
    results = {}

    for name, q in [('closed', data.q_closed), ('open', data.q_open)]:
        if with_choke and M is not None:
            p_steep = shadow_price_with_choke(q, G_STEEP, G_FLAT, M)
            p_flat = shadow_price_linear(q, G_FLAT)
        else:
            p_steep = shadow_price_linear(q, G_STEEP)
            p_flat = shadow_price_linear(q, G_FLAT)

        p_lower = max(0, min(p_steep, p_flat))
        p_upper = max(p_steep, p_flat)

        results[name] = {
            'q': q,
            'p_lower': p_lower,
            'p_upper': p_upper,
            'p_mid': (p_lower + p_upper) / 2
        }

    return results


# Welfare calculations

def compute_welfare_bounds(data: RationedData) -> Dict:
    """Compute robust bounds on misallocation and Harberger DWL."""
    Q_bar = data.total_quantity
    shortage = data.shortage

    # Geometric constant V
    V = 0.0
    for frac, q_obs in [(data.closed_frac, data.q_closed),
                        (data.open_frac, data.q_open)]:
        V += frac * (Q_bar - q_obs)**2

    # Slope-invariant ratio
    R = V / shortage**2 if shortage > 0 else 0

    # Welfare bounds from slope bounds
    Phi_at_steep = 0.5 * abs(G_STEEP) * V
    Phi_at_flat = 0.5 * abs(G_FLAT) * V

    Psi_at_steep = 0.5 * abs(G_STEEP) * shortage**2
    Psi_at_flat = 0.5 * abs(G_FLAT) * shortage**2

    return {
        'V': V,
        'R': R,
        'shortage': shortage,
        'Q_bar': Q_bar,
        'Phi_lower': min(Phi_at_steep, Phi_at_flat),
        'Phi_upper': max(Phi_at_steep, Phi_at_flat),
        'Psi_lower': min(Psi_at_steep, Psi_at_flat),
        'Psi_upper': max(Psi_at_steep, Psi_at_flat),
        'Phi_at_steep': Phi_at_steep,
        'Phi_at_flat': Phi_at_flat,
        'Psi_at_steep': Psi_at_steep,
        'Psi_at_flat': Psi_at_flat,
    }


def compute_welfare_piecewise(data: RationedData, g_steep: float, g_flat: float, M: float) -> Dict:
    """Compute welfare using piecewise demand with a choke constraint."""
    Q_bar = data.total_quantity
    shortage = data.shortage

    # Kink point (where the choke binds)
    p_zero_steep = shadow_price_linear(0, g_steep)
    if p_zero_steep <= M:
        q_kink = 0.0
        choke_binds = False
    else:
        q_kink = (M - P_ANCHOR + g_steep) / (g_steep - g_flat)
        choke_binds = True

    p_kink = P_ANCHOR + g_steep * (q_kink - 1.0)

    def P(q):
        if q >= q_kink:
            return P_ANCHOR + g_steep * (q - 1.0)
        else:
            return p_kink + g_flat * (q - q_kink)

    # Shadow prices
    p_closed = P(data.q_closed)
    p_open = P(data.q_open)
    p_star = P(Q_bar)

    # Numerical integration
    def integrate_welfare(q_start, q_end, n_points=500):
        if abs(q_end - q_start) < 1e-9:
            return 0.0
        qs = np.linspace(min(q_start, q_end), max(q_start, q_end), n_points)
        ps = np.array([P(q) for q in qs])
        return np.trapz(np.abs(ps - p_star), qs)

    # Misallocation by station type
    Phi = 0.0
    contributions = {}

    for name, (frac, q_obs) in [('closed', (data.closed_frac, data.q_closed)),
                                 ('open', (data.open_frac, data.q_open))]:
        integral = integrate_welfare(q_obs, Q_bar)
        contribution = frac * integral
        Phi += contribution

        contributions[name] = {
            'frac': frac,
            'q_obs': q_obs,
            'delta_q': Q_bar - q_obs,
            'integral': integral,
            'contribution': contribution,
            'shadow_price': P(q_obs)
        }

    # Harberger triangle (steep slope at the shortage)
    Psi = 0.5 * abs(g_steep) * shortage**2

    return {
        'Psi': Psi,
        'Phi': Phi,
        'R': Phi / Psi if Psi > 0 else 0,
        'q_kink': q_kink,
        'p_kink': p_kink,
        'choke_binds': choke_binds,
        'p_star': p_star,
        'p_closed': p_closed,
        'p_open': p_open,
        'contributions': contributions,
    }


def compute_joint_phi_bounds_two_market(
    data: RationedData,
    *,
    eps_L: float = EPSILON_L,
    eps_U: float = EPSILON_H,
    p_base: float = P_ANCHOR,
    M: float = M_CHOKE,
    q_max: float = 1.0,
    n_grid: int = 8001,
) -> Dict:
    """
    Joint robust bounds for Φ with a binding adding-up constraint.

    This is the "two markets, two demands" version the paper's Lemma 1 targets:
      - Open and closed/limiting are separate markets with separate inverse demands.
      - The adding-up constraint is enforced in the computation via X(p).
      - Kinks can arise from adding-up even without a choke cap.

    Implementation strategy (minimal assumptions):
      1) Work in *aggregate* quantity units so adding-up is q_O + q_C = Qbar.
         (Aggregate q_i = fraction_i * per-station q_i.)
      2) Use baseline normalization to get market-specific slope bounds:
           g_i in [-p_base/(eps_L*q_i^base), -p_base/(eps_U*q_i^base)].
      3) Convert baseline anchors into *interval* anchors (q_i^obs, p0_i) by
         bounding p0_i = P_i(q_i^obs) using the slope bounds.
      4) For each endpoint combination of (p0_O, p0_C), solve the lemma program
         with the quadratic "peel-off" penalty terms.
    """
    # Aggregate observed quantities (these sum to Qbar by construction).
    qO_obs = data.open_frac * data.q_open
    qC_obs = data.closed_frac * data.q_closed
    Qbar = qO_obs + qC_obs

    # Aggregate baseline quantities (pre-crisis, normalized total = 1).
    qO_base = data.open_frac * 1.0
    qC_base = data.closed_frac * 1.0

    def slope_bounds(q_base: float) -> Tuple[float, float]:
        g_L = -p_base / (eps_L * q_base)
        g_U = -p_base / (eps_U * q_base)
        return g_L, g_U

    gO_L, gO_U = slope_bounds(qO_base)
    gC_L, gC_U = slope_bounds(qC_base)

    def p0_bounds(q_base: float, q_obs: float, g_L: float, g_U: float) -> Tuple[float, float]:
        # Linear extrapolation from the baseline anchor (q_base, p_base).
        p_steep = p_base + g_L * (q_obs - q_base)
        p_flat = p_base + g_U * (q_obs - q_base)
        lo = min(p_steep, p_flat)
        hi = max(p_steep, p_flat)
        lo = max(0.0, lo)
        if np.isfinite(M):
            # Enforce both a global choke cap and consistency at observed q:
            # P(0) <= M with P'(q) <= g_U implies p0 = P(q_obs) <= M + g_U*q_obs.
            hi = min(hi, M, M + g_U * q_obs)
            lo = min(lo, M)
        if lo > hi:
            lo = hi
        return lo, hi

    pO_lo, pO_hi = p0_bounds(qO_base, qO_obs, gO_L, gO_U)
    pC_lo, pC_hi = p0_bounds(qC_base, qC_obs, gC_L, gC_U)

    combos = [
        ("low/low", pO_lo, pC_lo),
        ("low/high", pO_lo, pC_hi),
        ("high/low", pO_hi, pC_lo),
        ("high/high", pO_hi, pC_hi),
    ]

    upper_solutions = []
    lower_solutions = []

    for label, p0_O, p0_C in combos:
        mO = MarketSpec(
            name="open",
            q0=qO_obs,
            p0=p0_O,
            g_L=gO_L,
            g_U=gO_U,
            q_max=q_max,
            M=M,
        )
        mC = MarketSpec(
            name="closed",
            q0=qC_obs,
            p0=p0_C,
            g_L=gC_L,
            g_U=gC_U,
            q_max=q_max,
            M=M,
        )

        p_max = max(10.0, M) if np.isfinite(M) else 10.0
        upper = solve_joint_bound_two_market(mO, mC, Qbar=Qbar, mode="upper", p_min=0.0, p_max=p_max, n_grid=n_grid)
        upper["p0_combo"] = label
        upper["p0_open"] = p0_O
        upper["p0_closed"] = p0_C
        upper_solutions.append(upper)

        lower = solve_joint_bound_two_market(mO, mC, Qbar=Qbar, mode="lower", p_min=0.0, p_max=p_max, n_grid=n_grid)
        lower["p0_combo"] = label
        lower["p0_open"] = p0_O
        lower["p0_closed"] = p0_C
        lower_solutions.append(lower)

    upper_best = max(upper_solutions, key=lambda d: d["Phi"])
    lower_best = min(lower_solutions, key=lambda d: d["Phi"])

    return {
        "Qbar": Qbar,
        "q_open_obs": qO_obs,
        "q_closed_obs": qC_obs,
        "q_open_base": qO_base,
        "q_closed_base": qC_base,
        "slope_bounds": {
            "open": (gO_L, gO_U),
            "closed": (gC_L, gC_U),
        },
        "p0_bounds": {
            "open": (pO_lo, pO_hi),
            "closed": (pC_lo, pC_hi),
        },
        "upper_best": upper_best,
        "lower_best": lower_best,
        "upper_all": upper_solutions,
        "lower_all": lower_solutions,
    }


# Sensitivity analysis

def sensitivity_to_pbar(
    p_bars=None,
    *,
    open_frac: float = FRACTION_OPEN,
    closed_frac: float = FRACTION_CLOSED,
    epsilon: float = BASE_EPSILON,
):
    """How results vary with controlled price p_bar."""
    if p_bars is None:
        p_bars = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []
    for p_bar in p_bars:
        try:
            data = RationedData(
                p_bar=p_bar,
                open_frac=open_frac,
                closed_frac=closed_frac,
                epsilon=epsilon,
            )
            bounds = compute_welfare_bounds(data)
            pw = compute_welfare_piecewise(data, G_STEEP, G_FLAT, M_CHOKE)

            results.append({
                'p_bar': p_bar,
                'q_open': data.q_open,
                'q_closed': data.q_closed,
                'R_linear': bounds['R'],
                'R_piecewise': pw['R'],
                'Phi_piecewise': pw['Phi'],
                'Psi_piecewise': pw['Psi'],
            })
        except ValueError as e:
            print(f"p_bar={p_bar}: {e}")

    return results


def sensitivity_to_choke(M_values=None, *, data: Optional[RationedData] = None):
    """How results vary with choke price M."""
    if M_values is None:
        M_values = np.linspace(3, 8, 50)

    if data is None:
        data = RationedData()  # Default p_bar = 0.8

    results = []
    for M in M_values:
        pw = compute_welfare_piecewise(data, G_STEEP, G_FLAT, M)
        results.append({
            'M': M,
            'Phi': pw['Phi'],
            'Psi': pw['Psi'],
            'R': pw['R'],
            'p_closed': pw['p_closed'],
            'choke_binds': pw['choke_binds'],
        })

    return results


# Figures

def create_figures(output_dir: str = None, data: Optional[RationedData] = None):
    """Create figures for the rationing robust bounds section."""
    if output_dir is None:
        output_dir = THIS_DIR.parent / "output"
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if data is None:
        data = RationedData()
    bounds = compute_welfare_bounds(data)
    pw = compute_welfare_piecewise(data, G_STEEP, G_FLAT, M_CHOKE)

    # Figure 5: demand curves with rationing allocation
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    q_range = np.linspace(0.001, 1.3, 200)

    p_steep = [shadow_price_linear(q, G_STEEP) for q in q_range]
    p_flat = [shadow_price_linear(q, G_FLAT) for q in q_range]
    p_piecewise = [shadow_price_with_choke(q, G_STEEP, G_FLAT, M_CHOKE) for q in q_range]

    ax1.plot(q_range, p_steep, 'b--', linewidth=2, alpha=0.5,
             label=f'Steep ($\\varepsilon$={EPSILON_L})')
    ax1.plot(q_range, p_flat, 'g--', linewidth=2, alpha=0.5,
             label=f'Flat ($\\varepsilon$={EPSILON_H})')
    ax1.plot(q_range, p_piecewise, 'r-', linewidth=2.5,
             label='Piecewise extremal')

    ax1.axhline(y=M_CHOKE, color='gray', linestyle=':', alpha=0.7, label=f'Choke M={M_CHOKE}')

    # Mark allocations
    ax1.axvline(x=data.q_closed, color='coral', linestyle='--', alpha=0.6)
    ax1.axvline(x=data.q_open, color='forestgreen', linestyle='--', alpha=0.6)
    ax1.axvline(x=bounds['Q_bar'], color='black', linestyle=':', alpha=0.6)

    ax1.text(data.q_closed + 0.02, 0.5, f'Closed\nq={data.q_closed:.2f}', fontsize=9, color='coral')
    ax1.text(data.q_open + 0.02, 0.5, f'Open\nq={data.q_open:.2f}', fontsize=9, color='forestgreen')

    ax1.plot(1.0, 1.0, 'ko', markersize=8)
    ax1.annotate('Anchor', (1.0, 1.0), xytext=(1.05, 1.3), fontsize=9)

    ax1.set_xlabel('Quantity', fontsize=12)
    ax1.set_ylabel('Shadow Price P(q)', fontsize=12)
    ax1.set_title('Demand Curves with Rationed Allocation', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 1.3)
    ax1.set_ylim(0, 7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: shadow price bounds
    ax2 = axes[1]
    bounds_with_choke = compute_shadow_price_bounds(data, with_choke=True, M=M_CHOKE)
    bounds_no_choke = compute_shadow_price_bounds(data, with_choke=False)

    types = ['Open', 'Closed/Limiting']
    y_pos = np.arange(len(types))
    height = 0.35

    for i, name in enumerate(['open', 'closed']):
        nc = bounds_no_choke[name]
        wc = bounds_with_choke[name]

        ax2.barh(i + height/2, nc['p_upper'] - nc['p_lower'], left=nc['p_lower'],
                 height=height, color='steelblue', alpha=0.7, label='No choke' if i == 0 else '')
        ax2.barh(i - height/2, wc['p_upper'] - wc['p_lower'], left=wc['p_lower'],
                 height=height, color='coral', alpha=0.7, label=f'With choke M={M_CHOKE}' if i == 0 else '')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(types, fontsize=11)
    ax2.set_xlabel('Shadow Price', fontsize=12)
    ax2.set_title('Shadow Price Bounds (Rationed)', fontsize=11)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig1.savefig(output_dir / "figure_robust_demand_curves.pdf", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figure_robust_demand_curves.pdf'}")
    plt.close(fig1)

    # Figure 6: misallocation vs choke price
    fig2, ax = plt.subplots(figsize=(10, 6))

    choke_results = sensitivity_to_choke(data=data)
    M_vals = [r['M'] for r in choke_results]
    Phi_vals = [r['Phi'] for r in choke_results]
    R_vals = [r['R'] for r in choke_results]

    ax.plot(M_vals, Phi_vals, 'r-', linewidth=2.5, label='$\\mathcal{L}_{Mis}$ (misallocation)')

    # Mark where choke stops binding
    ax.axhline(y=bounds['Phi_at_steep'], color='blue', linestyle='--', alpha=0.7,
               label=f'Steep bound: $\\mathcal{{L}}_{{Mis}}$={bounds["Phi_at_steep"]:.3f}')
    ax.axhline(y=bounds['Phi_at_flat'], color='green', linestyle='--', alpha=0.7,
               label=f'Flat bound: $\\mathcal{{L}}_{{Mis}}$={bounds["Phi_at_flat"]:.3f}')

    # Mark baseline
    ax.axvline(x=M_CHOKE, color='orange', linestyle='-', linewidth=2)
    ax.plot(M_CHOKE, pw['Phi'], 'o', color='orange', markersize=12)
    ax.annotate(f'M={M_CHOKE}\n$\\mathcal{{L}}_{{Mis}}$={pw["Phi"]:.3f}', (M_CHOKE, pw['Phi']),
                xytext=(M_CHOKE+0.5, pw['Phi']+0.01), fontsize=10, color='orange')

    ax.set_xlabel('Choke Price M', fontsize=12)
    ax.set_ylabel('Misallocation $\\mathcal{L}_{Mis}$', fontsize=12)
    ax.set_title('Misallocation vs Choke Price (Rationed Allocation)', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(3, 8)
    ax.grid(True, alpha=0.3)

    fig2.savefig(output_dir / "figure_robust_phi_vs_choke.pdf", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figure_robust_phi_vs_choke.pdf'}")
    plt.close(fig2)

    # Figure 7: station-level robust-loss profiles under no choke.
    joint = compute_joint_phi_bounds_two_market(data, n_grid=4001, M=float("inf"))
    ub = joint["upper_best"]
    lb = joint["lower_best"]
    pO_lo, pO_hi = joint["p0_bounds"]["open"]
    pC_lo, pC_hi = joint["p0_bounds"]["closed"]
    psi_ref = pw["Psi"]

    fig4, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: admissible shadow-price intervals at observed allocations.
    axA = axes[0]
    y_labels = ["Open", "Closed/Limiting"]
    y_pos = np.arange(len(y_labels))
    lo_vals = [pO_lo, pC_lo]
    hi_vals = [pO_hi, pC_hi]
    mids = [0.5 * (pO_lo + pO_hi), 0.5 * (pC_lo + pC_hi)]
    for i in range(len(y_labels)):
        axA.hlines(y=i, xmin=lo_vals[i], xmax=hi_vals[i], color="steelblue", linewidth=5, alpha=0.8)
        axA.plot(mids[i], i, "o", color="black", markersize=6)
    axA.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="Baseline p=1")
    axA.set_yticks(y_pos)
    axA.set_yticklabels(y_labels, fontsize=10)
    axA.set_xlabel("Admissible shadow price at observed allocation", fontsize=11)
    axA.set_title("A. Shadow Price Ranges at Observed Allocation", fontsize=11)
    axA.grid(True, alpha=0.3, axis="x")
    axA.legend(loc="lower right", fontsize=9)

    # Panel B: higher-loss profile across admissible scenarios.
    axB = axes[1]
    for k, sol in enumerate(joint["upper_all"]):
        axB.plot(sol["p_grid"], sol["objective_grid"] * 100.0, linewidth=1.5, alpha=0.9, label=f"Scenario {k+1}")
    axB.plot(ub["p_star"], ub["Phi"] * 100.0, "o", color="black", markersize=7)
    axB.annotate(
        f"Highest loss\n$\\mathcal{{L}}_{{Mis}}$={ub['Phi']*100:.2f}%",
        xy=(ub["p_star"], ub["Phi"] * 100.0),
        xytext=(ub["p_star"] + 0.25, ub["Phi"] * 100.0 + 0.7),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    axB.axhline(y=psi_ref * 100.0, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="Harberger benchmark")
    axB.set_xlabel("Common shadow price $p$", fontsize=11)
    axB.set_ylabel("Estimated misallocation loss (% of baseline spending)", fontsize=11)
    axB.set_title("B. Higher-Loss Profiles Across Common Shadow Prices", fontsize=11)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="upper right", fontsize=8)

    # Panel C: lower-loss profile across admissible scenarios.
    axC = axes[2]
    for k, sol in enumerate(joint["lower_all"]):
        axC.plot(sol["p_grid"], sol["objective_grid"] * 100.0, linewidth=1.5, alpha=0.9, label=f"Scenario {k+1}")
    axC.plot(lb["p_star"], lb["Phi"] * 100.0, "o", color="black", markersize=7)
    axC.annotate(
        f"Lowest loss\n$\\mathcal{{L}}_{{Mis}}$={lb['Phi']*100:.2f}%",
        xy=(lb["p_star"], lb["Phi"] * 100.0),
        xytext=(lb["p_star"] + 0.25, lb["Phi"] * 100.0 + 0.7),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    axC.axhline(y=psi_ref * 100.0, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="Harberger benchmark")
    axC.set_xlabel("Common shadow price $p$", fontsize=11)
    axC.set_ylabel("Estimated misallocation loss (% of baseline spending)", fontsize=11)
    axC.set_title("C. Lower-Loss Profiles Across Common Shadow Prices", fontsize=11)
    axC.grid(True, alpha=0.3)
    axC.legend(loc="upper right", fontsize=8)

    fig4.suptitle(
        (
            "Station-Level Misallocation Loss Range Without Choke "
            f"($\\mathcal{{L}}_{{Mis}} \\in [{lb['Phi']*100:.2f}\\%,\\ {ub['Phi']*100:.2f}\\%]$ of baseline spending)"
        ),
        fontsize=12,
    )
    fig4.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig4.savefig(output_dir / "figure_station_joint_object.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'figure_station_joint_object.pdf'}")
    plt.close(fig4)

    # Figure 8: representative extremal demand curves that attain joint bounds.
    def _branch_fns(mkt: MarketSpec, mode: str, p_star: float):
        forward = p_star >= mkt.p0
        if mode == "upper":
            base = mkt.ell if forward else mkt.u
        else:
            base = mkt.u if forward else mkt.ell
        return base, forward

    def _representative_q_of_p(mkt: MarketSpec, mode: str, p_star: float, x_star: float, p_grid: np.ndarray) -> np.ndarray:
        """
        Build one representative extremal q(p) path that:
          - passes through the anchor (q0, p0),
          - hits the optimizer point (x*, p*),
          - traces a full curve by extending with the mode-implied envelope
            branch away from p*, with a short connector near p* for peel-off.
        """
        base_fn, forward = _branch_fns(mkt, mode, p_star)
        q_base = np.array([base_fn(float(p)) for p in p_grid], dtype=float)

        c_star = float(base_fn(p_star))
        if mkt.kappa <= 1e-12:
            delta_p = 0.0
        else:
            delta_p = abs(float(x_star) - c_star) / mkt.kappa
        span = abs(p_star - mkt.p0)
        delta_p = min(delta_p, span)

        q = q_base.copy()
        connector_tol = 1e-10

        if forward:
            if delta_p <= connector_tol:
                pass
            else:
                p_switch = max(mkt.p0, p_star - delta_p)
                q_switch = float(base_fn(p_switch))
                denom = p_star - p_switch
                slope = (float(x_star) - q_switch) / denom if abs(denom) > 1e-12 else 0.0

                m_conn = (p_grid > p_switch) & (p_grid <= p_star)
                q[m_conn] = q_switch + slope * (p_grid[m_conn] - p_switch)

                # Continue to the right of p* with parallel slope so q(p*)=x*.
                d = float(x_star) - c_star
                m_right = p_grid > p_star
                q[m_right] = q_base[m_right] + d
        else:
            if delta_p <= connector_tol:
                pass
            else:
                p_switch = min(mkt.p0, p_star + delta_p)
                q_switch = float(base_fn(p_switch))
                denom = p_star - p_switch
                slope = (float(x_star) - q_switch) / denom if abs(denom) > 1e-12 else 0.0

                m_conn = (p_grid >= p_star) & (p_grid < p_switch)
                q[m_conn] = q_switch + slope * (p_grid[m_conn] - p_switch)

                # Continue to the left of p* with parallel slope so q(p*)=x*.
                d = float(x_star) - c_star
                m_left = p_grid < p_star
                q[m_left] = q_base[m_left] + d

        return q

    from matplotlib.lines import Line2D

    def _plot_station_joint_extremals(
        joint_sol: Dict,
        *,
        market_M: float,
        title_suffix: str,
        output_stem: str,
    ) -> None:
        gO_L, gO_U = joint_sol["slope_bounds"]["open"]
        gC_L, gC_U = joint_sol["slope_bounds"]["closed"]
        ub_local = joint_sol["upper_best"]
        lb_local = joint_sol["lower_best"]

        def _make_markets(sol: Dict) -> Tuple[MarketSpec, MarketSpec]:
            m_open = MarketSpec(
                name="open",
                q0=float(joint_sol["q_open_obs"]),
                p0=float(sol["p0_open"]),
                g_L=float(gO_L),
                g_U=float(gO_U),
                q_max=1.0,
                M=float(market_M),
            )
            m_closed = MarketSpec(
                name="closed",
                q0=float(joint_sol["q_closed_obs"]),
                p0=float(sol["p0_closed"]),
                g_L=float(gC_L),
                g_U=float(gC_U),
                q_max=1.0,
                M=float(market_M),
            )
            return m_open, m_closed

        # Stack panels vertically for better readability in the paper layout.
        fig5, (axU, axL) = plt.subplots(2, 1, figsize=(10.5, 11.0), sharex=True, sharey=True)

        panel_specs = [
            ("upper", ub_local, axU, "A. Higher-Loss Case"),
            ("lower", lb_local, axL, "B. Lower-Loss Case"),
        ]
        colors = {"open": "forestgreen", "closed": "coral"}
        share = {"open": float(data.open_frac), "closed": float(data.closed_frac)}

        def _to_per_station(market_name: str, q_agg: np.ndarray) -> np.ndarray:
            return q_agg / share[market_name]

        # Shared p-range so both panels trace full curves on the same domain.
        m_open_ub, m_closed_ub = _make_markets(ub_local)
        m_open_lb, m_closed_lb = _make_markets(lb_local)
        markets_for_range = [m_open_ub, m_closed_ub, m_open_lb, m_closed_lb]

        def _p_at_q_zero(mkt: MarketSpec) -> float:
            # Candidate price at q=0 under each admissible slope through (q0, p0).
            return float(
                max(
                    mkt.p0 + mkt.g_L * (0.0 - mkt.q0),
                    mkt.p0 + mkt.g_U * (0.0 - mkt.q0),
                )
            )

        p_hi_plot = max([1.0, *(_p_at_q_zero(m) for m in markets_for_range)]) + 0.2
        p_grid_local = np.linspace(0.0, p_hi_plot, 3000)

        def _linear_completion_q_of_p(
            p_grid: np.ndarray,
            *,
            q0: float,
            p0: float,
            x_star: float,
            p_star: float,
        ) -> np.ndarray:
            # Straight-line completion through anchor and optimizer.
            if abs(p_star - p0) <= 1e-12:
                return np.full_like(p_grid, q0, dtype=float)
            slope = (x_star - q0) / (p_star - p0)
            return q0 + slope * (p_grid - p0)

        all_q_plot_values = []

        for mode_name, sol, axp, title in panel_specs:
            m_open, m_closed = _make_markets(sol)
            x_open, x_closed = [float(v) for v in sol["x_star"]]
            p_star = float(sol["p_star"])

            for mkt, x_star in [
                (m_open, x_open),
                (m_closed, x_closed),
            ]:
                q_rep = _representative_q_of_p(mkt, mode_name, p_star, x_star, p_grid_local)
                color = colors[mkt.name]
                q_rep_plot = q_rep.copy()
                q_rep_plot[(q_rep_plot < 0.0) | (q_rep_plot > mkt.q_max)] = np.nan
                q_rep_station = _to_per_station(mkt.name, q_rep_plot)

                # Solid only on the identified segment [min(p0,p*), max(p0,p*)].
                # Outside that segment, a completion exists but is not identified.
                p_lo = min(mkt.p0, p_star)
                p_hi = max(mkt.p0, p_star)
                ok = np.isfinite(q_rep_plot)
                ident = ok & (p_grid_local >= p_lo) & (p_grid_local <= p_hi)
                non_ident = ok & ~ident
                all_q_plot_values.extend(q_rep_station[np.isfinite(q_rep_station)].tolist())

                # For no-choke visualizations, force the dashed completion to be
                # straight to avoid implying extra identified curvature.
                if not np.isfinite(market_M):
                    q_dashed = _linear_completion_q_of_p(
                        p_grid_local,
                        q0=float(mkt.q0),
                        p0=float(mkt.p0),
                        x_star=float(x_star),
                        p_star=float(p_star),
                    )
                    q_dashed[(q_dashed < 0.0) | (q_dashed > mkt.q_max)] = np.nan
                    q_dashed_station = _to_per_station(mkt.name, q_dashed)
                    non_ident = np.isfinite(q_dashed) & ~ident
                    all_q_plot_values.extend(q_dashed_station[np.isfinite(q_dashed_station)].tolist())
                    axp.plot(
                        q_dashed_station[non_ident],
                        p_grid_local[non_ident],
                        color=color,
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.75,
                    )
                else:
                    q_dashed = _linear_completion_q_of_p(
                        p_grid_local,
                        q0=float(mkt.q0),
                        p0=float(mkt.p0),
                        x_star=float(x_star),
                        p_star=float(p_star),
                    )
                    q_dashed[(q_dashed < 0.0) | (q_dashed > mkt.q_max)] = np.nan
                    q_dashed_station = _to_per_station(mkt.name, q_dashed)
                    all_q_plot_values.extend(q_dashed_station[np.isfinite(q_dashed_station)].tolist())

                    # With choke: keep the high-price side completion from the
                    # envelope (can kink at the choke), but keep the low-price
                    # side completion straight to avoid artificial right-side kinks.
                    low_price_side = np.isfinite(q_dashed_station) & (p_grid_local < p_lo)
                    high_price_side = np.isfinite(q_rep_station) & (p_grid_local > p_hi)

                    axp.plot(
                        q_dashed_station[low_price_side],
                        p_grid_local[low_price_side],
                        color=color,
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.75,
                    )
                    if mkt.name == "closed" and p_hi < market_M - 1e-12:
                        # Visual cue only: connect dashed closed-side completion
                        # to the choke point (q=0, p=M).
                        if abs(p_hi - mkt.p0) <= 1e-9:
                            q_hi_station = mkt.q0 / share[mkt.name]
                        else:
                            q_hi_station = x_star / share[mkt.name]
                        all_q_plot_values.extend([q_hi_station, 0.0])
                        axp.plot(
                            [q_hi_station, 0.0],
                            [p_hi, market_M],
                            color=color,
                            linestyle="--",
                            linewidth=1.8,
                            alpha=0.75,
                        )
                    else:
                        axp.plot(
                            q_rep_station[high_price_side],
                            p_grid_local[high_price_side],
                            color=color,
                            linestyle="--",
                            linewidth=1.8,
                            alpha=0.75,
                        )
                axp.plot(
                    q_rep_station[ident],
                    p_grid_local[ident],
                    color=color,
                    linewidth=2.8,
                )
                axp.plot(
                    mkt.q0 / share[mkt.name],
                    mkt.p0,
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                )
                axp.plot(
                    x_star / share[mkt.name],
                    p_star,
                    marker="*",
                    color=color,
                    markersize=10,
                )

            axp.axhline(y=p_star, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
            axp.set_title(
                f"{title}\n"
                f"Estimated $\\mathcal{{L}}_{{Mis}}$ = {sol['Phi']*100:.2f}% of baseline spending, common shadow price = {p_star:.2f}",
                fontsize=11,
            )
            axp.set_xlabel("Per-station quantity $q_i$", fontsize=11)
            axp.grid(True, alpha=0.22)

        axU.set_ylabel("Shadow price $p$", fontsize=11)
        q_hi_plot = max(all_q_plot_values) * 1.05 if all_q_plot_values else 1.4
        q_hi_plot = max(1.4, min(q_hi_plot, 3.0))
        axU.set_xlim(0.0, q_hi_plot)
        axL.set_xlim(0.0, q_hi_plot)
        axU.set_ylim(0.0, p_hi_plot)

        legend_handles = [
            Line2D([0], [0], color="forestgreen", lw=2.8, label="Open market (identified segment)"),
            Line2D([0], [0], color="forestgreen", lw=1.8, ls="--", alpha=0.75, label="Open market (admissible extension)"),
            Line2D([0], [0], color="coral", lw=2.8, label="Closed/limiting market (identified segment)"),
            Line2D([0], [0], color="coral", lw=1.8, ls="--", alpha=0.75, label="Closed/limiting market (admissible extension)"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="white", lw=0, label="Observed allocation point"),
            Line2D([0], [0], marker="*", color="black", lw=0, markersize=10, label="Balanced-allocation point"),
        ]
        fig5.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        fig5.suptitle(
            "Station-Level Demand Curves Consistent with Observed Rationing "
            f"{title_suffix} (displayed in per-station units)",
            fontsize=13,
        )
        fig5.tight_layout(rect=[0, 0.08, 1, 0.95])

        fig_pdf = output_dir / f"{output_stem}.pdf"
        fig_png = output_dir / f"{output_stem}.png"
        fig5.savefig(fig_pdf, dpi=150, bbox_inches="tight")
        fig5.savefig(fig_png, dpi=200, bbox_inches="tight")
        print(f"Saved: {fig_pdf}")
        print(f"Saved: {fig_png}")
        plt.close(fig5)

    _plot_station_joint_extremals(
        joint,
        market_M=float("inf"),
        title_suffix="(No Choke)",
        output_stem="figure_station_joint_extremal_curves",
    )

    joint_with_choke = compute_joint_phi_bounds_two_market(data, n_grid=4001, M=M_CHOKE)
    _plot_station_joint_extremals(
        joint_with_choke,
        market_M=M_CHOKE,
        title_suffix=f"(With Choke M={M_CHOKE:g})",
        output_stem="figure_station_joint_extremal_curves_with_choke",
    )

    # Figure 8b: station-level shadow-price intervals (no choke vs with choke),
    # styled to match the right panel of figure_robust_demand_curves.
    fig6, ax6 = plt.subplots(figsize=(9, 4.8))
    types = ["Open", "Closed/Limiting"]
    y_pos = np.arange(len(types))
    height = 0.34

    for i, name in enumerate(["open", "closed"]):
        nc_lo, nc_hi = joint["p0_bounds"][name]
        wc_lo, wc_hi = joint_with_choke["p0_bounds"][name]

        ax6.barh(
            i + height / 2,
            nc_hi - nc_lo,
            left=nc_lo,
            height=height,
            color="steelblue",
            alpha=0.75,
            label="No choke" if i == 0 else "",
        )
        ax6.barh(
            i - height / 2,
            wc_hi - wc_lo,
            left=wc_lo,
            height=height,
            color="coral",
            alpha=0.75,
            label=f"With choke M={M_CHOKE:g}" if i == 0 else "",
        )

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(types, fontsize=11)
    ax6.set_xlabel("Shadow price at observed allocation $p_{0,i}$", fontsize=12)
    ax6.set_title("Station-Level Shadow Price Ranges (No Choke vs With Choke)", fontsize=12)
    ax6.axvline(x=1.0, color="gray", linestyle="--", alpha=0.6, label="Baseline p=1")
    ax6.grid(True, alpha=0.3, axis="x")
    ax6.legend(loc="upper right", fontsize=9)

    fig6.tight_layout()
    fig6_pdf = output_dir / "figure_station_joint_shadow_bounds.pdf"
    fig6_png = output_dir / "figure_station_joint_shadow_bounds.png"
    fig6.savefig(fig6_pdf, dpi=150, bbox_inches="tight")
    fig6.savefig(fig6_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig6_pdf}")
    print(f"Saved: {fig6_png}")
    plt.close(fig6)

    # Extra: R vs p_bar sensitivity (not in paper figures)
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    pbar_results = sensitivity_to_pbar(
        np.linspace(0.5, 1.0, 30),
        open_frac=data.open_frac,
        closed_frac=data.closed_frac,
        epsilon=data.epsilon,
    )

    p_bars = [r['p_bar'] for r in pbar_results]
    q_opens = [r['q_open'] for r in pbar_results]
    q_closeds = [r['q_closed'] for r in pbar_results]
    Rs = [r['R_piecewise'] for r in pbar_results]

    ax1 = axes[0]
    ax1.plot(p_bars, q_opens, 'g-', linewidth=2, label='Open stations')
    ax1.plot(p_bars, q_closeds, 'r-', linewidth=2, label='Closed/Limiting')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.axvline(x=0.8, color='black', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Controlled Price $\\bar{p}$', fontsize=12)
    ax1.set_ylabel('Quantity per Station', fontsize=12)
    ax1.set_title('Allocation vs Controlled Price', fontsize=11)
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(p_bars, Rs, 'b-', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='R=1')
    ax2.axvline(x=0.8, color='black', linestyle=':', alpha=0.7, label='Baseline')
    ax2.set_xlabel('Controlled Price $\\bar{p}$', fontsize=12)
    ax2.set_ylabel('Misallocation Ratio $\\mathcal{R}$', fontsize=12)
    ax2.set_title('$\\mathcal{R}$ vs Price Control Depth', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(output_dir / "figure_robust_pbar_sensitivity.pdf", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figure_robust_pbar_sensitivity.pdf'}")
    plt.close(fig3)

    return data, bounds, pw


# Main

def run_analysis(data: Optional[RationedData] = None):
    """Run full analysis and print results."""

    print("=" * 70)
    print("ROBUST BOUNDS WITH RATIONING ALLOCATION")
    print("=" * 70)

    if data is None:
        data = RationedData()

    print(f"\nRationing allocation (p_bar = {data.p_bar}):")
    print(f"  Open ({data.open_frac*100:.1f}%): q = {data.q_open:.3f}")
    print(f"  Closed/Limiting ({data.closed_frac*100:.1f}%): q = {data.q_closed:.3f}")
    print(f"  Total: {data.total_quantity:.3f} (shortage = {data.shortage:.1%})")

    print(f"\nSlope bounds from epsilon in [{EPSILON_L}, {EPSILON_H}]:")
    print(f"  g in [{G_STEEP:.2f}, {G_FLAT:.2f}]")

    # Shadow price bounds
    print("\n" + "-" * 50)
    print("SHADOW PRICE BOUNDS")
    print("-" * 50)

    bounds_nc = compute_shadow_price_bounds(data, with_choke=False)
    bounds_wc = compute_shadow_price_bounds(data, with_choke=True, M=M_CHOKE)

    print(f"\n{'Station':<15} {'q':<8} {'No Choke':<20} {f'With Choke M={M_CHOKE:g}':<20}")
    print("-" * 65)
    for name in ['open', 'closed']:
        nc = bounds_nc[name]
        wc = bounds_wc[name]
        print(f"{name.capitalize():<15} {nc['q']:<8.3f} [{nc['p_lower']:.2f}, {nc['p_upper']:.2f}]"
              f"        [{wc['p_lower']:.2f}, {wc['p_upper']:.2f}]")

    # Welfare bounds
    print("\n" + "-" * 50)
    print("WELFARE BOUNDS")
    print("-" * 50)

    bounds = compute_welfare_bounds(data)

    print(f"\nGeometric constant V = {bounds['V']:.6f}")
    print(f"Shortage s = {bounds['shortage']:.6f}")
    print(f"Linear ratio = V/s^2 = {bounds['R']:.2f}  (= {bounds['R']*100:.1f}% of L_Harb)")

    print(f"\nMisallocation bounds (linear):")
    print(f"  L_Mis in [{bounds['Phi_lower']*100:.2f}%, {bounds['Phi_upper']*100:.2f}%]")

    print(f"\nHarberger bounds (linear):")
    print(f"  L_Harb in [{bounds['Psi_lower']*100:.3f}%, {bounds['Psi_upper']*100:.3f}%]")

    # Piecewise calculation
    print("\n" + "-" * 50)
    print(f"PIECEWISE WELFARE (with choke M={M_CHOKE:g})")
    print("-" * 50)

    pw = compute_welfare_piecewise(data, G_STEEP, G_FLAT, M_CHOKE)

    print(f"\nKink at q = {pw['q_kink']:.3f}")
    print(f"Choke binds: {pw['choke_binds']}")
    print(f"\nShadow prices:")
    print(f"  Open: {pw['p_open']:.2f}")
    print(f"  Closed/Limiting: {pw['p_closed']:.2f}")
    print(f"  Efficient p*: {pw['p_star']:.2f}")

    print(f"\nWelfare:")
    print(f"  L_Mis = {pw['Phi']*100:.2f}%")
    print(f"  L_Harb = {pw['Psi']*100:.3f}%")
    print(f"  L_Mis/L_Harb: {pw['R']*100:.1f}%  (ratio = {pw['R']:.2f})")

    # Joint robust bounds with a binding adding-up constraint
    print("\n" + "-" * 50)
    print("JOINT ROBUST L_Mis (ADDING-UP BINDS)")
    print("-" * 50)

    joint = compute_joint_phi_bounds_two_market(data, n_grid=4001)

    pO_lo, pO_hi = joint["p0_bounds"]["open"]
    pC_lo, pC_hi = joint["p0_bounds"]["closed"]

    print(
        f"\nAggregate quantities (q_O + q_C = Q̄):\n"
        f"  q_O^obs = {joint['q_open_obs']:.3f}, q_C^obs = {joint['q_closed_obs']:.3f}, Q̄ = {joint['Qbar']:.3f}\n"
        f"  Baseline anchors: q_O^base = {joint['q_open_base']:.3f}, q_C^base = {joint['q_closed_base']:.3f}, p^base = {P_ANCHOR:.1f}"
    )
    print(
        f"\nImplied anchor-price bounds at observed allocation (from slope bounds + baseline):\n"
        f"  p0_O ∈ [{pO_lo:.2f}, {pO_hi:.2f}]\n"
        f"  p0_C ∈ [{pC_lo:.2f}, {pC_hi:.2f}]"
    )

    ub = joint["upper_best"]
    lb = joint["lower_best"]
    Psi_ref = pw["Psi"]
    ub_pct_harberger = (ub["Phi"] / Psi_ref * 100.0) if Psi_ref > 0 else float("nan")
    lb_pct_harberger = (lb["Phi"] / Psi_ref * 100.0) if Psi_ref > 0 else float("nan")

    print("\nLemma 1 joint bounds (with peel-off penalty terms):")
    print(
        f"  Upper L_Mis = {ub['Phi']*100:.2f}%  (= {ub_pct_harberger:.1f}% of L_Harb; p*={ub['p_star']:.2f}, x*={ub['x_star']}, p0 combo={ub['p0_combo']})"
    )
    print(
        f"  Lower L_Mis = {lb['Phi']*100:.2f}%  (= {lb_pct_harberger:.1f}% of L_Harb; p*={lb['p_star']:.2f}, x*={lb['x_star']}, p0 combo={lb['p0_combo']})"
    )

    # Sensitivity to p_bar
    print("\n" + "-" * 50)
    print("SENSITIVITY TO p_bar")
    print("-" * 50)

    print(f"\n{'p_bar':<8} {'q_open':<10} {'q_closed':<12} {'L_Mis/L_Harb (%)':<16}")
    print("-" * 40)
    for r in sensitivity_to_pbar(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        open_frac=data.open_frac,
        closed_frac=data.closed_frac,
        epsilon=data.epsilon,
    ):
        print(f"{r['p_bar']:<8.2f} {r['q_open']:<10.3f} {r['q_closed']:<12.3f} {r['R_piecewise']*100:<10.1f}")

    return data, bounds, pw


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Station-level robust bounds with optional state-subset filtering "
            "for the station-share calibration."
        )
    )
    parser.add_argument(
        "--state-subset",
        type=str,
        choices=["all", "sales36"],
        default="all",
        help=(
            "State sample used to aggregate open/closed station shares: "
            "`all` (default AAA sample) or `sales36` (states with observed sales volume)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: output/ or scenario subfolder).",
    )
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="Write a JSON metadata file describing the calibration/sample used.",
    )
    args = parser.parse_args()

    include_states, subset_meta = resolve_state_subset(args.state_subset, repo_root=REPO_ROOT)
    frac_meta = derive_station_fractions_from_aaa(include_states=include_states)

    data = RationedData(
        open_frac=float(frac_meta["open_frac"]),
        closed_frac=float(frac_meta["closed_frac"]),
    )

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        base_out = THIS_DIR.parent / "output"
        out_dir = base_out if args.state_subset == "all" else (base_out / f"scenario_{args.state_subset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"State subset for station-share aggregation: {subset_meta['subset_id']} "
        f"({subset_meta['subset_label']})"
    )
    print(f"States used in station calibration: {frac_meta['n_states']}")
    print(
        "Derived station shares from AAA counts: "
        f"open={data.open_frac:.4f}, closed/limiting={data.closed_frac:.4f}"
    )
    if args.state_subset != "all":
        print(f"Scenario output directory: {out_dir}")

    data, bounds, pw = run_analysis(data=data)
    create_figures(output_dir=str(out_dir), data=data)
    joint = compute_joint_phi_bounds_two_market(data, n_grid=4001)
    ub = joint["upper_best"]
    lb = joint["lower_best"]
    Psi_ref = pw["Psi"]
    ub_pct_harberger = (ub["Phi"] / Psi_ref * 100.0) if Psi_ref > 0 else float("nan")
    lb_pct_harberger = (lb["Phi"] / Psi_ref * 100.0) if Psi_ref > 0 else float("nan")

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    bounds_wc = compute_shadow_price_bounds(data, with_choke=True, M=M_CHOKE)

    print(f"""
RATIONING ROBUST BOUNDS (p_bar = {data.p_bar}):

Allocation:
  - Open ({data.open_frac*100:.1f}%): q = {data.q_open:.3f}
  - Closed/Limiting ({data.closed_frac*100:.1f}%): q = {data.q_closed:.3f}

Shadow price bounds (with choke M = {M_CHOKE}):
  - Open: P in [{bounds_wc['open']['p_lower']:.2f}, {bounds_wc['open']['p_upper']:.2f}]
  - Closed/Limiting: P in [{bounds_wc['closed']['p_lower']:.2f}, {bounds_wc['closed']['p_upper']:.2f}]

	Welfare (piecewise with choke):
	  - Misallocation: L_Mis = {pw['Phi']*100:.2f}%
	  - Harberger: L_Harb = {pw['Psi']*100:.3f}%
	  - Ratio L_Mis/L_Harb: {pw['R']*100:.1f}% (ratio = {pw['R']:.2f})

	Linear bounds (no choke):
	  - L_Mis in [{bounds['Phi_lower']*100:.2f}%, {bounds['Phi_upper']*100:.2f}%]
	  - Ratio L_Mis/L_Harb: {bounds['R']*100:.1f}% (ratio = {bounds['R']:.2f}, slope-invariant)

	Joint robust bounds (adding-up binds; with peel-off penalties):
	  - Upper L_Mis = {ub['Phi']*100:.2f}% (= {ub_pct_harberger:.1f}% of L_Harb)
	  - Lower L_Mis = {lb['Phi']*100:.2f}% (= {lb_pct_harberger:.1f}% of L_Harb)
""")

    if args.write_meta or args.state_subset != "all":
        meta = {
            "subset": subset_meta,
            "station_share_calibration": {
                "n_states": int(frac_meta["n_states"]),
                "open_frac": float(data.open_frac),
                "closed_frac": float(data.closed_frac),
                "source": str(frac_meta["source"]),
            },
            "analysis_params": {
                "p_bar": float(data.p_bar),
                "shortage_rate": float(SHORTAGE_RATE),
                "epsilon_base": float(BASE_EPSILON),
                "epsilon_bounds": [float(EPSILON_L), float(EPSILON_H)],
                "choke": float(M_CHOKE),
            },
            "key_outputs": {
                "joint_phi_lower_pct": float(lb["Phi"] * 100.0),
                "joint_phi_upper_pct": float(ub["Phi"] * 100.0),
                "joint_ratio_lower": float(lb["Phi"] / Psi_ref) if Psi_ref > 0 else None,
                "joint_ratio_upper": float(ub["Phi"] / Psi_ref) if Psi_ref > 0 else None,
            },
        }
        meta_path = out_dir / "station_robust_bounds_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
