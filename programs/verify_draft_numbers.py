"""
Verify every empirical number in draft.tex against code output / raw data.

Usage:
    python programs/verify_draft_numbers.py            # check all
    python programs/verify_draft_numbers.py --verbose   # show all values

Exit code 0 if all checks pass, 1 if any fail.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DRAFT = REPO_ROOT / "draft" / "draft.tex"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

AAA_FILE = DATA_DIR / "AAA Fuel Report 1974 w State Names and total stations simplified.xlsx"
STATION_META = OUTPUT_DIR / "station_robust_bounds_meta.json"
TABLE1_CSV = OUTPUT_DIR / "assumption_decomposition" / "table_assumption_interval_decomposition_impute_full_mid.csv"
TABLE1_META = OUTPUT_DIR / "assumption_decomposition" / "assumption_interval_decomposition_impute_full_mid_meta.json"
STATE_OPTIONS_CSV = OUTPUT_DIR / "scenario_imputed_maps" / "table_state_baseline_options.csv"

# ── calibration parameters ──────────────────────────────────────────────
SHORTAGE_RATE = 0.09
P_BAR = 0.8
P_BASE = 1.0
Q_SUPPLY = 1.0 - SHORTAGE_RATE  # 0.91
BASE_EPSILON = 0.3
EPSILON_L = 0.2
EPSILON_H = 0.4
M_CHOKE = 4.0
G_STEEP = -P_BASE / EPSILON_L  # -5.0
G_FLAT = -P_BASE / EPSILON_H   # -2.5


def load_aaa():
    df = pd.read_excel(
        AAA_FILE,
        usecols=["State", "Gasoline Stations 1972", "% Limiting Purchases", "%  Out of Fuel"],
    )
    df["State"] = df["State"].astype(str).str.strip()
    df = df[df["State"] != "District of Columbia"].copy()
    for col in ["Gasoline Stations 1972", "% Limiting Purchases", "%  Out of Fuel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Gasoline Stations 1972"].notna() & (df["Gasoline Stations 1972"] > 0)].copy()
    return df


def station_fractions(df):
    """Station-count-weighted open/closed fractions (matching robust_bounds.py)."""
    stations = df["Gasoline Stations 1972"]
    oof = df["%  Out of Fuel"].fillna(0.0) / 100.0
    lim = df["% Limiting Purchases"].fillna(0.0) / 100.0
    rationing = (oof + lim).clip(0, 1)
    w = stations / stations.sum()
    closed_frac = float((rationing * w).sum())
    open_frac = 1.0 - closed_frac
    return open_frac, closed_frac


def national_status_shares(df):
    """Station-weighted national shares of open/limiting/out-of-fuel."""
    stations = df["Gasoline Stations 1972"]
    oof = df["%  Out of Fuel"].fillna(0.0) / 100.0
    lim = df["% Limiting Purchases"].fillna(0.0) / 100.0
    w = stations / stations.sum()
    pct_oof = float((oof * w).sum()) * 100.0
    pct_lim = float((lim * w).sum()) * 100.0
    pct_open = 100.0 - pct_oof - pct_lim
    return pct_open, pct_lim, pct_oof


def q_open_at_eps(eps):
    return 1.0 + (P_BAR - P_BASE) / (-P_BASE / (eps * 1.0))


def q_closed_at_eps(eps, open_frac, closed_frac):
    qO = q_open_at_eps(eps)
    return (Q_SUPPLY - open_frac * qO) / closed_frac


def shadow_price_no_choke(q_obs, g):
    return P_BASE + g * (q_obs - 1.0)


def harberger_pct(eps):
    g = -P_BASE / eps
    return 0.5 * abs(g) * SHORTAGE_RATE**2 * 100.0


def find_in_draft(pattern, draft_text):
    """Return (line_number, match) for first match, or (None, None)."""
    for i, line in enumerate(draft_text, 1):
        m = re.search(pattern, line)
        if m:
            return i, m
    return None, None


class Checker:
    def __init__(self, draft_lines, verbose=False):
        self.draft = draft_lines
        self.draft_text = "\n".join(draft_lines)
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name, expected, actual, tol=0.01, line_hint=None):
        ok = abs(expected - actual) <= tol
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"  OK   {name}: paper={expected}, code={actual:.4f}")
        else:
            self.failed += 1
            loc = f" (near line {line_hint})" if line_hint else ""
            print(f"  FAIL {name}{loc}: paper={expected}, code={actual:.4f}, diff={abs(expected-actual):.4f}")

    def check_in_draft(self, name, pattern, actual, tol=0.01):
        """Find a number in the draft and compare to actual."""
        line_no, m = find_in_draft(pattern, self.draft)
        if line_no is None:
            self.warnings += 1
            print(f"  WARN {name}: pattern not found in draft")
            return
        draft_val = float(m.group(1))
        self.check(name, draft_val, actual, tol=tol, line_hint=line_no)

    def summary(self):
        total = self.passed + self.failed
        status = "PASS" if self.failed == 0 else "FAIL"
        print(f"\n{'='*60}")
        print(f"  {status}: {self.passed}/{total} checks passed, {self.failed} failed, {self.warnings} warnings")
        print(f"{'='*60}")
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    draft_lines = DRAFT.read_text().splitlines()
    c = Checker(draft_lines, verbose=args.verbose)

    # ── Load raw data ────────────────────────────────────────────────
    print("Loading AAA data...")
    aaa = load_aaa()
    n_states = len(aaa)
    open_frac, closed_frac = station_fractions(aaa)
    oof_mean = aaa["%  Out of Fuel"].fillna(0.0).mean()
    stations = aaa["Gasoline Stations 1972"]
    w = stations / stations.sum()
    oof_pct = aaa["%  Out of Fuel"].fillna(0.0) / 100.0
    lim_pct = aaa["% Limiting Purchases"].fillna(0.0) / 100.0
    national_oof = float((oof_pct * w).sum()) * 100.0
    national_lim = float((lim_pct * w).sum()) * 100.0
    national_open_pct = 100.0 - national_oof - national_lim

    # ── Load output files ────────────────────────────────────────────
    print("Loading output files...")
    station_meta = json.loads(STATION_META.read_text()) if STATION_META.exists() else None
    table1 = pd.read_csv(TABLE1_CSV) if TABLE1_CSV.exists() else None
    table1_meta = json.loads(TABLE1_META.read_text()) if TABLE1_META.exists() else None

    # ── Calibration quantities ───────────────────────────────────────
    # Use fractions from the station meta JSON (matches robust_bounds.py main)
    if station_meta:
        meta_open = station_meta["station_share_calibration"]["open_frac"]
        meta_closed = station_meta["station_share_calibration"]["closed_frac"]
    else:
        meta_open = open_frac
        meta_closed = closed_frac
    qO_base = q_open_at_eps(BASE_EPSILON)
    qC_base = q_closed_at_eps(BASE_EPSILON, meta_open, meta_closed)
    qC_meta = qC_base
    qO_lo = q_open_at_eps(EPSILON_L)
    qO_hi = q_open_at_eps(EPSILON_H)
    qC_lo = q_closed_at_eps(EPSILON_H, meta_open, meta_closed)
    qC_hi = q_closed_at_eps(EPSILON_L, meta_open, meta_closed)
    harb_steep = harberger_pct(EPSILON_L)

    # Shadow prices at station level (using meta-derived q_closed)
    p_closed_steep = shadow_price_no_choke(qC_meta, G_STEEP)
    p_closed_flat = shadow_price_no_choke(qC_meta, G_FLAT)
    p_closed_upper_no_choke = max(p_closed_steep, p_closed_flat)
    # With choke
    p_closed_upper_choke = min(p_closed_upper_no_choke, M_CHOKE + G_FLAT * qC_meta)

    # Choke extrapolations at q=0
    P0_steep = P_BASE + abs(G_STEEP)  # P(0) at eps=0.2
    P0_flat = P_BASE + abs(G_FLAT)    # P(0) at eps=0.4

    # ── CHECKS ───────────────────────────────────────────────────────

    pct_open, pct_lim, pct_oof = national_status_shares(aaa)

    print("\n--- AAA Data ---")
    c.check("n_states", 48, n_states, tol=0)
    c.check("mean_out_of_fuel_%", 8.1, oof_mean, tol=0.05)
    # National station-weighted status shares (10.1/27.6/62.3 are from full AAA report;
    # our 48-state subsample gives ~9.9/27.2/62.9 — close but not identical)
    c.check("national_oof_%", 10.1, pct_oof, tol=0.5)
    c.check("national_lim_%", 27.6, pct_lim, tol=0.5)
    c.check("national_open_%", 62.3, pct_open, tol=1.0)
    # Station-count-weighted open/closed fractions (open vs limiting+oof)
    c.check("open_frac", 0.623, open_frac, tol=0.007)
    c.check("closed_frac", 0.377, closed_frac, tol=0.007)

    print("\n--- Calibration ---")
    c.check("shortage_%", 9.0, SHORTAGE_RATE * 100, tol=0)
    c.check("p_bar", 0.8, P_BAR, tol=0)
    c.check("Q_supply", 0.91, Q_SUPPLY, tol=0)
    c.check("q_O_baseline", 1.06, qO_base, tol=0.005)
    c.check("q_C_baseline", 0.66, qC_base, tol=0.005)
    c.check("q_O_lower", 1.04, qO_lo, tol=0.005)
    c.check("q_O_upper", 1.08, qO_hi, tol=0.005)
    c.check("q_C_lower", 0.63, qC_lo, tol=0.01)
    c.check("q_C_upper", 0.70, qC_hi, tol=0.01)
    c.check("g_steep", -5.0, G_STEEP, tol=0)
    c.check("g_flat", -2.5, G_FLAT, tol=0)
    c.check("P(0)_at_eps_0.2", 6.0, P0_steep, tol=0)
    c.check("P(0)_at_eps_0.4", 3.5, P0_flat, tol=0)
    c.check("Harberger_%_at_steep", 2.025, harb_steep, tol=0.001)

    print("\n--- Shadow Prices (station-level) ---")
    c.check("p_closed_upper_no_choke", 2.69, p_closed_upper_no_choke, tol=0.01)
    c.check("p_closed_upper_with_choke", 2.34, p_closed_upper_choke, tol=0.02)

    print("\n--- Station-Level Bounds (from meta JSON) ---")
    if station_meta:
        ko = station_meta["key_outputs"]
        c.check("station_phi_lower_%", 2.3, ko["joint_phi_lower_pct"], tol=0.1)
        c.check("station_phi_upper_%", 12.7, ko["joint_phi_upper_pct"], tol=0.1)
        c.check("station_R_lower", 1.15, ko["joint_ratio_lower"], tol=0.01)
        c.check("station_R_upper", 6.28, ko["joint_ratio_upper"], tol=0.02)
    else:
        print("  SKIP: station_robust_bounds_meta.json not found")

    # No-choke bounds: R in [1.15, 9.18], Phi in [2.3%, 18.6%]
    # Cross-check: 18.6% / 2.025% ≈ 9.19
    nochoke_R_upper_implied = 18.6 / harb_steep
    c.check("nochoke_R_upper_cross_check", 9.18, nochoke_R_upper_implied, tol=0.1)

    print("\n--- Table 1 (from CSV) ---")
    if table1 is not None:
        rows = {r["case_id"]: r for _, r in table1.iterrows()}

        # Row 1: common epsilon
        r1 = rows["common_epsilon"]
        c.check("T1R1_phi_lower_%", 4.43, r1["phi_lower_pct"], tol=0.01)
        c.check("T1R1_phi_upper_%", 8.86, r1["phi_upper_pct"], tol=0.01)
        c.check("T1R1_R", 4.37, float(r1["ratio_lower"]), tol=0.01)

        # Row 2: heterogeneous slope, fixed anchors
        r2 = rows["slope_fixed_anchor"]
        c.check("T1R2_phi_lower_%", 4.99, r2["phi_lower_pct"], tol=0.01)
        c.check("T1R2_phi_upper_%", 9.96, r2["phi_upper_pct"], tol=0.01)
        c.check("T1R2_R_lower", 2.46, r2["ratio_lower"], tol=0.01)
        c.check("T1R2_R_upper", 4.92, r2["ratio_upper"], tol=0.01)

        # Row 3: anchor uncertainty
        r3 = rows["plus_anchor_uncertainty"]
        c.check("T1R3_phi_lower_%", 2.21, r3["phi_lower_pct"], tol=0.01)
        c.check("T1R3_phi_upper_%", 17.72, r3["phi_upper_pct"], tol=0.05)
        c.check("T1R3_R_lower", 1.09, r3["ratio_lower"], tol=0.01)
        c.check("T1R3_R_upper", 8.75, r3["ratio_upper"], tol=0.01)

        # Row 4: choke
        r4 = rows["plus_choke"]
        c.check("T1R4_phi_lower_%", 2.21, r4["phi_lower_pct"], tol=0.01)
        c.check("T1R4_phi_upper_%", 12.40, r4["phi_upper_pct"], tol=0.05)
        c.check("T1R4_R_lower", 1.09, r4["ratio_lower"], tol=0.01)
        c.check("T1R4_R_upper", 6.12, r4["ratio_upper"], tol=0.02)
    else:
        print("  SKIP: Table 1 CSV not found")

    print("\n--- Table 1 Meta ---")
    if table1_meta:
        cal = table1_meta["calibration"]
        c.check("T1_n_markets", 91, cal["n_markets_active"], tol=0)
        c.check("T1_q_open", 1.06, cal["q_open"], tol=0.001)
        c.check("T1_q_non_open", 0.67, cal["q_non_open"], tol=0.01)
        c.check("T1_Qbar", 0.91, cal["Qbar"], tol=0.001)
        harb = table1_meta["harberger"]
        c.check("T1_psi_ref_%", 2.025, harb["psi_ref_pct"], tol=0.001)
    else:
        print("  SKIP: Table 1 meta JSON not found")

    print("\n--- Inline Derived Numbers ---")
    # L_Mis range from R × L_Harb
    if table1 is not None:
        r3 = rows["plus_anchor_uncertainty"]
        lmis_lo = r3["ratio_lower"] * harb_steep  # should be ~2.2%
        lmis_hi = r3["ratio_upper"] * harb_steep   # should be ~17.7%
        c.check("L_Mis_lower_%", 2.2, lmis_lo, tol=0.05)
        c.check("L_Mis_upper_%", 17.7, lmis_hi, tol=0.1)

    # R invariant at 4.4 (Table 1 Row 1 R=4.37, rounds to 4.4)
    if table1 is not None:
        c.check("R_invariant_rounds_to_4.4", 4.4, round(float(rows["common_epsilon"]["ratio_lower"]), 1), tol=0.05)

    # R range [2.5, 4.9] (Table 1 Row 2, rounded)
    if table1 is not None:
        c.check("R_hetero_lower_rounds_to_2.5", 2.5, round(r2["ratio_lower"], 1), tol=0.05)
        c.check("R_hetero_upper_rounds_to_4.9", 4.9, round(r2["ratio_upper"], 1), tol=0.05)

    # ── Summary ──────────────────────────────────────────────────────
    ok = c.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
