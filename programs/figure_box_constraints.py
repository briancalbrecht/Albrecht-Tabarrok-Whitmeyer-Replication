"""Figure 3: feasible set and corner solutions under a price ceiling."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path

# Figure styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'font.family': 'sans-serif',
})

# Parameters (matching Figure 2)

# Demands at the ceiling p_bar = 0.5
D1_bar = 0.75  # Market 1 max demand
D2_bar = 0.5   # Market 2 max demand

# Total supply under the ceiling
Q_bar = 1.0  # Less than D1_bar + D2_bar = 1.25, but > D1_bar = 0.75

# Corner solutions
# E1: market 1 fully satisfied, market 2 gets remainder
E1 = (D1_bar, Q_bar - D1_bar)  # = (0.75, 0.25)
# E2: market 2 fully satisfied, market 1 gets remainder
E2 = (Q_bar - D2_bar, D2_bar)  # = (0.5, 0.5)

# Figure
fig, ax = plt.subplots(figsize=(8, 7))

# Colors
color_feasible = 'C0'
color_supply = 'C2'
color_iso = 'C1'

# Box constraint region (light shading)

# Box: 0 <= q1 <= D1_bar, 0 <= q2 <= D2_bar
box_vertices = [
    (0, 0),
    (D1_bar, 0),  # (0.75, 0)
    (D1_bar, D2_bar),  # (0.75, 0.5)
    (0, D2_bar),  # (0, 0.5)
]
box_poly = Polygon(box_vertices, alpha=0.15, facecolor='gray',
                   edgecolor='gray', linewidth=1, linestyle='--')
ax.add_patch(box_poly)

# Feasible segment: supply line clipped by the box

# Feasible segment
ax.plot([E1[0], E2[0]], [E1[1], E2[1]], color=color_feasible, linewidth=4,
        solid_capstyle='round', label='Feasible allocations', zorder=4)

# Supply line: q1 + q2 = Q_bar
q1_line = np.linspace(0, 1.0, 100)
q2_line = Q_bar - q1_line
ax.plot(q1_line, q2_line, color=color_supply, linewidth=2,
        label=r'Supply: $q_1 + q_2 = \bar{Q}$')

# Iso-cost curve: c1*q1 + c2*q2 = k, with c1 < c2 so E1 minimizes cost
iso_slope = -0.5  # = -c_1/c_2 where c_1 < c_2
iso_intercept = E1[1] - iso_slope * E1[0]  # k/c_2 where k = c_1*E1[0] + c_2*E1[1]
q2_iso = iso_slope * q1_line + iso_intercept
ax.plot(q1_line, q2_iso, color=color_iso, linewidth=1.5, linestyle='--', alpha=0.7,
        label=r'Iso-cost: $c_1 q_1 + c_2 q_2$')

# Mark corner solutions

ax.plot(*E1, 'ko', markersize=10, zorder=5)
ax.plot(*E2, 'ko', markersize=10, zorder=5)

ax.annotate('E1', xy=E1, xytext=(E1[0] + 0.05, E1[1] + 0.05),
            fontsize=12, fontweight='bold')
ax.annotate('E2', xy=E2, xytext=(E2[0] + 0.05, E2[1] + 0.05),
            fontsize=12, fontweight='bold')

# Axis labels
ax.set_xlabel(r'Quantity $q_1$')
ax.set_ylabel(r'Quantity $q_2$')

# Labels for D_i(p_bar)
ax.text(D1_bar, 0.02, r'$D_1(\bar{p})$', fontsize=11, ha='center', va='bottom')
ax.text(0.02, D2_bar, r'$D_2(\bar{p})$', fontsize=11, ha='left', va='bottom')

# Label for supply line
ax.text(0.18, 0.78, r'$q_1 + q_2 = \bar{Q}$', fontsize=12, rotation=-45,
        ha='center', va='bottom', color=color_supply)

# Label for iso-cost curve
ax.text(0.30, 0.45, r'$c_1 q_1 + c_2 q_2 = \mathrm{constant}$', fontsize=12, rotation=-27,
        ha='center', va='bottom', color=color_iso)
ax.text(0.25, 0.49, r'$(c_1 < c_2)$', fontsize=11, color=color_iso, style='italic', rotation=-27,
        ha='center', va='top')

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_aspect('equal')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Title
ax.set_title('Feasible Allocation Set under Price Ceiling')

plt.tight_layout()
out_dir = Path(__file__).resolve().parent.parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "figure_box_constraints.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: figure_box_constraints.pdf")
plt.close()

print(f"\nParameters:")
print(f"  D1(p_bar) = {D1_bar}")
print(f"  D2(p_bar) = {D2_bar}")
print(f"  Q_bar = {Q_bar}")
print(f"  E1 = {E1}")
print(f"  E2 = {E2}")
