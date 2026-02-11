"""Figure 2: two-market misallocation under a price ceiling."""

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

# Linear demand setup (matches Figure 3 geometry)

# Market 1: P = a1 - b1*Q (higher intercept)
a1, b1 = 1.25, 1.0  # Choke price = 1.25, slope = 1

# Market 2: P = a2 - b2*Q (lower intercept)
a2, b2 = 1.0, 1.0  # Choke price = 1.0, slope = 1

# Price ceiling
p_bar = 0.5

# Demands at the ceiling
q1_bar = (a1 - p_bar) / b1  # = 0.75
q2_bar = (a2 - p_bar) / b2  # = 0.5

# Total supply under ceiling (shortage overall, but enough to fully serve market 1)
Q_bar = 1.0  # Less than q1_bar + q2_bar = 1.25, but > D1_bar = 0.75

# Efficient allocation: equalize shadow prices subject to q1 + q2 = Q_bar
q1_eff = 0.625
q2_eff = 0.375
p_star = a1 - b1 * q1_eff  # = 0.625

# Corner allocation: market 1 gets full demand, market 2 gets the residual
q1_corner = q1_bar  # = 0.75 = D1_bar (exactly satisfied, no shortage)
q2_corner = Q_bar - q1_bar  # = 0.25 (shortage in market 2)
p1_corner = a1 - b1 * q1_corner  # = 0.5 = p_bar (shadow price = ceiling)
p2_corner = a2 - b2 * q2_corner  # = 0.75 > p_bar (shortage)

# 2x3 layout: markets 1-2-aggregate for panels A and B
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

q_range = np.linspace(0, 1.5, 100)

# Colors
color_demand = 'C0'
color_supply = 'C2'
color_dwl = 'C1'

# Panel A: efficient allocation
ax1, ax2, ax3 = axes[0]

# Market 1 (efficient)
p1_curve = np.maximum(a1 - b1 * q_range, 0)
ax1.plot(q_range, p1_curve, color_demand, linewidth=2, label='Demand $D_1$')
ax1.axhline(y=p_bar, color='gray', linestyle='--', linewidth=1.5, label=r'Ceiling $\bar{p}$')
ax1.axhline(y=p_star, color='C4', linestyle=':', linewidth=1.5, label=r'Shadow $p^*$')

# DWL triangle a
triangle_a = Polygon([(q1_eff, p_bar), (q1_bar, p_bar), (q1_eff, p_star)],
                     alpha=0.4, facecolor=color_dwl, edgecolor=color_dwl, linewidth=1.5)
ax1.add_patch(triangle_a)
ax1.text(q1_bar + 0.05, 0.55, 'a', fontsize=18, fontweight='bold', color=color_dwl)

ax1.plot(q1_eff, p_star, 'ko', markersize=8)
ax1.axvline(x=q1_eff, color='k', linestyle=':', alpha=0.3)
ax1.set_xlim(0, 1.5)
ax1.set_ylim(0, 1.3)
ax1.set_xlabel('Quantity $q_1$')
ax1.set_ylabel('Price $p$')
ax1.set_title('Market 1')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Market 2 (efficient)
p2_curve = np.maximum(a2 - b2 * q_range, 0)
ax2.plot(q_range, p2_curve, color_demand, linewidth=2, label='Demand $D_2$')
ax2.axhline(y=p_bar, color='gray', linestyle='--', linewidth=1.5, label=r'Ceiling $\bar{p}$')
ax2.axhline(y=p_star, color='C4', linestyle=':', linewidth=1.5, label=r'Shadow $p^*$')

# DWL triangle b
triangle_b = Polygon([(q2_eff, p_bar), (q2_bar, p_bar), (q2_eff, p_star)],
                     alpha=0.4, facecolor=color_dwl, edgecolor=color_dwl, linewidth=1.5)
ax2.add_patch(triangle_b)
ax2.text(q2_bar + 0.05, 0.55, 'b', fontsize=18, fontweight='bold', color=color_dwl)

ax2.plot(q2_eff, p_star, 'ko', markersize=8)
ax2.axvline(x=q2_eff, color='k', linestyle=':', alpha=0.3)
ax2.set_xlim(0, 1.5)
ax2.set_ylim(0, 1.3)
ax2.set_xlabel('Quantity $q_2$')
ax2.set_ylabel('Price $p$')
ax2.set_title('Market 2')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Aggregate (efficient). Demand has a kink where market 2 drops out.
p_agg_high = np.linspace(a2, a1, 50)
q_agg_high = (a1 - p_agg_high) / b1
p_agg_low = np.linspace(0.01, a2, 50)
q_agg_low = (a1 - p_agg_low)/b1 + (a2 - p_agg_low)/b2
q_agg = np.concatenate([q_agg_low, q_agg_high])
p_agg = np.concatenate([p_agg_low, p_agg_high])

ax3.plot(q_agg, p_agg, color_demand, linewidth=2, label='Aggregate $D$')
ax3.axhline(y=p_bar, color='gray', linestyle='--', linewidth=1.5, label=r'Ceiling $\bar{p}$')
ax3.axhline(y=p_star, color='C4', linestyle=':', linewidth=1.5, label=r'Shadow $p^*$')
ax3.axvline(x=Q_bar, color=color_supply, linestyle='-', linewidth=2, label=r'Supply $\bar{Q}$')

# DWL triangle c = a + b
Q_bar_total = q1_bar + q2_bar
triangle_c = Polygon([(Q_bar, p_bar), (Q_bar_total, p_bar), (Q_bar, p_star)],
                     alpha=0.4, facecolor=color_dwl, edgecolor=color_dwl, linewidth=1.5)
ax3.add_patch(triangle_c)
ax3.text(Q_bar_total + 0.05, 0.55, 'c', fontsize=18, fontweight='bold', color=color_dwl)

ax3.plot(Q_bar, p_star, 'ko', markersize=8)
ax3.set_xlim(0, 2.0)
ax3.set_ylim(0, 1.3)
ax3.set_xlabel('Quantity $Q$')
ax3.set_ylabel('Price $p$')
ax3.set_title('Aggregate Market\n(Panel A: Efficient, $a + b = c$)')
ax3.legend(loc='upper right', framealpha=0.9)
ax3.grid(True, alpha=0.3)

# Panel B: corner allocation
ax4, ax5, ax6 = axes[1]

# Market 1 (corner)
ax4.plot(q_range, p1_curve, color_demand, linewidth=2, label='Demand $D_1$')
ax4.axhline(y=p_bar, color='gray', linestyle='--', linewidth=1.5, label=r'Ceiling $\bar{p}$')

# No shortage in market 1, so DWL in market 1 is zero
ax4.text(0.85, 0.55, 'a = 0', fontsize=14, fontweight='bold', color=color_dwl)

ax4.plot(q1_corner, p1_corner, 'ko', markersize=8)
ax4.axvline(x=q1_corner, color='k', linestyle=':', alpha=0.3)
ax4.set_xlim(0, 1.5)
ax4.set_ylim(0, 1.3)
ax4.set_xlabel('Quantity $q_1$')
ax4.set_ylabel('Price $p$')
ax4.set_title('Market 1 (no shortage)')
ax4.legend(loc='upper right', framealpha=0.9)
ax4.grid(True, alpha=0.3)

# Market 2 (corner)
ax5.plot(q_range, p2_curve, color_demand, linewidth=2, label='Demand $D_2$')
ax5.axhline(y=p_bar, color='gray', linestyle='--', linewidth=1.5, label=r'Ceiling $\bar{p}$')

# Market 2 has shortage: shadow price above the ceiling
ax5.axhline(y=p2_corner, color='C3', linestyle=':', linewidth=1.5, label=r'Shadow $p_2$')

# DWL triangle b
triangle_b_corner = Polygon([(q2_corner, p_bar), (q2_corner, p2_corner), (q2_bar, p_bar)],
                            alpha=0.4, facecolor=color_dwl, edgecolor=color_dwl, linewidth=1.5)
ax5.add_patch(triangle_b_corner)
ax5.text(q2_bar + 0.05, 0.55, 'b', fontsize=18, fontweight='bold', color=color_dwl)

# Mark market 2 allocation
ax5.plot(q2_corner, p2_corner, 'ko', markersize=8)
ax5.axvline(x=q2_corner, color='k', linestyle=':', alpha=0.3)

ax5.set_xlim(0, 1.5)
ax5.set_ylim(0, 1.3)
ax5.set_xlabel('Quantity $q_2$')
ax5.set_ylabel('Price $p$')
ax5.set_title('Market 2 (shortage)')
ax5.legend(loc='upper right', framealpha=0.9)
ax5.grid(True, alpha=0.3)

# Aggregate (corner)
ax6.plot(q_agg, p_agg, color_demand, linewidth=2, label='Aggregate $D$')
ax6.axhline(y=p_bar, color='gray', linestyle='--', linewidth=1.5, label=r'Ceiling $\bar{p}$')
ax6.axvline(x=Q_bar, color=color_supply, linestyle='-', linewidth=2, label=r'Supply $\bar{Q}$')

# Show both shadow prices
ax6.axhline(y=p1_corner, color='C3', linestyle=':', linewidth=1, alpha=0.7)
ax6.axhline(y=p2_corner, color='C3', linestyle=':', linewidth=1, alpha=0.7)

# Total DWL is larger than the efficient Harberger triangle
triangle_total = Polygon([(Q_bar, p_bar), (Q_bar_total, p_bar), (Q_bar, p_star)],
                         alpha=0.4, facecolor=color_dwl, edgecolor=color_dwl, linewidth=1.5)
ax6.add_patch(triangle_total)
ax6.text(Q_bar_total + 0.05, 0.55, 'c', fontsize=18, fontweight='bold', color=color_dwl)

# Note: triangle c uses efficient p* to show the Harberger benchmark
ax6.annotate('$a + b > c$\n(misallocation adds DWL)', xy=(0.4, 1.15), fontsize=12,
             fontweight='bold', color='C3', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='C3', alpha=0.8))
ax6.text(1.35, 0.58, '$c$ = aggregate\nHarberger\n(efficient case)', fontsize=9,
         color='gray', ha='left', va='bottom', style='italic')

ax6.plot(Q_bar, p1_corner, 'ko', markersize=8)
ax6.set_xlim(0, 2.0)
ax6.set_ylim(0, 1.3)
ax6.set_xlabel('Quantity $Q$')
ax6.set_ylabel('Price $p$')
ax6.set_title('Aggregate Market\n(Panel B: Corner, $a + b > c$)')
ax6.legend(loc='upper right', framealpha=0.9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = Path(__file__).resolve().parent.parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "figure_misallocation_two_market.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: figure_misallocation_two_market.pdf")
plt.close()

print("\nPanel A - Efficient allocation:")
print(f"  q1* = {q1_eff}, q2* = {q2_eff}")
print(f"  Shadow price p* = {p_star}")
print(f"  DWL: a + b = c")

print("\nPanel B - Corner allocation:")
print(f"  q1 = {q1_corner}, q2 = {q2_corner}")
print(f"  Shadow prices: p1 = {p1_corner}, p2 = {p2_corner}")
print(f"  DWL: a + b > c (misallocation loss)")
