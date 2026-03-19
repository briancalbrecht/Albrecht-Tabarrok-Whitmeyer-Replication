"""Generate Figure 1: state-level rationing map (AAA, Feb 1974)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
DATA_FILE = PACKAGE_ROOT / "data" / "AAA Fuel Report 1974 w State Names and total stations simplified.xlsx"
OUTPUT_DIR = PACKAGE_ROOT / "output"

# State name to USPS abbreviation
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

def load_data():
    """Load AAA survey data and compute total rationing rates."""
    df = pd.read_excel(DATA_FILE)
    df = df[df['State'].notna()].copy()

    # Total rationing = out of fuel + limiting purchases
    df['out_of_fuel'] = df['%  Out of Fuel'].fillna(0)
    df['limiting'] = df['% Limiting Purchases'].fillna(0)
    df['total_rationing'] = df['out_of_fuel'] + df['limiting']

    # Map state names to abbreviations for plotting
    df['abbrev'] = df['State'].map(STATE_ABBREVS)

    return df


def generate_bar_chart(df, output_path=None):
    """Fallback: horizontal stacked bars by state."""
    if output_path is None:
        output_path = OUTPUT_DIR / "figure_rationing_total_1974_bar.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort for readable labels
    df_sorted = df.sort_values('total_rationing', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 14))

    y_pos = np.arange(len(df_sorted))

    # Stacked bars: out of fuel + limiting
    bars_out = ax.barh(y_pos, df_sorted['out_of_fuel'], color='darkred',
                       label='Out of Fuel', alpha=0.9)
    bars_limit = ax.barh(y_pos, df_sorted['limiting'], left=df_sorted['out_of_fuel'],
                         color='orange', label='Limiting Purchases', alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['State'], fontsize=8)
    ax.set_xlabel('Percentage of Stations (%)', fontsize=11)
    ax.set_title('Gasoline Station Rationing by State, February 1974', fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    # Mean line for reference
    mean_total = df_sorted['total_rationing'].mean()
    ax.axvline(mean_total, color='black', linestyle='--', alpha=0.7,
               label=f'Mean: {mean_total:.1f}%')

    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved bar chart: {output_path}")
    return True


if __name__ == '__main__':
    print("Loading data...")
    df = load_data()

    print(f"\nTotal rationing statistics:")
    print(f"  Mean: {df['total_rationing'].mean():.1f}%")
    print(f"  Min:  {df['total_rationing'].min():.1f}% ({df.loc[df['total_rationing'].idxmin(), 'State']})")
    print(f"  Max:  {df['total_rationing'].max():.1f}% ({df.loc[df['total_rationing'].idxmax(), 'State']})")

    print(f"\nOut of fuel statistics:")
    print(f"  Mean: {df['out_of_fuel'].mean():.1f}%")
    print(f"  Max:  {df['out_of_fuel'].max():.1f}% ({df.loc[df['out_of_fuel'].idxmax(), 'State']})")

    print(f"\nLimiting purchases statistics:")
    print(f"  Mean: {df['limiting'].mean():.1f}%")
    print(f"  Max:  {df['limiting'].max():.1f}% ({df.loc[df['limiting'].idxmax(), 'State']})")

    print("\nTop 10 states by total rationing:")
    for _, row in df.nlargest(10, 'total_rationing').iterrows():
        print(f"  {row['State']}: {row['total_rationing']:.1f}% ({row['out_of_fuel']:.1f}% out + {row['limiting']:.1f}% limiting)")

    print("\nGenerating bar chart (supplemental)...")
    generate_bar_chart(df)
