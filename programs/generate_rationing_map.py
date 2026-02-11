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


def generate_map_plotly(df, output_path=None):
    """Deprecated: Plotly choropleth (kept for reference)."""
    raise RuntimeError("Plotly export disabled in this environment.")


def generate_total_rationing_map(df, output_path=None):
    """Deprecated: Plotly choropleth (kept for reference)."""
    raise RuntimeError("Plotly export disabled in this environment.")


def generate_side_by_side_maps(df, output_path=None):
    """Deprecated: Plotly export disabled in this environment."""
    if output_path is None:
        output_path = OUTPUT_DIR / "figure_rationing_side_by_side_1974.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio

        # Bins for out-of-fuel share (0-30%)
        out_bins = [0, 5, 10, 15, 20, 30]
        out_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%']
        df['out_bin'] = pd.cut(df['out_of_fuel'], bins=out_bins, labels=out_labels, include_lowest=True)
        df['out_bin_num'] = pd.cut(df['out_of_fuel'], bins=out_bins, labels=range(len(out_labels)), include_lowest=True).astype(float)

        # Bins for limiting purchases (0-80%)
        lim_bins = [0, 15, 30, 45, 60, 80]
        lim_labels = ['0-15%', '15-30%', '30-45%', '45-60%', '60-80%']
        df['lim_bin'] = pd.cut(df['limiting'], bins=lim_bins, labels=lim_labels, include_lowest=True)
        df['lim_bin_num'] = pd.cut(df['limiting'], bins=lim_bins, labels=range(len(lim_labels)), include_lowest=True).astype(float)

        # Two-panel layout
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
            subplot_titles=('A. Out of Fuel', 'B. Limiting Purchases'),
            horizontal_spacing=0.02
        )

        # Discrete bin palettes
        out_colors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
        lim_colors = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']

        # Left: out of fuel
        fig.add_trace(
            go.Choropleth(
                locations=df['abbrev'],
                z=df['out_bin_num'],
                locationmode='USA-states',
                colorscale=[[i/4, out_colors[i]] for i in range(5)],
                zmin=0,
                zmax=4,
                colorbar=dict(
                    title='Out of Fuel',
                    tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
                    ticktext=out_labels,
                    len=0.4,
                    y=0.5,
                    x=0.44
                ),
                hovertemplate='%{location}<br>Out of Fuel: %{customdata:.1f}%<extra></extra>',
                customdata=df['out_of_fuel']
            ),
            row=1, col=1
        )

        # Right: limiting purchases
        fig.add_trace(
            go.Choropleth(
                locations=df['abbrev'],
                z=df['lim_bin_num'],
                locationmode='USA-states',
                colorscale=[[i/4, lim_colors[i]] for i in range(5)],
                zmin=0,
                zmax=4,
                colorbar=dict(
                    title='Limiting',
                    tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
                    ticktext=lim_labels,
                    len=0.4,
                    y=0.5,
                    x=1.0
                ),
                hovertemplate='%{location}<br>Limiting: %{customdata:.1f}%<extra></extra>',
                customdata=df['limiting']
            ),
            row=1, col=2
        )

        # Layout and title
        fig.update_layout(
            title_text='Gasoline Station Rationing by State, February 1974',
            title_x=0.5,
            title_font_size=16,
            geo=dict(
                scope='usa',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
            ),
            geo2=dict(
                scope='usa',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            font=dict(size=11)
        )

        # Save PNG via kaleido
        pio.write_image(fig, str(output_path), width=1400, height=500, scale=2)
        print(f"Saved: {output_path}")
        return True

    except Exception as e:
        print(f"Error generating side-by-side maps: {e}")
        return False


def generate_map_matplotlib(df, output_path=None):
    """Deprecated: geopandas map not configured here."""
    try:
        import geopandas as gpd

        # Natural Earth low-res doesn't include US states; force fallback
        us_states = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # This won't work for US states, need a different approach
        raise ImportError("Need US-specific shapefile")

    except (ImportError, Exception):
        print("Geopandas approach not available, using simple bar chart instead...")
        return generate_bar_chart(df, output_path)


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
