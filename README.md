# Paper Figure/Table Replication Package

This folder is the download package for the paper-facing figures and table.
It includes the code, raw data inputs, and static paper assets needed to run the package as distributed.

## Paper

This package reproduces paper outputs for:
- **Chaos and Misallocation under Price Controls** by Brian C. Albrecht, Alex Tabarrok, and Mark Whitmeyer.

## Included Inputs

Raw data required by the retained scripts:
- `data/AAA Fuel Report 1974 w State Names and total stations simplified.xlsx`
- `data/Raw Data/Full_Merged_Data_by_State.csv`

Static paper assets included locally:
- `assets/figure_rationing_total_1974.pdf`
- `assets/random_scenarios_matrix.pdf`
- `assets/figure_state_avg_shadow_map_impute_full_mid.pdf`
- `assets/figure_state_avg_shadow_map_known_exact.pdf`

## What It Generates

Run `make all` to write the paper figure PDFs into `output/`:
- `figure_rationing_total_1974.pdf` (copied from `assets/`)
- `figure_misallocation_two_market.pdf` (generated)
- `figure_box_constraints.pdf` (generated)
- `random_scenarios_matrix.pdf` (copied from `assets/`)
- `figure_station_joint_extremal_curves.pdf` (generated)
- `figure_station_joint_extremal_curves_with_choke.pdf` (generated)
- `figure_state_avg_shadow_map_impute_full_mid.pdf` (copied from `assets/`)
- `figure_state_avg_shadow_map_known_exact.pdf` (copied from `assets/`)

Optional:
- `make table` writes `output/table_assumption_interval_decomposition_impute_full_mid.csv`.

Additional generated diagnostic files (PNGs/JSON/CSV) are also written to `output/`.

Note: the two state map PDFs are shipped as assets because Plotly PDF export depends on a working local Chrome/Chromium runtime.

## What Is Excluded

- Hill-model code
- LaTeX compilation
- non-paper output pipelines

## Usage

Quick start:

```bash
cd paper_figure_replication
pip install -r requirements.txt
make all
```

Optional table build:

```bash
make table
```

Clean generated files:

```bash
make clean
```

## Dependencies

Install the minimal packages listed in:
- `requirements.txt`
