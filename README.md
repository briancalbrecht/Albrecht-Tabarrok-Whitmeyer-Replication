# Paper Figure/Table Replication Package

This folder is the download package for the paper-facing figures and table.
It includes the code, raw data inputs, and static paper assets needed to run the package as distributed.

## Included Inputs

Raw data required by the retained scripts:
- `data/AAA Fuel Report 1974 w State Names and total stations simplified.xlsx`
- `data/Raw Data/Full_Merged_Data_by_State.csv`

Static paper assets included locally:
- `assets/figure_rationing_total_1974.pdf`
- `assets/random_scenarios_matrix.pdf`
- `assets/figure_state_avg_shadow_map_impute_full_mid.pdf`
- `assets/figure_state_avg_shadow_map_known_exact.pdf`

## Paper Outputs Covered

`make all` produces/checks these figure PDFs in `output/`:
1. `figure_rationing_total_1974.pdf`
2. `figure_misallocation_two_market.pdf`
3. `figure_box_constraints.pdf`
4. `random_scenarios_matrix.pdf`
5. `figure_station_joint_extremal_curves.pdf`
6. `figure_station_joint_extremal_curves_with_choke.pdf`
7. `figure_state_avg_shadow_map_impute_full_mid.pdf`
8. `figure_state_avg_shadow_map_known_exact.pdf`

Optional table target:
1. `make table` writes `table_assumption_interval_decomposition_impute_full_mid.csv`

Note: the two state map PDFs are shipped as static assets because Plotly PDF export depends on a working local Chrome/Chromium runtime.

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
