PYTHON ?= python3
PROGRAMS := programs
ASSETS := assets
OUT := output
MPLCONFIGDIR ?= /tmp/mplconfig

ENV_VARS := MPLBACKEND=Agg MPLCONFIGDIR=$(MPLCONFIGDIR) KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

FIG1 := $(OUT)/figure_rationing_total_1974.pdf
FIG2 := $(OUT)/figure_misallocation_two_market.pdf
FIG3 := $(OUT)/figure_box_constraints.pdf
FIG4 := $(OUT)/random_scenarios_matrix.pdf
FIG5 := $(OUT)/figure_station_joint_extremal_curves.pdf
FIG6 := $(OUT)/figure_station_joint_extremal_curves_with_choke.pdf
FIG7 := $(OUT)/figure_state_avg_shadow_map_impute_full_mid.pdf
FIG8 := $(OUT)/figure_state_avg_shadow_map_known_exact.pdf

PAPER_FIGURES := $(FIG1) $(FIG2) $(FIG3) $(FIG4) $(FIG5) $(FIG6) $(FIG7) $(FIG8)
TABLE1 := $(OUT)/table_assumption_interval_decomposition_impute_full_mid.csv

R_MAIN := $(ASSETS)/figure_rationing_total_1974.pdf
R_CHAOS := $(ASSETS)/random_scenarios_matrix.pdf
R_MAP_MAIN := $(ASSETS)/figure_state_avg_shadow_map_impute_full_mid.pdf
R_MAP_KNOWN := $(ASSETS)/figure_state_avg_shadow_map_known_exact.pdf

.PHONY: all figures table check clean help

all: figures check

figures: $(PAPER_FIGURES)

$(OUT):
	@mkdir -p $(OUT) $(OUT)/scenario_imputed_maps $(OUT)/scenario_known_exact_maps $(OUT)/assumption_decomposition $(MPLCONFIGDIR)

# --- Static assets stored locally in assets/ ---
$(FIG1): $(R_MAIN) | $(OUT)
	@if [ ! -f "$(R_MAIN)" ]; then echo "Missing static asset: $(R_MAIN)"; exit 1; fi
	@cp -f "$(R_MAIN)" "$@"

$(FIG4): $(R_CHAOS) | $(OUT)
	@if [ ! -f "$(R_CHAOS)" ]; then echo "Missing static asset: $(R_CHAOS)"; exit 1; fi
	@cp -f "$(R_CHAOS)" "$@"

$(FIG7): $(R_MAP_MAIN) | $(OUT)
	@if [ ! -f "$(R_MAP_MAIN)" ]; then echo "Missing static asset: $(R_MAP_MAIN)"; exit 1; fi
	@cp -f "$(R_MAP_MAIN)" "$@"

$(FIG8): $(R_MAP_KNOWN) | $(OUT)
	@if [ ! -f "$(R_MAP_KNOWN)" ]; then echo "Missing static asset: $(R_MAP_KNOWN)"; exit 1; fi
	@cp -f "$(R_MAP_KNOWN)" "$@"

# --- Theory figures ---
$(FIG2): | $(OUT)
	@cd $(PROGRAMS) && $(ENV_VARS) $(PYTHON) figure_misallocation_two_market.py
	@if [ ! -f "$@" ]; then echo "Missing generated figure: $@"; exit 1; fi

$(FIG3): | $(OUT)
	@cd $(PROGRAMS) && $(ENV_VARS) $(PYTHON) figure_box_constraints.py
	@if [ ! -f "$@" ]; then echo "Missing generated figure: $@"; exit 1; fi

# --- Station robust figures (single run produces both) ---
$(OUT)/.stamp_station: | $(OUT)
	@cd $(PROGRAMS) && $(ENV_VARS) $(PYTHON) robust_bounds.py --state-subset all --write-meta --out-dir ../$(OUT)
	@touch "$@"

$(FIG5): $(OUT)/.stamp_station | $(OUT)
	@if [ ! -f "$@" ]; then echo "Missing generated figure: $@"; exit 1; fi

$(FIG6): $(OUT)/.stamp_station | $(OUT)
	@if [ ! -f "$@" ]; then echo "Missing generated figure: $@"; exit 1; fi

# --- Assumption decomposition table (optional) ---
table: $(TABLE1)

$(TABLE1): | $(OUT)
	@$(ENV_VARS) $(PYTHON) -u $(PROGRAMS)/assumption_interval_decomposition.py \
		--scenario-id impute_full_mid \
		--adding-up national \
		--out-dir $(OUT)/assumption_decomposition \
		--n-grid 1201 \
		--outer-search-grid 401 \
		--outer-max-iters 1 \
		--outer-starts 2 \
		--outer-coord-grid 2
	@cp -f $(OUT)/assumption_decomposition/table_assumption_interval_decomposition_impute_full_mid.csv "$@"

check:
	@for f in $(PAPER_FIGURES); do \
		if [ ! -f "$$f" ]; then echo "Missing figure: $$f"; exit 1; fi; \
	done
	@echo "All paper figure PDFs are present in $(OUT)/."

clean:
	@rm -f $(OUT)/*.pdf $(OUT)/*.csv $(OUT)/.stamp_* $(OUT)/scenario_imputed_maps/* $(OUT)/scenario_known_exact_maps/* $(OUT)/assumption_decomposition/*
	@echo "Cleaned generated files in $(OUT)/."

help:
	@echo "Targets:"
	@echo "  make all                # Build all paper figure PDFs"
	@echo "  make figures            # Same as all, without final check banner"
	@echo "  make table              # Build table_assumption_interval_decomposition_impute_full_mid.csv"
	@echo "  make check              # Verify all 8 paper figure PDFs exist"
	@echo "  make clean              # Remove local generated PDFs and stamps"
