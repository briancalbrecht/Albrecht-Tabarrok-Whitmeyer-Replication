"""
Runtime environment checks for the replication pipeline.

Usage:
    python programs/check_runtime_env.py
    python programs/check_runtime_env.py --require-map-export
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Mirror Makefile/scripting defaults so dependency checks do not trip OpenMP SHM issues.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

REQUIRED_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("plotly", "plotly"),
    ("kaleido", "kaleido"),
    ("openpyxl", "openpyxl"),
)

REQUIRED_DATA_FILES: Tuple[str, ...] = (
    "data/AAA Fuel Report 1974 w State Names and total stations simplified.xlsx",
    "data/Raw Data/Full_Merged_Data_by_State.csv",
)


def _check_python_version() -> Optional[str]:
    if sys.version_info < (3, 9):
        return f"Need Python >= 3.9, found {sys.version.split()[0]}"
    return None


def _check_package(import_name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:
        return False, str(exc)


def _check_data_files(repo_root: Path) -> List[str]:
    missing: List[str] = []
    for rel in REQUIRED_DATA_FILES:
        path = repo_root / rel
        if not path.exists():
            missing.append(str(path))
    return missing


def _check_plotly_export() -> Tuple[bool, str]:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as exc:
        return False, f"Could not import plotly export modules: {exc}"

    try:
        fig = go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1], mode="lines"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plotly_export_smoke.png"
            pio.write_image(fig, str(out), width=200, height=120, scale=1)
            if not out.exists() or out.stat().st_size <= 0:
                return False, "write_image returned without creating output file."
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local runtime dependencies for replication.")
    parser.add_argument(
        "--require-map-export",
        action="store_true",
        help="Fail if Plotly static image export is unavailable (requires kaleido + local Chrome/Chromium).",
    )
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    py_err = _check_python_version()
    if py_err:
        errors.append(py_err)
    else:
        print(f"[ok] python {sys.version.split()[0]}")

    for display_name, import_name in REQUIRED_PACKAGES:
        ok, info = _check_package(import_name)
        if ok:
            print(f"[ok] {display_name} {info}")
        else:
            errors.append(f"Missing/invalid package `{display_name}`: {info}")

    missing_files = _check_data_files(REPO_ROOT)
    if missing_files:
        for path in missing_files:
            errors.append(f"Missing required data file: {path}")
    else:
        print("[ok] required data files present")

    export_ok, export_info = _check_plotly_export()
    if export_ok:
        print("[ok] plotly static export (kaleido/chrome) available")
    else:
        message = (
            "Plotly static export unavailable. State map PDF/PNG export will fail. "
            f"Details: {export_info}"
        )
        if args.require_map_export:
            errors.append(message)
        else:
            warnings.append(message)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
