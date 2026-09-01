#!/usr/bin/env python3
"""Rebuild both CPS adapters in isolation and compare frozen derived tables."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "results"
TABLES = ["rths_run_metrics.csv", "rths_fault_metrics.csv", "water_hil_session_metrics.csv",
          "openmct_run_metrics.csv", "openmct_representative_10ms.csv"]


def compare_csv(expected: Path, actual: Path) -> float:
    a, b = pd.read_csv(expected), pd.read_csv(actual)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise AssertionError(f"schema/shape mismatch: {expected.name}")
    for col in a.select_dtypes(exclude="number").columns:
        if not a[col].fillna("<NA>").equals(b[col].fillna("<NA>")):
            raise AssertionError(f"text mismatch: {expected.name}:{col}")
    cols = list(a.select_dtypes(include="number").columns)
    delta = float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float)))) if cols else 0.0
    if delta > 1e-12:
        raise AssertionError(f"numeric mismatch {delta}: {expected.name}")
    return delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rths-raw", type=Path, required=True)
    ap.add_argument("--water-raw", type=Path, required=True)
    ap.add_argument("--openmct-raw", type=Path, required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="cps-rebuild-") as td:
        out = Path(td)
        subprocess.run([sys.executable, str(ROOT / "run_rths.py"), "--raw", str(args.rths_raw),
                        "--out", str(out)], check=True)
        subprocess.run([sys.executable, str(ROOT / "run_water_hil.py"), "--raw", str(args.water_raw),
                        "--out", str(out)], check=True)
        subprocess.run([sys.executable, str(ROOT / "run_openmct.py"), "--raw", str(args.openmct_raw),
                        "--out", str(out)], check=True)
        deltas = {name: compare_csv(FROZEN / name, out / name) for name in TABLES}
    print(f"PASS: isolated raw-to-derived rebuild matched all tables; max deltas={deltas}")


if __name__ == "__main__":
    main()
