#!/usr/bin/env python3
"""Rebuild physical metrics from a user-downloaded raw cache and compare."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def compare_csv(reference: Path, candidate: Path) -> None:
    a, b = pd.read_csv(reference), pd.read_csv(candidate)
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            assert np.allclose(a[col], b[col], rtol=1e-12, atol=1e-14, equal_nan=True), col
        else:
            assert a[col].astype(str).tolist() == b[col].astype(str).tolist(), col


def compare_json(reference: Path, candidate: Path) -> None:
    a, b = json.loads(reference.read_text()), json.loads(candidate.read_text())
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], float):
            assert np.isclose(a[key], b[key], rtol=1e-12, atol=1e-154), key
        else:
            assert a[key] == b[key], key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="physical-r1-rebuild-") as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, str(root / "analyse_physical_timeseries.py"),
                        "--input", str(args.raw_dir), "--output", str(out)], check=True)
        subprocess.run([sys.executable, str(root / "analyse_topology_extension.py"),
                        "--input", str(args.raw_dir), "--edges", str(args.edges),
                        "--output", str(out)], check=True)
        for name in ("physical_metrics.csv", "topology_metrics.csv"):
            compare_csv(root / "results" / name, out / name)
        for name in ("validation.json", "topology_validation.json"):
            compare_json(root / "results" / name, out / name)
    print("PASS: raw R1 traces reproduce both released physical metric chains")


if __name__ == "__main__":
    main()
