#!/usr/bin/env python3
"""Fail-closed checks for the dimensionless boundary outputs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "boundary_law/results"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    with (RESULTS / "dimensionless_boundary_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (RESULTS / "scalar_sharpness.csv").open(newline="") as handle:
        sharp = list(csv.DictReader(handle))
    assert summary["input_rows"] == 576
    assert summary["expanded_rows"] == 2304 == len(rows)
    keys = [(r["topology"], r["heterogeneity"], r["alpha"],
             r["gamma_state"], r["seed"], r["epsilon"]) for r in rows]
    assert len(set(keys)) == 2304
    numeric = ("heterogeneity", "alpha", "gamma_state", "epsilon", "delta",
               "tail_error", "certificate_number", "normalized_tail_residual",
               "certificate_utilization")
    assert all(math.isfinite(float(r[k])) for r in rows for k in numeric)
    recomputed_certified = 0
    recomputed_tail_within = 0
    recomputed_exceedances = 0
    for row in rows:
        delta = float(row["delta"])
        tail = float(row["tail_error"])
        epsilon = float(row["epsilon"])
        number = epsilon / delta
        residual = tail / epsilon
        certified = int(number >= 1.0)
        within = int(residual <= 1.0)
        assert abs(float(row["certificate_number"]) - number) < 1e-12
        assert abs(float(row["normalized_tail_residual"]) - residual) < 1e-12
        assert int(row["certified"]) == certified
        assert int(row["observed_within_tolerance"]) == within
        recomputed_certified += certified
        recomputed_tail_within += within
        recomputed_exceedances += certified * (1 - within)
    assert summary["ultimate_certified_rows"] == recomputed_certified == 1308
    assert summary["finite_tail_within_tolerance_rows"] == recomputed_tail_within == 2096
    assert summary["finite_tail_exceedances_among_ultimate_certified_rows"] == recomputed_exceedances == 0
    assert summary["scalar_sharpness_max_abs_error"] <= 1e-12
    assert all(abs(float(r["certificate_utilization"]) -
                   float(r["tail_error"]) / float(r["delta"])) < 1e-12 for r in rows)
    assert all(abs(float(r["ratio"]) - 1.0) < 1e-12 for r in sharp)
    result_rows = (RESULTS / "dimensionless_boundary_rows.csv").read_bytes()
    source_rows = (ROOT / "figures/source_data/Fig1_dimensionless_boundary_rows.csv").read_bytes()
    result_sharp = (RESULTS / "scalar_sharpness.csv").read_bytes()
    source_sharp = (ROOT / "figures/source_data/Fig1_scalar_sharpness.csv").read_bytes()
    assert result_rows == source_rows and result_sharp == source_sharp
    # The repository candidate intentionally excludes the 52-MB TIFF; the
    # submission workspace retains it for production use.
    for suffix in ("pdf", "svg", "png"):
        path = ROOT / "figures" / f"Fig1_dimensionless_boundary.{suffix}"
        assert path.exists() and path.stat().st_size > 1000
    tiff = ROOT / "figures" / "Fig1_dimensionless_boundary.tiff"
    if tiff.exists():
        assert tiff.stat().st_size > 1000
    print("PASS: 576-run dimensionless boundary, exact scalar sharpness and editable figure")


if __name__ == "__main__":
    main()
