#!/usr/bin/env python3
"""Fail-closed checks for the finite-resource boundary table."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    with (RESULTS / "finite_resource_bounds.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert len({(r["topology"], r["control_authority"]) for r in rows}) == 15
    for row in rows:
        values = {key: float(value) for key, value in row.items()
                  if key not in {"topology", "nodes", "actuation_upper_lower_gap"}}
        assert all(math.isfinite(value) and value >= 0 for value in values.values())
        assert values["upper_radius"] >= values["actuation_lower"]
        if values["actuation_lower"] > 0:
            assert math.isclose(float(row["actuation_upper_lower_gap"]),
                                values["upper_radius"] / values["actuation_lower"],
                                rel_tol=1e-12)
        else:
            assert row["actuation_upper_lower_gap"] == ""
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["rows"] == len(rows)
    assert summary["minimum_positive_actuation_gap"] >= 1.0
    print("PASS: 15 finite-resource upper/lower comparisons validated")


if __name__ == "__main__":
    main()
