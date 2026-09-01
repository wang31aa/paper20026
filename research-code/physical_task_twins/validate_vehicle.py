#!/usr/bin/env python3
"""Validate completeness and paired outcomes of the vehicle task grid."""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    with (HERE / "results/vehicle_physical_tasks.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    keys = {(r["stress"], r["delay_s"], r["policy"]) for r in rows}
    assert len(keys) == 36
    for row in rows:
        for key in ("stress", "delay_s", "first_violation_s", "minimum_margin_m",
                    "within_margin_fraction"):
            assert math.isfinite(float(row[key]))
        assert 0 <= float(row["within_margin_fraction"]) <= 1
        assert row["policy"] in {"all_coupled", "gated", "gated_safety_filter"}
    paired = {}
    for row in rows:
        paired.setdefault((row["stress"], row["delay_s"]), {})[row["policy"]] = row
    assert all(set(pair) == {"all_coupled", "gated", "gated_safety_filter"}
               for pair in paired.values())
    print("PASS: 36 vehicle physical-task policy conditions validated")


if __name__ == "__main__":
    main()
