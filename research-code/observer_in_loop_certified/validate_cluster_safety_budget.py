#!/usr/bin/env python3
"""Recompute the manuscript's illustrative cluster-boundary counts."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "observer_in_loop_certified/results_r4/raw_runs.csv"
SOURCE = ROOT / "figures/source_data/Main_text_cluster_safety_budget_scan.csv"


def main() -> None:
    with RAW.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reported = list(csv.DictReader(stream))

    assert len(runs) == 576
    assert all(row["finite"] == "True" for row in runs)
    for row in reported:
        budget = float(row["normalized_budget"])
        certified = sum(float(run["delta"]) <= budget for run in runs)
        observed = sum(
            float(run["tail20_max_tracking"]) <= budget for run in runs
        )
        assert certified == int(row["certified_radius_count"])
        assert observed == int(row["observed_tail_count"])
        assert abs(certified / len(runs) - float(row["certified_radius_fraction"])) < 1e-15
        assert abs(observed / len(runs) - float(row["observed_tail_fraction"])) < 1e-15
        assert int(row["total_runs"]) == len(runs)
    print("PASS: four cluster-safety budgets recomputed over all 576 frozen runs")


if __name__ == "__main__":
    main()
