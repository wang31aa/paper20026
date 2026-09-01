#!/usr/bin/env python3
import csv
import math
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "results/robot_closed_loop_tasks.csv").open()))
assert len(rows) == 400
assert len({(r["run"], r["fault"], r["policy"]) for r in rows}) == 400
assert {r["policy"] for r in rows} == {"all_coupled", "binary_gate", "continuous_weight", "connectivity_gate"}
for row in rows:
    for key in ("first_full_loss_s", "first_core_loss_s", "full_within_fraction",
                "core_within_fraction", "full_tail_error", "core_tail_error",
                "control_effort", "mean_candidate_weight"):
        value = float(row[key])
        assert math.isfinite(value) and value >= 0
    assert 0 <= float(row["full_within_fraction"]) <= 1
    assert 0 <= float(row["core_within_fraction"]) <= 1
    assert 0 <= float(row["mean_candidate_weight"]) <= 1
print("PASS: 400 closed-loop robot evaluations independently parsed and checked")
