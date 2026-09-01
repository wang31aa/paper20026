#!/usr/bin/env python3
"""Create plotting-only numeric-index tables from validated result tables."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def write(path, fields, data):
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader(); out.writerows(data)

bounds = read(ROOT / "finite_resource_boundary/results/finite_resource_bounds.csv")
topology = {"direct_pinning": 0, "directed_chain": 1, "directed_cyclic": 2}
write(ROOT / "finite_resource_boundary/results/finite_resource_figure.csv",
      ["topology_index", "upper_lower_gap"],
      [{"topology_index": topology[r["topology"]],
        "upper_lower_gap": r["actuation_upper_lower_gap"]}
       for r in bounds if r["actuation_upper_lower_gap"]])

vehicle = read(ROOT / "physical_task_twins/results/vehicle_physical_tasks.csv")
policy = {"all_coupled": 0, "gated": 1, "gated_safety_filter": 2}
write(ROOT / "physical_task_twins/results/vehicle_figure.csv",
      ["policy_index", "minimum_margin_m"],
      [{"policy_index": policy[r["policy"]], "minimum_margin_m": r["minimum_margin_m"]} for r in vehicle])

robot = read(ROOT / "physical_task_twins/results/robot_formation_tasks.csv")
condition = {"none": 0, "formation_change": 1, "swinging": 2, "noisy": 3, "brute_force": 4}
for name in ("all_coupled", "gated"):
    subset = [{"condition_index": condition[r["fault"]], "within_tolerance_fraction": r["within_tolerance_fraction"]}
              for r in robot if r["policy"] == name]
    write(ROOT / f"physical_task_twins/results/robot_figure_rows_{'all' if name == 'all_coupled' else 'gated'}.csv",
          ["condition_index", "within_tolerance_fraction"], subset)
