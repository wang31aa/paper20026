#!/usr/bin/env python3
"""Fail-closed validator for the preregistered MuSCAT participation V2 matrix."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
import sys

RHO = (0.10, 0.20, 0.40, 0.80, 1.60, 2.40)
SEEDS = (1, 2, 3)
POLICIES = ("permanent", "two_layer")
TAIL_THRESHOLD = 2.5e-4


def load(path: pathlib.Path) -> list[list[float]]:
    with path.open(newline="") as handle:
        rows = [[float(x) for x in row] for row in csv.reader(handle)]
    if len(rows) != 360 or any(len(row) != 16 for row in rows):
        raise AssertionError(f"{path.name}: expected 360 x 16, got {len(rows)} rows")
    if not all(math.isfinite(x) for row in rows for x in row):
        raise AssertionError(f"{path.name}: non-finite value")
    if not any(row[15] > 0 for row in rows):
        raise AssertionError(f"{path.name}: no non-zero applied torque")
    return rows


def metrics(rows: list[list[float]]) -> dict[str, float | bool]:
    # Columns: seed,rho,step,node,target(3),omega(3),observer(3),residual,edges,torque.
    task = []
    for row in rows:
        task.append(math.sqrt(sum((row[7 + j] - row[4 + j]) ** 2 for j in range(3))))
    tail = task[int(0.8 * len(task)) :]
    return {
        "peak_task_error_rad_s": max(task),
        "tail_max_task_error_rad_s": max(tail),
        "tail_mean_task_error_rad_s": statistics.fmean(tail),
        "edge_time_s": sum(row[14] for row in rows if int(row[3]) == 1),
        "torque_squared_integral_Nm2_s": sum(row[15] ** 2 for row in rows),
        "success": max(tail) <= TAIL_THRESHOLD,
    }


def main(directory: str) -> None:
    base = pathlib.Path(directory)
    result: dict[str, dict[str, dict[str, float | bool]]] = {p: {} for p in POLICIES}
    for seed in SEEDS:
        for index, rho in enumerate(RHO, 1):
            for policy in POLICIES:
                path = base / f"muscat_participation_s{seed:02d}_r{index:02d}_{policy}.csv"
                rows = load(path)
                if any(int(row[0]) != seed or abs(row[1] - rho) > 1e-12 for row in rows):
                    raise AssertionError(f"{path.name}: condition identifiers changed")
                result[policy][f"s{seed:02d}_rho{rho:.2f}"] = metrics(rows)

    summary = {}
    for policy in POLICIES:
        by_rho = []
        for rho in RHO:
            records = [result[policy][f"s{s:02d}_rho{rho:.2f}"] for s in SEEDS]
            by_rho.append({
                "rho": rho,
                "successes": sum(bool(x["success"]) for x in records),
                "median_tail_max_rad_s": statistics.median(float(x["tail_max_task_error_rad_s"]) for x in records),
                "median_edge_time_s": statistics.median(float(x["edge_time_s"]) for x in records),
                "median_energy_Nm2_s": statistics.median(float(x["torque_squared_integral_Nm2_s"]) for x in records),
            })
        summary[policy] = by_rho

    # Frozen prediction: within this source-class parameter box the information
    # term improves with rho and no upper failure is predicted on the tested grid.
    # This is evaluated, never enforced: disagreement remains visible in output.
    permanent_tail = [x["median_tail_max_rad_s"] for x in summary["permanent"]]
    monotone_tolerance = 2e-8
    checks = {
        "all_36_conditions_present": len(list(base.glob("muscat_participation_*.csv"))) == 36,
        "predicted_nonincreasing_permanent_tail": all(
            permanent_tail[i + 1] <= permanent_tail[i] + monotone_tolerance
            for i in range(len(permanent_tail) - 1)
        ),
        "predicted_no_high_rho_failure": summary["permanent"][-1]["successes"] == len(SEEDS),
    }
    payload = {
        "schema": "MUSCAT_PARTICIPATION_V2",
        "frozen_prediction": "monotone_or_saturating; no finite upper failure on rho grid",
        "tail_threshold_rad_s": TAIL_THRESHOLD,
        "summary": summary,
        "prediction_checks": checks,
        "prediction_qualified": all(checks.values()),
        "boundary": "source-class designed-heterogeneity computation; not HIL, flight, physical recovery, or cross-domain universality",
    }
    print("PARTICIPATION_V2=" + json.dumps(payload, sort_keys=True))
    # Plumbing failures are fatal. A failed scientific prediction is reported but
    # deliberately does not erase the run or invite post-hoc parameter changes.


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_participation_v2.py ARTIFACT_DIRECTORY")
    main(sys.argv[1])

