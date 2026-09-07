#!/usr/bin/env python3
"""Validate and descriptively summarize paired MuSCAT source-class trajectories."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
if len(paths) != 2:
    raise SystemExit("FAIL: expected permanent and two-layer artifacts")
rows = []
summaries = {}
for path in paths:
    data = [[float(x) for x in r] for r in csv.reader(path.open())]
    if not data or any(len(r) != 14 for r in data):
        raise SystemExit(f"FAIL: bad schema in {path}")
    if not all(math.isfinite(x) for r in data for x in r):
        raise SystemExit(f"FAIL: non-finite data in {path}")
    if max(r[13] for r in data) <= 0:
        raise SystemExit(f"FAIL: no applied wheel torque in {path}")
    if len({int(r[1]) for r in data}) != 2:
        raise SystemExit(f"FAIL: expected two spacecraft in {path}")
    rows.append(data)
    steps = sorted({int(r[0]) for r in data})
    tail_start = steps[max(0, int(0.75 * len(steps)) - 1)]
    task_errors = []
    observer_errors = []
    tail_errors = []
    effort = 0.0
    communication = 0.0
    for r in data:
        target, omega, observer = r[2:5], r[5:8], r[8:11]
        task_error = math.sqrt(sum((a-b)**2 for a, b in zip(omega, target)))
        observer_error = math.sqrt(sum((a-b)**2 for a, b in zip(observer, target)))
        task_errors.append(task_error)
        observer_errors.append(observer_error)
        if int(r[0]) >= tail_start:
            tail_errors.append(task_error)
        effort += r[13] ** 2
        if int(r[1]) == 1:
            communication += r[12]
    policy = "two_layer" if "two_layer" in path.stem else "permanent"
    summaries[policy] = {
        "peak_task_rate_error_rad_s": max(task_errors),
        "tail_mean_task_rate_error_rad_s": sum(tail_errors) / len(tail_errors),
        "tail_max_task_rate_error_rad_s": max(tail_errors),
        "peak_observer_rate_error_rad_s": max(observer_errors),
        "applied_torque_squared_integral_Nm2_s": effort,
        "directed_edge_time_s": communication,
        "spacecraft_time_records": len(data),
    }
if len(rows[0]) != len(rows[1]):
    raise SystemExit("FAIL: paired trajectories differ in length")
print("PASS: paired source-class runs contain finite two-spacecraft state, observer, graph and applied-wheel records")
print("DESCRIPTIVE_METRICS=" + json.dumps(summaries, sort_keys=True))
print("BOUNDARY: this qualifies computation plumbing only; outcome prediction remains NOT_QUALIFIED")
