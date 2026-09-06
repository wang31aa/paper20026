#!/usr/bin/env python3
"""Validate paired MuSCAT source-class trajectories without outcome claims."""
from __future__ import annotations
import csv, math, sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
if len(paths) != 2:
    raise SystemExit("FAIL: expected permanent and two-layer artifacts")
rows = []
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
if len(rows[0]) != len(rows[1]):
    raise SystemExit("FAIL: paired trajectories differ in length")
print("PASS: paired source-class runs contain finite two-spacecraft state, observer, graph and applied-wheel records")
print("BOUNDARY: this qualifies computation plumbing only; outcome prediction remains NOT_QUALIFIED")
