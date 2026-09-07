#!/usr/bin/env python3
"""Fail closed unless the MATLAB MuSCAT smoke artifact has the frozen schema."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


path = Path(sys.argv[1])
rows = [[float(value) for value in row] for row in csv.reader(path.open())]
if len(rows) != 2 or any(len(row) != 27 for row in rows):
    raise SystemExit("FAIL: expected two spacecraft and 27 frozen fields")
if [row[0] for row in rows] != [1.0, 2.0]:
    raise SystemExit("FAIL: spacecraft identifiers changed")
if not all(math.isfinite(value) for row in rows for value in row):
    raise SystemExit("FAIL: non-finite MATLAB state or control")
if any(row[20] not in (0.0, 1.0) for row in rows):
    raise SystemExit("FAIL: crash event is not Boolean")
print("PASS: MATLAB MuSCAT smoke schema, finiteness and event fields validated")

parameter_path = path.with_name("muscat_physical_parameters.csv")
physical = [[float(value) for value in row] for row in csv.reader(parameter_path.open())]
if len(physical) != 2 or any(len(row) != 11 for row in physical):
    raise SystemExit("FAIL: expected two spacecraft and 11 physical-parameter fields")
if [row[0] for row in physical] != [1.0, 2.0]:
    raise SystemExit("FAIL: physical-parameter spacecraft identifiers changed")
if not all(math.isfinite(value) for row in physical for value in row):
    raise SystemExit("FAIL: non-finite mass or inertia")
if any(row[1] <= 0 for row in physical):
    raise SystemExit("FAIL: non-positive spacecraft mass")
for row in physical:
    inertia = [row[2:5], row[5:8], row[8:11]]
    if any(inertia[i][i] <= 0 for i in range(3)):
        raise SystemExit("FAIL: non-positive principal inertia entry")
    if max(abs(inertia[i][j] - inertia[j][i]) for i in range(3) for j in range(3)) > 1e-8:
        raise SystemExit("FAIL: inertia tensor is not symmetric")
print("PASS: source-object mass and inertia tensor validated")
