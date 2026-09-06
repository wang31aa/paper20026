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
