#!/usr/bin/env python3
import csv, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'results/robot_formation_tasks.csv').open()))
assert len(rows)==200
assert len({(r['run'],r['fault'],r['policy']) for r in rows})==200
assert all(math.isfinite(float(r[k])) for r in rows for k in ('first_loss_s','within_tolerance_fraction','tail_error'))
assert all(0<=float(r['within_tolerance_fraction'])<=1 for r in rows)
print('PASS: 200 held-out robot formation-integrity conditions validated')
