#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
R=Path(__file__).resolve().parent/'results'
rows=list(csv.DictReader((R/'summary.csv').open()))
assert len(rows)==24
assert {(int(r['seed']),r['policy']) for r in rows} == {(s,p) for s in range(12) for p in ('all_coupled','gated')}
for r in rows:
    assert all(math.isfinite(float(r[k])) for k in ('survival_time','safe_time_fraction','final_trusted_nodes'))
allv=[float(r['survival_time']) for r in rows if r['policy']=='all_coupled']
gate=[float(r['survival_time']) for r in rows if r['policy']=='gated']
assert math.isclose(sum(sorted(allv)[5:7])/2,22.955,abs_tol=1e-12)
assert gate == [30.0]*12
assert json.load((R/'summary.json').open())['gated']['safe_time_fraction']==1.0
print('PASS: 12 paired residual-gating trials and reported medians validated')
