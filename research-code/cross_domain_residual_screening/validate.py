#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
R=Path(__file__).resolve().parent/'results';rows=list(csv.DictReader((R/'all_cases.csv').open()));s=json.loads((R/'summary.json').read_text())
assert len(rows)==11==s['cases'];assert len({r['case'] for r in rows})==11
assert all(math.isfinite(float(r['effect_ratio'])) for r in rows)
assert {r['case'] for r in rows if r['dynamic_gate_eligible']=='yes'}=={'closed nonlinear network','heterogeneous cluster stress','robot','vehicle','uav'}
assert next(r for r in rows if r['case']=='structural RTHS')['result']=='fail'
assert next(r for r in rows if r['case']=='swing-governor grid')['result']=='worse_than_local'
print('PASS: 11 retained cases, eligibility tiers and endpoint-specific outcomes')
