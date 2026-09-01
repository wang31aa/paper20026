#!/usr/bin/env python3
import csv,math
from pathlib import Path
H=Path(__file__).resolve().parent/'results'
logs=list(csv.DictReader((H/'unified_log.csv').open()));summ=list(csv.DictReader((H/'pressure_summary.csv').open()));lodo=list(csv.DictReader((H/'leave_domain_out.csv').open()))
assert len(summ)==3*24*5 and len(lodo)==3
assert len({(r['domain'],r['condition'],r['policy']) for r in summ})==len(summ)
assert {r['policy'] for r in summ}=={'all_coupled','residual_gate','connectivity_gate','physical_filter','heuristic_high_gain'}
assert all(math.isfinite(float(r[k])) for r in summ for k in ('minimum_margin','tail_margin','control_energy','risk_score'))
assert all(r['split']=='exploratory_synthetic' for r in logs)
assert all(r['theorem_assumptions_pass']=='0' and r['physical_mapping_pass']=='0' for r in logs)
assert all(r[k]=='NA' or 0<=float(r[k])<=1 for r in lodo for k in ('sensitivity','specificity','false_negative_rate'))
print(f'PASS: {len(logs)} unified rows, {len(summ)} paired run-policy units, 3 held-domain tests')
