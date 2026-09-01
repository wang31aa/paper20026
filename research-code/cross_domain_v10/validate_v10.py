#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, itertools, json, math
from pathlib import Path

R=Path(__file__).resolve().parent
C=json.loads((R/'V10_FROZEN_CONTRACT.json').read_text())
S=json.loads((R/'results/v10_summary.json').read_text())
rows=list(csv.DictReader((R/'results/v10_runs.csv').open()))
expected=list(itertools.product(C['domains'],C['sizes'],C['topologies'],C['eta'],C['heldout_seeds'],C['policies']))
keys=[(r['domain'],int(r['n']),r['topology'],float(r['eta']),int(r['seed']),r['policy']) for r in rows]
assert S['contract_sha256']==hashlib.sha256((R/'V10_FROZEN_CONTRACT.json').read_bytes()).hexdigest()
assert len(rows)==len(expected)==3840
assert len(set(keys))==len(keys) and set(keys)==set(expected)
for r in rows:
    for k in ('eta','rho','gamma','task_error','control_energy'):
        assert math.isfinite(float(r[k])),(k,r)
    assert float(r['gamma'])>0 and abs(float(r['rho'])*float(r['gamma'])-float(r['eta']))<1e-9
    ok=(float(r['task_error']) <= (C['robot_task_tube'] if r['domain']=='robot' else C['motor_speed_tube']))
    if r['domain']=='robot':
        assert math.isfinite(float(r['minimum_task_margin']))
        ok=ok and float(r['minimum_task_margin'])>=C['robot_minimum_spacing']
    assert int(r['success'])==int(ok)
print(json.dumps({'status':'PASS','rows':len(rows),'unique_keys':len(set(keys)),
                  'contract_sha256':S['contract_sha256']},indent=2))
