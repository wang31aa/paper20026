#!/usr/bin/env python3
import csv, hashlib, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
c=json.loads((ROOT/'V9_FROZEN_CONTRACT.json').read_text())
rows=list(csv.DictReader((ROOT/'results/v9_runs.csv').open()))
expected=len(c['domains'])*len(c['sizes'])*len(c['topologies'])*len(c['rho'])*len(c['heldout_seeds'])*len(c['policies'])
assert len(rows)==expected==15840
keys=set()
for r in rows:
    key=(r['domain'],int(r['n']),r['topology'],float(r['rho']),int(r['seed']),r['policy'])
    assert key not in keys; keys.add(key)
    tail=float(r['tail_error']); energy=float(r['control_energy']); margin=float(r['minimum_task_margin'])
    assert math.isfinite(tail) and math.isfinite(energy) and energy>=0
    if r['domain']=='robot': recomputed=tail<=.55 and margin>=.25
    elif r['domain']=='vehicle': recomputed=margin>=0 and tail<=1.2
    else: recomputed=tail<=.45
    assert int(r['success'])==int(recomputed)
s=json.loads((ROOT/'results/v9_summary.json').read_text())
digest=hashlib.sha256((ROOT/'V9_FROZEN_CONTRACT.json').read_bytes()).hexdigest()
assert s['contract_sha256']==digest
print(f'PASS V9: {expected} unique frozen evaluations; task endpoints independently recomputed; contract {digest}')
