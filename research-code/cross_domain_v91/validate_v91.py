#!/usr/bin/env python3
import csv,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent;C=json.loads((R/'V91_FROZEN_CONTRACT.json').read_text());rows=list(csv.DictReader((R/'results/v91_runs.csv').open()))
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho'])*len(C['heldout_seeds'])*len(C['policies']);assert len(rows)==expected==7680
keys=set()
for x in rows:
 k=(x['domain'],int(x['n']),x['topology'],float(x['rho']),int(x['seed']),x['policy']);assert k not in keys;keys.add(k)
 e=float(x['task_error']);m=float(x['minimum_task_margin']);en=float(x['control_energy']);assert all(math.isfinite(v) for v in (e,m,en))
 ok=(e<=.55 and m>=.25) if x['domain']=='robot' else (m>=0 and e<=1.2);assert int(x['success'])==int(ok)
s=json.loads((R/'results/v91_summary.json').read_text());h=hashlib.sha256((R/'V91_FROZEN_CONTRACT.json').read_bytes()).hexdigest();assert s['contract_sha256']==h
print(f'PASS V9.1: {expected} unique evaluations and physical endpoints; contract {h}')
