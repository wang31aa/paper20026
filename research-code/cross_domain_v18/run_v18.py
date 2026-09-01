#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, json, sys

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'cross_domain_v17'))
import run_v17 as core
C=json.loads((HERE/'V18_FROZEN_CONTRACT.json').read_text())
P=json.loads((HERE/'V18_FROZEN_PREDICTOR.json').read_text())
if P['contract_sha256']!=hashlib.sha256((HERE/'V18_FROZEN_CONTRACT.json').read_bytes()).hexdigest():
    raise SystemExit('frozen contract hash mismatch')
core.v11.C['heldout_seeds']=C['heldout_seeds']; core.v11.C['heldout_faults']=C['fault_rotation']+['mixed']
core.v13.C['confirmation_seeds']=C['heldout_seeds']; core.v13.C['fault_rotation']=C['fault_rotation']

tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['heldout_sizes']
       for t in C['heldout_topologies'] for r in C['heldout_rho']
       for s in C['heldout_seeds'] for p in C['policies']]
with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(core.execute,tasks,chunksize=20))
out=HERE/'results'; out.mkdir(exist_ok=True); fields=sorted({k for r in rows for k in r})
with (out/'v18_heldout_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
(out/'v18_manifest.json').write_text(json.dumps({'rows':len(rows),'unique_rows':len({(r['paired_replay_id'],r['policy']) for r in rows}),
 'contract_sha256':P['contract_sha256'],'predictor_sha256':hashlib.sha256((HERE/'V18_FROZEN_PREDICTOR.json').read_bytes()).hexdigest(),
 'claim_boundary':C['claim_boundary']},indent=2)+'\n')
print(json.dumps({'rows':len(rows),'unique_rows':len(rows)}))
