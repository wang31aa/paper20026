#!/usr/bin/env python3
import csv,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import run_v13
R=Path(__file__).resolve().parent;C=json.loads((R/'V14_BASELINE_CONTRACT.json').read_text());O=R/'results'
tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in C['seeds'] for p in C['policies']]
with ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(run_v13.simulate,tasks,chunksize=20))
with (O/'v14_baseline_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'conditions':len(rows)//len(C['policies'])}))
