#!/usr/bin/env python3
import csv,json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path.insert(0,str(ROOT/'cross_domain_v28'));import run_v28 as v
C=json.loads((R/'V29_OOD_CONTRACT.json').read_text());O=R/'results';O.mkdir(exist_ok=True)
tasks=[(d,n,r,s,p) for d in v.C['domains'] for n in C['sizes'] for r in C['rho_grid'] for s in C['seeds'] for p in C['policies']]
with ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(v.simulate,tasks,chunksize=8))
with (O/'v29_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows)}))
