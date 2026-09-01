#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, json, math, sys
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'cross_domain_v17')); import run_v17 as core
C=json.loads((HERE/'V19_FROZEN_CONTRACT.json').read_text()); P=json.loads((HERE/'V19_FROZEN_PREDICTOR.json').read_text())
assert P['contract_sha256']==hashlib.sha256((HERE/'V19_FROZEN_CONTRACT.json').read_bytes()).hexdigest()
core.v11.C['topologies']=C['heldout_topologies']; core.v13.C['topologies']=C['heldout_topologies']
core.v11.C['heldout_seeds']=C['heldout_seeds']; core.v11.C['heldout_faults']=['delay','packet_loss','sensor_bias','actuator_loss']
core.v13.C['confirmation_seeds']=C['heldout_seeds']; core.v13.C['fault_rotation']=['delay','packet_loss','sensor_bias','actuator_loss']

def vector(r):
    tops=['chain','ring','random_directed','clustered','switching']; faults=['development','delay','packet_loss','mixed','sensor_bias','actuator_loss']
    base=[math.log(float(r['n'])),float(r['rho']),float(r['rho'])**2,float(r['gamma']),float(r['visible_fraction']),float(r['heterogeneous_fraction']),float(r.get('minimum_parameter_spread') or 0.)]
    return np.asarray(base+[float(r['policy']==x) for x in C['policies'][1:]]+[float(r['topology']==x) for x in tops[1:]]+[float(r['fault']==x) for x in faults[1:]])

def predict(r):
    m=P['models'][r['domain']]; z=(vector(r)-np.asarray(m['mean']))/np.asarray(m['scale']); X=np.asarray(m['x']); y=np.asarray(m['y'])
    dist=np.linalg.norm(X-z,axis=1); idx=np.argsort(dist)[:C['prediction']['k']]; weights=1/(dist[idx]+1e-6)
    prob=float(np.dot(weights,y[idx])/weights.sum()); return prob,int(prob>=C['prediction']['decision_threshold'])

tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['heldout_sizes'] for t in C['heldout_topologies'] for r in C['heldout_rho'] for s in C['heldout_seeds'] for p in C['policies']]
with ThreadPoolExecutor(max_workers=16) as pool: rows=list(pool.map(core.execute,tasks,chunksize=32))
for row in rows: row['predicted_probability'],row['predicted_success']=predict(row)
out=HERE/'results';out.mkdir(exist_ok=True);fields=sorted({k for r in rows for k in r})
with (out/'v19_heldout_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
(out/'v19_manifest.json').write_text(json.dumps({'rows':len(rows),'unique_rows':len({(r['paired_replay_id'],r['policy']) for r in rows}),'contract_sha256':P['contract_sha256'],'predictor_sha256':hashlib.sha256((HERE/'V19_FROZEN_PREDICTOR.json').read_bytes()).hexdigest(),'claim_boundary':C['claim_boundary']},indent=2)+'\n')
print(json.dumps({'rows':len(rows),'unique_rows':len(rows)}))
