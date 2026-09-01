#!/usr/bin/env python3
"""Prospective V20 computation using the physically coherent parameter mode."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, sys

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'cross_domain_v17'))
import run_v17 as core

C=json.loads((HERE/'V20_FROZEN_CONTRACT.json').read_text())
core.v11.PARAMETER_MODE='coherent'
core.v11.C['topologies']=C['topologies']
core.v11.C['development_seeds']=C['development_seeds']
core.v11.C['heldout_seeds']=C['heldout_seeds']
core.v11.C['heldout_faults']=C['fault_rotation']
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

def execute(task):
    domain,n,top,rho,seed,policy=task
    row=dict(core.v11.task(task))
    vf,hf,_=core.v11.condition(seed)
    rng,A,H,gamma,het,bias,inn=core.v11.common(seed,n,top,rho,hf,vf)
    pars=core.v11.domain_parameters(domain,rng,n,het)
    spread={k:float(v.max()-v.min()) for k,v in pars.items()}
    row.update(parameter_spread=json.dumps(spread,sort_keys=True),minimum_parameter_spread=min(spread.values()),
               paired_replay_id=f'{domain}|{n}|{top}|{rho}|{seed}',
               parameter_family='correlated_physically_coherent_v1',
               evidence_class='prospective_heterogeneous_computational_stress_test')
    return row

def main():
    seeds=C['development_seeds']+C['heldout_seeds']
    tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    with ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(execute,tasks,chunksize=24))
    fields=sorted({k for x in rows for k in x})
    with (OUT/'v20_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V20_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
              'parameter_registry_sha256':hashlib.sha256((ROOT/'parameter_qualification/PHYSICAL_PARAMETER_REGISTRY.json').read_bytes()).hexdigest(),
              'rows':len(rows),'unique_rows':len({(x['paired_replay_id'],x['policy']) for x in rows}),
              'claim_boundary':C['claim_boundary']}
    (OUT/'v20_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
