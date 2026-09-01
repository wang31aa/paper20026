#!/usr/bin/env python3
"""Traceable re-execution of V25 with complete node-level parameter ledgers."""
from __future__ import annotations
import csv, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V26_PARAMETER_TRACE_CONTRACT.json').read_text())
sys.path[:0]=[str(ROOT/'cross_domain_v25')]
import run_v25 as parent
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

UNITS={
 'uav6dof':{'mass':'kg model scale','drag':'s^-1','bandwidth':'s^-1','reserve':'dimensionless'},
 'vehicle':{'tau':'s','authority':'dimensionless','drag':'s^-1'},
 'motor':{'resistance_ratio':'dimensionless','inductance_ratio':'dimensionless','inertia_ratio':'dimensionless','friction_ratio':'dimensionless','torque_constant_ratio':'dimensionless'}}
EVIDENCE={'uav6dof':'LITERATURE_CONSTRAINED_AND_AUTHOR_DECLARED','vehicle':'RECORD_CONSTRAINED_AND_AUTHOR_DECLARED','motor':'AUTHOR_DECLARED_NORMALIZED'}

def parameters(domain,n,top,rho,seed):
    if domain=='motor':
        rng=np.random.default_rng(seed+113*n+19*parent.motor.C['topologies'].index(top))
        values=parent.motor.sample_motor(rng,n)
    else:
        vf,hf,_=parent.plant.condition(seed)
        rng,A,H,gamma,het,bias,inn=parent.plant.common(seed,n,top,rho,hf,vf)
        values=parent.plant.domain_parameters(domain,rng,n,het)
    payload={k:np.asarray(v,dtype=float).tolist() for k,v in sorted(values.items())}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    return values,hashlib.sha256(raw).hexdigest()

def execute(task):
    row=dict(parent.execute(task));domain,n,top,rho,seed,policy=task
    values,h=parameters(domain,n,top,rho,seed)
    row['parameter_sha256']=h
    row['parameter_unit_class']='NORMALIZED_RATIO_MODEL_NOT_SI' if domain=='motor' else 'DOMAIN_MODEL_UNITS'
    row['parameter_evidence_class']=EVIDENCE[domain]
    row['trace_contract_version']='26.0'
    return row

def main():
    seeds=C['development_seeds']+C['heldout_seeds']
    tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    try:
        with ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(execute,tasks,chunksize=12))
    except PermissionError:
        with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(execute,tasks))
    fields=sorted({k for r in rows for k in r})
    with (OUT/'v26_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

    ledger=[]
    for d in C['domains']:
      for n in C['sizes']:
       for t in C['topologies']:
        for rho in C['rho_grid']:
         for seed in seeds:
          values,h=parameters(d,n,t,rho,seed)
          block=f'{d}|{n}|{t}|{rho}|{seed}'
          for name,array in sorted(values.items()):
           key={'R':'resistance_ratio','L':'inductance_ratio','J':'inertia_ratio','B':'friction_ratio','Kt':'torque_constant_ratio'}.get(name,name)
           for node,value in enumerate(array):
            ledger.append({'paired_replay_id':block,'domain':d,'n':n,'topology':t,'rho':rho,'seed':seed,'node':node,'parameter':key,'value':float(value),'unit':UNITS[d][key],'evidence_class':EVIDENCE[d],'parameter_sha256':h})
    with (OUT/'v26_node_parameters.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,ledger[0]);w.writeheader();w.writerows(ledger)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V26_PARAMETER_TRACE_CONTRACT.json').read_bytes()).hexdigest(),'rows':len(rows),'parameter_rows':len(ledger),'unique_trajectory_keys':len({(r['paired_replay_id'],r['policy']) for r in rows}),'claim_boundary':C['claim_boundary']}
    (OUT/'v26_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
