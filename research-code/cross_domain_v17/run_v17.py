#!/usr/bin/env python3
"""Run a common grid while preserving each domain's own heterogeneous plant."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, json, sys, types
import numpy as np

# The retained domain simulators use only two small dense SciPy linear-algebra
# routines.  Provide deterministic NumPy equivalents when SciPy is absent so
# the frozen experiment does not depend on an unrecorded package installation.
try:
    import scipy.linalg  # noqa: F401
except ModuleNotFoundError:
    def _expm(a):
        a=np.asarray(a,dtype=float); n=a.shape[0]
        norm=float(np.linalg.norm(a,ord=np.inf))
        scale=max(0,int(np.ceil(np.log2(norm/.5)))) if norm>.5 else 0
        scaled=a/(2**scale); result=np.eye(n); term=np.eye(n)
        for k in range(1,81):
            term=term@scaled/k; result+=term
            if np.linalg.norm(term,ord=np.inf) <= 1e-15*max(1.,np.linalg.norm(result,ord=np.inf)):
                break
        for _ in range(scale): result=result@result
        return result
    def _lyap(a,q):
        a=np.asarray(a,dtype=float); q=np.asarray(q,dtype=float); n=a.shape[0]
        operator=np.kron(np.eye(n),a)+np.kron(a.conj(),np.eye(n))
        return np.linalg.solve(operator,q.reshape(-1,order='F')).reshape((n,n),order='F')
    scipy_module=types.ModuleType('scipy')
    linalg_module=types.ModuleType('scipy.linalg')
    linalg_module.expm=_expm
    linalg_module.solve_continuous_lyapunov=_lyap
    scipy_module.linalg=linalg_module
    sys.modules['scipy']=scipy_module
    sys.modules['scipy.linalg']=linalg_module

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'cross_domain_v11'))
sys.path.insert(0,str(ROOT/'cross_domain_v13'))
import run_v11 as v11
import run_v13 as v13

C=json.loads((HERE/'V17_FROZEN_CONTRACT.json').read_text())
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
v11.C['heldout_seeds']=C['heldout_seeds']
v11.C['heldout_faults']=C['fault_rotation']+['mixed']
v13.C['confirmation_seeds']=C['heldout_seeds']
v13.C['fault_rotation']=C['fault_rotation']

def v11_spread(domain,n,top,seed):
    vf,hf,_=v11.condition(seed)
    rng,A,H,gamma,het,bias,inn=v11.common(seed,n,top,.35,hf,vf)
    if domain=='uav6dof':
        pars={'mass':np.where(het,rng.uniform(.75,1.35,n),1.),'drag':np.where(het,rng.uniform(.10,.28,n),.16),'bandwidth':np.where(het,rng.uniform(2.2,4.8,n),3.8),'reserve':np.where(het,rng.uniform(.72,1.,n),1.)}
    elif domain=='vehicle':
        pars={'lag':np.where(het,rng.uniform(.35,.9,n),.5),'authority':np.where(het,rng.uniform(.65,1.05,n),1.),'drag':np.where(het,rng.uniform(.015,.045,n),.025)}
    else:
        pars={'resistance':np.where(het,rng.uniform(.7,1.5,n),1.),'inductance':np.where(het,rng.uniform(.06,.16,n),.1),'inertia':np.where(het,rng.uniform(.025,.07,n),.04),'friction':np.where(het,rng.uniform(.018,.055,n),.03),'torque_constant':np.where(het,rng.uniform(.75,1.2,n),1.)}
    return {k:float(np.ptp(v)) for k,v in pars.items()}

def execute(task):
    domain,n,top,rho,seed,policy=task
    if domain in ('uav6dof','vehicle','motor'):
        row=v11.task(task); spread=v11_spread(domain,n,top,seed)
    else:
        row=v13.simulate(task); spread=json.loads(row['parameter_spread'])
    row=dict(row)
    row['eta']=float(row.get('eta',row['rho']*row['gamma']))
    row['parameter_spread']=json.dumps(spread,sort_keys=True)
    row['minimum_parameter_spread']=min(spread.values())
    row['endpoint_definition']=C['task_endpoints'][domain]
    row['paired_replay_id']=f"{domain}|{n}|{top}|{rho}|{seed}"
    row['evidence_class']='domain_specific_heterogeneous_computational_intervention'
    return row

def main():
    tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes']
           for t in C['topologies'] for r in C['rho_grid']
           for s in C['heldout_seeds'] for p in C['policies']]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(execute,tasks,chunksize=20))
    fields=sorted({k for row in rows for k in row})
    path=OUT/'v17_runs.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V17_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
              'rows':len(rows),'unique_rows':len({(x['paired_replay_id'],x['policy']) for x in rows}),
              'domains':C['domains'],'claim_boundary':C['claim_boundary']}
    (OUT/'v17_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
