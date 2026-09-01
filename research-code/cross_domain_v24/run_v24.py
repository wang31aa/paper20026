#!/usr/bin/env python3
"""Frozen V24 common-metric directed-switching computation."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V24_COMMON_METRIC_SWITCHING_CONTRACT.json').read_text())
sys.path[:0]=[str(ROOT/'cross_domain_v17'),str(ROOT/'cross_domain_v21')]
import run_v17 as release
import run_v21 as motor
plant=release.v11
plant.PARAMETER_MODE='coherent'; plant.CORRECTED_DISTRIBUTED_OBSERVER=True; plant.DYNAMIC_SWITCHING=True
plant.SWITCH_PERIOD_STEPS=round(C['switch_period_s']/plant.DT)
plant.C['topologies']=C['topologies']; plant.C['development_seeds']=C['development_seeds']; plant.C['heldout_seeds']=C['heldout_seeds']; plant.C['heldout_faults']=C['fault_rotation']
motor.CORRECTED_DISTRIBUTED_OBSERVER=True; motor.DYNAMIC_SWITCHING=True; motor.SWITCH_PERIOD_STEPS=round(C['switch_period_s']/motor.DT)
motor.C['topologies']=C['topologies']; motor.C['development_seeds']=C['development_seeds']; motor.C['heldout_seeds']=C['heldout_seeds']
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

def matrix_checks(mats):
    ones=np.ones(len(mats[0][0])); inv=[]; radii=[]; mu=[]; asym=[]; hashes=[]
    for A,vis,H,T in mats:
        inv.append(float(np.max(np.abs(H@ones-vis))))
        radii.append(float(max(abs(np.linalg.eigvals(T)))))
        mu.append(float(np.min(np.linalg.eigvalsh(H+H.T))))
        asym.append(float(np.linalg.norm(A-A.T,ord='fro')))
        hashes.append(hashlib.sha256(A.tobytes()).hexdigest())
    return max(inv),max(radii),min(mu),min(asym),len(set(hashes))

def plant_checks(domain,n,top,rho,seed):
    vf,hf,_=plant.condition(seed)
    rng,A0,H0,gamma0,het,bias,innovations=plant.common(seed,n,top,rho,hf,vf)
    # Replay the exact RNG state and heterogeneous-node mask used by
    # ``plant.task``.  This makes the recorded spread a property of the
    # parameters that entered the state equation, not a second draw.
    pars=plant.domain_parameters(domain,rng,n,het)
    spread={k:float(np.ptp(x)) for k,x in pars.items()}
    epochs=range(math.ceil(plant.STEPS/plant.SWITCH_PERIOD_STEPS)) if top=='certified_switching' else range(1)
    mats=[]
    for epoch in epochs:
        A,vis,H,_=plant.graph_data(n,top,vf,epoch)
        T=plant.observer_transition(n,top,vf,rho,'base',plant.condition(seed)[2],epoch)
        mats.append((A,vis,H,T))
    return spread,matrix_checks(mats),vf,hf

def execute(task):
    domain,n,top,rho,seed,policy=task
    if domain=='motor':
        mapping={'all_coupled':'all_coupled_pi','global_gain_reduction':'global_gain_reduction','connectivity_gate':'connectivity_gate_pi','two_layer_gate':'two_layer_pi'}
        row=dict(motor.execute((n,top,rho,seed,mapping[policy])))
        row['parameter_ratio_spread']=row.pop('parameter_spread'); row['minimum_parameter_ratio_spread']=row.pop('minimum_parameter_spread'); row['policy']=policy
        epochs=range(math.ceil(motor.STEPS/motor.SWITCH_PERIOD_STEPS)) if top=='certified_switching' else range(1)
        mats=[]
        for epoch in epochs:
            A,vis,H,T,_=motor.observer_data(n,top,rho,epoch); mats.append((A,vis,H,T))
        inv,rad,mu,asym,distinct=matrix_checks(mats); vf=.2; hf=1.
    else:
        row=dict(plant.task(task)); spread,(inv,rad,mu,asym,distinct),vf,hf=plant_checks(domain,n,top,rho,seed)
        row['parameter_spread_model_units']=json.dumps(spread,sort_keys=True); row['minimum_parameter_spread_model_units']=min(spread.values())
    row.update(split='development' if seed in C['development_seeds'] else 'heldout',observer_identity_error=inv,
      observer_transition_radius=rad,common_metric_mu=mu,directed_asymmetry=asym,distinct_adjacencies=distinct,
      actual_switch_count=max(0,distinct-1),target_visible_fraction=vf,heterogeneous_fraction=hf,
      paired_replay_id=f'{domain}|{n}|{top}|{rho}|{seed}',evidence_class=C['evidence_class'])
    return row

def main():
    seeds=C['development_seeds']+C['heldout_seeds']
    tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    try:
        with ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(execute,tasks,chunksize=12))
    except PermissionError:
        # Some sandboxed runners disallow POSIX semaphore discovery.  The
        # deterministic computation is unchanged under the thread fallback.
        with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(execute,tasks))
    fields=sorted({k for row in rows for k in row})
    with (OUT/'v24_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V24_COMMON_METRIC_SWITCHING_CONTRACT.json').read_bytes()).hexdigest(),
      'rows':len(rows),'unique_rows':len({(x['paired_replay_id'],x['policy']) for x in rows}),'evidence_class':C['evidence_class']}
    (OUT/'v24_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
