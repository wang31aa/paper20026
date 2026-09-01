#!/usr/bin/env python3
"""Prospective corrected-observer and genuine-switching computation."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V23_CORRECTED_OBSERVER_SWITCHING_CONTRACT.json').read_text())
sys.path[:0]=[str(ROOT/'cross_domain_v17'),str(ROOT/'cross_domain_v21')]
import run_v17 as release
import run_v21 as motor
plant=release.v11
plant.PARAMETER_MODE='coherent';plant.CORRECTED_DISTRIBUTED_OBSERVER=True;plant.DYNAMIC_SWITCHING=True
plant.SWITCH_PERIOD_STEPS=round(C['switch_period_s']/plant.DT)
plant.C['topologies']=C['topologies'];plant.C['development_seeds']=C['development_seeds'];plant.C['heldout_seeds']=C['heldout_seeds'];plant.C['heldout_faults']=C['fault_rotation']
motor.CORRECTED_DISTRIBUTED_OBSERVER=True;motor.DYNAMIC_SWITCHING=True;motor.SWITCH_PERIOD_STEPS=round(C['switch_period_s']/motor.DT)
OUT=HERE/'results';OUT.mkdir(exist_ok=True)

def spread_and_checks(domain,n,top,rho,seed):
    vf,hf,_=plant.condition(seed);rng,A,H,gamma,het,bias,inn=plant.common(seed,n,top,rho,hf,vf)
    pars=plant.domain_parameters(domain,rng,n,het)
    spread={k:float(np.ptp(x)) for k,x in pars.items()}
    modes=range(math.ceil(plant.STEPS/plant.SWITCH_PERIOD_STEPS)) if top=='switching' else range(1)
    inv=[]; hashes=[]; radii=[]
    for epoch in modes:
        Ae,vis,He,_=plant.graph_data(n,top,vf,epoch)
        Te=plant.observer_transition(n,top,vf,rho,'base',plant.condition(seed)[2],epoch)
        inv.append(float(np.max(np.abs(He@np.ones(n)-vis))))
        hashes.append(hashlib.sha256(Ae.tobytes()).hexdigest())
        radii.append(float(max(abs(np.linalg.eigvals(Te)))))
    return spread,max(inv),max(radii),len(set(hashes)),vf,hf

def execute(task):
    domain,n,top,rho,seed,policy=task
    if domain=='motor':
        mapping={'all_coupled':'all_coupled_pi','global_gain_reduction':'global_gain_reduction','connectivity_gate':'connectivity_gate_pi','two_layer_gate':'two_layer_pi'}
        row=dict(motor.execute((n,top,rho,seed,mapping[policy])))
        # Motor uses all nodes as heterogeneous correlated virtual devices.
        ratios=json.loads(row['parameter_spread']);row['parameter_ratio_spread']=row.pop('parameter_spread')
        row['minimum_parameter_ratio_spread']=row.pop('minimum_parameter_spread')
        row['policy']=policy;spread=ratios
        A0,v0,H0,T0,g0=motor.observer_data(n,top,rho,0);epochs=range(math.ceil(motor.STEPS/motor.SWITCH_PERIOD_STEPS)) if top=='switching' else range(1)
        mats=[motor.observer_data(n,top,rho,e) for e in epochs]
        inv=max(float(np.max(np.abs(H@np.ones(n)-v))) for A,v,H,T,g in mats)
        rad=max(float(max(abs(np.linalg.eigvals(T)))) for A,v,H,T,g in mats)
        distinct=len({hashlib.sha256(A.tobytes()).hexdigest() for A,v,H,T,g in mats});vf=.2;hf=1.
    else:
        row=dict(plant.task(task));spread,inv,rad,distinct,vf,hf=spread_and_checks(domain,n,top,rho,seed)
        row['parameter_spread_si']=json.dumps(spread,sort_keys=True);row['minimum_parameter_spread_si']=min(spread.values())
    row.update(split='development' if seed in C['development_seeds'] else 'heldout',
      observer_identity_error=inv,observer_transition_radius=rad,distinct_adjacencies=distinct,
      actual_switch_count=max(0,distinct-1),target_visible_fraction=vf,heterogeneous_fraction=hf,
      paired_replay_id=f'{domain}|{n}|{top}|{rho}|{seed}',evidence_class=C['evidence_class'])
    return row

def main():
    seeds=C['development_seeds']+C['heldout_seeds'];tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(execute,tasks,chunksize=12))
    fields=sorted({k for row in rows for k in row})
    with (OUT/'v23_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V23_CORRECTED_OBSERVER_SWITCHING_CONTRACT.json').read_bytes()).hexdigest(),'rows':len(rows),'unique_rows':len({(x['paired_replay_id'],x['policy']) for x in rows}),'evidence_class':C['evidence_class']}
    (OUT/'v23_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
