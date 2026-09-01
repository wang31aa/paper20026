#!/usr/bin/env python3
"""Frozen fair-baseline extension of V24."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
C=json.loads((HERE/'V25_FAIR_BASELINE_CONTRACT.json').read_text())
sys.path[:0]=[str(ROOT/'cross_domain_v17'),str(ROOT/'cross_domain_v21')]
import run_v17 as release
import run_v21 as motor
plant=release.v11
plant.PARAMETER_MODE='coherent';plant.CORRECTED_DISTRIBUTED_OBSERVER=True;plant.DYNAMIC_SWITCHING=True;plant.CORRECTED_INDEPENDENT_TRACKING=True
plant.SWITCH_PERIOD_STEPS=round(2.4/plant.DT);plant.C['topologies']=C['topologies'];plant.C['development_seeds']=C['development_seeds'];plant.C['heldout_seeds']=C['heldout_seeds'];plant.C['heldout_faults']=C['fault_rotation']
motor.CORRECTED_DISTRIBUTED_OBSERVER=True;motor.DYNAMIC_SWITCHING=True;motor.SWITCH_PERIOD_STEPS=round(2.4/motor.DT);motor.C['topologies']=C['topologies'];motor.C['development_seeds']=C['development_seeds'];motor.C['heldout_seeds']=C['heldout_seeds']
OUT=HERE/'results';OUT.mkdir(exist_ok=True)

def graph_checks(domain,n,top,rho,seed):
    if domain=='motor':
        epochs=range(math.ceil(motor.STEPS/motor.SWITCH_PERIOD_STEPS)) if top=='certified_switching' else range(1)
        mats=[motor.observer_data(n,top,rho,e) for e in epochs]
        return min(float(np.linalg.eigvalsh(H+H.T).min()) for A,v,H,T,g in mats),len({hashlib.sha256(A.tobytes()).hexdigest() for A,v,H,T,g in mats})
    vf,hf,fault=plant.condition(seed);epochs=range(math.ceil(plant.STEPS/plant.SWITCH_PERIOD_STEPS)) if top=='certified_switching' else range(1)
    mats=[plant.graph_data(n,top,vf,e)[:3] for e in epochs]
    return min(float(np.linalg.eigvalsh(H+H.T).min()) for A,v,H in mats),len({hashlib.sha256(A.tobytes()).hexdigest() for A,v,H in mats})

def execute(task):
    domain,n,top,rho,seed,policy=task
    if domain=='motor':
        mapping={'all_coupled':'all_coupled_pi','global_gain_reduction':'global_gain_reduction','independent_tracking':'decentralized_pi','connectivity_gate':'connectivity_gate_pi','physical_filter':'current_limited_pi','two_layer_gate':'two_layer_pi'}
        row=dict(motor.execute((n,top,rho,seed,mapping[policy])));row['parameter_ratio_spread']=row.pop('parameter_spread');row['minimum_parameter_ratio_spread']=row.pop('minimum_parameter_spread');row['policy']=policy
    else:
        row=dict(plant.task(task));vf,hf,_=plant.condition(seed);rng,A,H,gamma,het,bias,inn=plant.common(seed,n,top,rho,hf,vf);pars=plant.domain_parameters(domain,rng,n,het);spread={k:float(np.ptp(x)) for k,x in pars.items()};row['parameter_spread_model_units']=json.dumps(spread,sort_keys=True);row['minimum_parameter_spread_model_units']=min(spread.values())
    mu,distinct=graph_checks(domain,n,top,rho,seed)
    row.update(split='development' if seed in C['development_seeds'] else 'heldout',common_metric_mu=mu,distinct_adjacencies=distinct,paired_replay_id=f'{domain}|{n}|{top}|{rho}|{seed}',evidence_class=C['evidence_class'])
    return row

def main():
    seeds=C['development_seeds']+C['heldout_seeds'];tasks=[(d,n,t,r,s,p) for d in C['domains'] for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    try:
        with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(execute,tasks,chunksize=12))
    except PermissionError:
        with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(execute,tasks))
    fields=sorted({k for row in rows for k in row})
    with (OUT/'v25_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V25_FAIR_BASELINE_CONTRACT.json').read_bytes()).hexdigest(),'rows':len(rows),'unique_rows':len({(r['paired_replay_id'],r['policy']) for r in rows}),'evidence_class':C['evidence_class']}
    (OUT/'v25_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
