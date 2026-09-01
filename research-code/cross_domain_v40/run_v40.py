#!/usr/bin/env python3
"""Prospective eight-domain test of a frozen approximate viability supervisor."""
from __future__ import annotations
import csv, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
CONTRACT_PATH=HERE/'V40_FROZEN_SUPERVISOR_CONTRACT.json'; OUTPUT_PREFIX='v40'
C=json.loads(CONTRACT_PATH.read_text())
sys.path[:0]=[str(ROOT/'cross_domain_v25'),str(ROOT/'cross_domain_v27')]
import run_v25 as main3
import run_v27 as ext5
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
MAIN=set(['uav6dof','vehicle','motor'])

def run_one(t):
    d,n,rho,seed,p=t
    if d in MAIN:
        row=dict(main3.execute((d,n,'certified_switching',rho,seed,p)))
    else:
        row=dict(ext5.simulate(d,n,rho,seed,p))
    row['split']='development' if seed in (C['development_seeds_main'] if d in MAIN else C['development_seeds_extension']) else 'heldout'
    row['v40_pair_id']=f'{d}|{n}|{rho}|{seed}'
    return row

def choose(dev):
    policies=C['candidate_policies']; energies=[float(r['control_energy']) for r in dev]; messages=[float(r['communication_messages']) for r in dev]
    em=max(energies) or 1.; mm=max(messages) or 1.; scored=[]
    for rank,p in enumerate(policies):
        rr=[r for r in dev if r['policy']==p]
        score=(sum(int(r['task_success']) for r in rr),float(np.quantile([float(r['minimum_physical_margin']) for r in rr],.10)),-float(np.mean([float(r['control_energy'])/em+float(r['communication_messages'])/mm for r in rr])),-rank)
        scored.append((score,p))
    return max(scored)[1]

def main():
    tasks=[]
    for d in C['domains']:
        seeds=(C['development_seeds_main']+C['heldout_seeds_main']) if d in MAIN else (C['development_seeds_extension']+C['heldout_seeds_extension'])
        for n in C['sizes']:
            for rho in C['rho_grid']:
                for seed in seeds:
                    for p in C['candidate_policies']: tasks.append((d,n,rho,seed,p))
    with ThreadPoolExecutor(max_workers=8) as ex: rows=list(ex.map(run_one,tasks))
    selected={}
    for d in C['domains']:
        for n in C['sizes']:
            for rho in C['rho_grid']:
                dev=[r for r in rows if r['domain']==d and int(r['n'])==n and float(r['rho'])==rho and r['split']=='development']
                selected[(d,n,rho)]=choose(dev)
    for r in rows:
        r['frozen_supervisor_policy']=selected[(r['domain'],int(r['n']),float(r['rho']))]
        r['selected_by_supervisor']=int(r['policy']==r['frozen_supervisor_policy'])
    fields=sorted({k for r in rows for k in r})
    with (OUT/f'{OUTPUT_PREFIX}_all_policy_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    choices=[{'domain':d,'n':n,'rho':rho,'policy':p} for (d,n,rho),p in sorted(selected.items())]
    with (OUT/f'{OUTPUT_PREFIX}_frozen_choices.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,choices[0].keys());w.writeheader();w.writerows(choices)
    held=[r for r in rows if r['split']=='heldout']; sel=[r for r in held if r['selected_by_supervisor']]
    summary=[]
    for d in C['domains']:
        for label,rr in [('supervisor',[r for r in sel if r['domain']==d])]+[(p,[r for r in held if r['domain']==d and r['policy']==p]) for p in C['candidate_policies']]:
            summary.append({'domain':d,'policy':label,'n':len(rr),'success_rate':sum(int(r['task_success']) for r in rr)/len(rr),'median_margin':float(np.median([float(r['minimum_physical_margin']) for r in rr])),'median_energy':float(np.median([float(r['control_energy']) for r in rr])),'median_messages':float(np.median([float(r['communication_messages']) for r in rr]))})
    with (OUT/f'{OUTPUT_PREFIX}_heldout_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,summary[0].keys());w.writeheader();w.writerows(summary)
    manifest={'contract_sha256':hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),'rows':len(rows),'heldout_selected_rows':len(sel),'claim_boundary':C['claim_boundary']}
    (OUT/f'{OUTPUT_PREFIX}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
