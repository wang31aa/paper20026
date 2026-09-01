#!/usr/bin/env python3
"""Frozen ninth-domain theorem-class holdout with extremal witnesses."""
from __future__ import annotations
import csv, hashlib, json, math, random
from pathlib import Path
H=Path(__file__).resolve().parent; C=json.loads((H/'V45_FROZEN_CONTRACT.json').read_text());O=H/'results';O.mkdir(exist_ok=True)

def bounds(rho):
    rr=[]; pp=[]
    for a,d0,db,u,v in zip(C['a'],C['D0'],C['Db'],C['U'],C['V']):
        r=max(0.0,d0+rho*db-u)/a; p=r+v*C['h']/rho;rr.append(r);pp.append(p)
    psi=max(max(p/C['epsilon_peak'] for p in pp),max(r/C['epsilon_ultimate'] for r in rr))
    return rr,pp,psi

def trajectory(rho,seed,kind):
    rr,pp,psi=bounds(rho);rng=random.Random(seed);peak=ultimate=0.0
    for a,r,p in zip(C['a'],rr,pp):
        sign=-1 if rng.random()<.5 else 1
        if kind=='adversarial':e=sign*p; target=sign*r
        else:e=sign*rng.uniform(0,p);target=sign*rng.uniform(0,r)
        local_peak=abs(e)
        dt=.002;steps=6000
        tail=[]
        for k in range(steps):
            e += dt*(-a*(e-target));local_peak=max(local_peak,abs(e))
            if k>=steps-500:tail.append(abs(e))
        peak=max(peak,local_peak);ultimate=max(ultimate,max(tail))
    success=int(peak<=C['epsilon_peak']+1e-10 and ultimate<=C['epsilon_ultimate']+1e-4)
    return {'domain':C['domain'],'rho':rho,'seed':seed,'kind':kind,'psi':psi,'predicted_safe':int(psi<=1),
            'observed_safe':success,'peak_error':peak,'ultimate_error':ultimate}

rows=[]
pred=[]
for rho in C['rho_grid']:
    rr,pp,psi=bounds(rho);pred.append({'rho':rho,'psi':psi,'predicted_safe':int(psi<=1),'r_max':max(rr),'p_max':max(pp)})
    rows.append(trajectory(rho,0,'adversarial'))
    rows += [trajectory(rho,s,'random') for s in C['random_seeds']]
for name,data in [('v45_predictions.csv',pred),('v45_runs.csv',rows)]:
    with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,data[0].keys());w.writeheader();w.writerows(data)
fp=sum(r['predicted_safe'] and not r['observed_safe'] for r in rows);fn=sum((not r['predicted_safe']) and r['observed_safe'] for r in rows)
report={'status':'PASS_EXECUTION','contract_sha256':hashlib.sha256((H/'V45_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
        'predicted_safe_rhos':[r['rho'] for r in pred if r['predicted_safe']],
        'trajectories':len(rows),'false_safe':fp,'false_unsafe':fn,
        'adversarial_agreement':all(r['predicted_safe']==r['observed_safe'] for r in rows if r['kind']=='adversarial'),
        'evidence_class':C['claim_boundary']}
(O/'V45_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
