#!/usr/bin/env python3
"""Two-stage development/frozen-heldout heterogeneous motor regime test."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,json,sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'cross_domain_v21')); import run_v21 as core
C=json.load(open(HERE/'V22_REGIME_FROZEN_CONTRACT.json'))

def classify(vals):
    if max(vals)<.2:return 'empty'
    best=max(range(len(vals)),key=lambda i:vals[i])
    if best not in (0,len(vals)-1) and vals[best]>=vals[0]+.05 and vals[best]>=vals[-1]+.05:return 'window'
    if all(b-a>=-.05 for a,b in zip(vals,vals[1:])):return 'monotone_non_decreasing'
    return 'mixed'

def one(task):
    regime,n,t,r,s,exposure,authority=task
    row=core.execute((n,t,r,s,C['policy'],exposure,authority)); row['regime']=regime; return row

def run(split):
    seeds=C[f'{split}_seeds']; tasks=[]
    for name,q in C['regimes'].items():
      for n in C['sizes']:
       for t in C['topologies']:
        for r in C['rho_grid']:
         for s in seeds: tasks.append((name,n,t,r,s,q['exposure_scale'],q['authority_scale']))
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,tasks,chunksize=18))
    fields=sorted({k for x in rows for k in x})
    with (OUT/f'v22_{split}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    return rows

def predictions(rows):
    pred={}
    for name in C['regimes']:
        vals=[]
        for rho in C['rho_grid']:
            x=[r for r in rows if r['regime']==name and r['rho']==rho]
            vals.append(sum(r['task_success'] for r in x)/len(x))
        pred[name]={'development_curve':vals,'predicted_class':classify(vals)}
    payload={'contract_sha256':hashlib.sha256((HERE/'V22_REGIME_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),'predictions':pred}
    (OUT/'V22_FROZEN_PREDICTIONS.json').write_text(json.dumps(payload,indent=2)+'\n');return payload

def main():
    dev=run('development'); p=predictions(dev)
    # The prediction file is written and hashed before held-out trajectories exist.
    p['prediction_sha256']=hashlib.sha256((OUT/'V22_FROZEN_PREDICTIONS.json').read_bytes()).hexdigest()
    (OUT/'V22_PREDICTION_HASH.json').write_text(json.dumps({'prediction_sha256':p['prediction_sha256']},indent=2)+'\n')
    held=run('heldout')
    result={}
    for name in C['regimes']:
        vals=[]
        for rho in C['rho_grid']:
            x=[r for r in held if r['regime']==name and r['rho']==rho]; vals.append(sum(r['task_success'] for r in x)/len(x))
        result[name]={'heldout_curve':vals,'observed_class':classify(vals),'predicted_class':p['predictions'][name]['predicted_class'],'class_match':classify(vals)==p['predictions'][name]['predicted_class']}
    out={'rows':len(dev)+len(held),'development_rows':len(dev),'heldout_rows':len(held),'results':result,'evidence_boundary':C['evidence_boundary']}
    (OUT/'v22_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
