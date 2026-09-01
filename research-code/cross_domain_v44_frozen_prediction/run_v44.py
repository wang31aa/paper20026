#!/usr/bin/env python3
"""Sequentially freeze response predictions, then execute disjoint holdouts."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V44_STAGE1_CONTRACT.json').read_text())
sys.path[:0]=[str(ROOT/'cross_domain_v25'),str(ROOT/'cross_domain_v28')]
import run_v25 as main3
import run_v28 as ext5
MAIN={'uav6dof','vehicle','motor'}; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

def execute(t):
    domain,n,rho,seed=t
    if domain in MAIN:
        row=dict(main3.execute((domain,n,'certified_switching',rho,seed,C['policy'])))
    else:
        row=dict(ext5.simulate((domain,n,rho,seed,C['policy'])))
    return {k:row.get(k,'') for k in ('domain','n','rho','seed','policy','task_success',
        'minimum_physical_margin','first_failure_time','tail_error','control_energy',
        'communication_messages','parameter_sha256','common_metric_mu')}

def classify(y):
    tol=C['classification_rule']['difference_tolerance']; gap=C['classification_rule']['endpoint_gap']
    if max(y)-min(y)<=tol:return 'plateau'
    d=[b-a for a,b in zip(y,y[1:])]
    if all(x>=-tol for x in d):return 'monotone_improving'
    if all(x<=tol for x in d):return 'monotone_worsening'
    m=max(y); ids=[i for i,v in enumerate(y) if abs(v-m)<=tol]
    if ids and min(ids)>0 and max(ids)<len(y)-1 and m-y[0]>=gap and m-y[-1]>=gap:return 'finite_window'
    return 'irregular_or_unresolved'

def aggregate(rows):
    g=defaultdict(list)
    for r in rows:g[(r['domain'],float(r['rho']))].append(int(r['task_success']))
    out={}
    for d in C['domains']:
        rates=[sum(g[(d,r)])/len(g[(d,r)]) for r in C['rho_grid']]
        safe=[r for r,y in zip(C['rho_grid'],rates) if y>=.5]
        out[d]={'rates':rates,'response_class':classify(rates),
                'predicted_safe_interval':([] if not safe else [min(safe),max(safe)])}
    return out

def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [cen-half,cen+half]

def tasks(split):
    out=[]
    for d in C['domains']:
        ss=C[f'{split}_seeds_main'] if d in MAIN else C[f'{split}_seeds_extension']
        out += [(d,n,r,s) for n in C['sizes'] for r in C['rho_grid'] for s in ss]
    return out

def main():
    with ThreadPoolExecutor(max_workers=8) as ex:dev=list(ex.map(execute,tasks('development')))
    pred=aggregate(dev)
    stage2={'schema_version':'44.1','stage1_sha256':hashlib.sha256((HERE/'V44_STAGE1_CONTRACT.json').read_bytes()).hexdigest(),
            'predictions':pred,'heldout_seed_lists':{'main':C['heldout_seeds_main'],'extension':C['heldout_seeds_extension']},
            'locked_before_heldout':True,'claim_boundary':C['claim_boundary']}
    stage2_path=HERE/'V44_STAGE2_FROZEN_PREDICTIONS.json'
    stage2_path.write_text(json.dumps(stage2,indent=2)+'\n')
    stage2_hash=hashlib.sha256(stage2_path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=8) as ex:held=list(ex.map(execute,tasks('heldout')))
    observed=aggregate(held); detail=[]; total_false_safe=0
    for d in C['domains']:
        interval=pred[d]['predicted_safe_interval']; tp=tn=fp=fn=0
        for rho,rate in zip(C['rho_grid'],observed[d]['rates']):
            predicted=bool(interval and interval[0]<=rho<=interval[1]); actual=rate>=.5
            tp+=predicted and actual;tn+=(not predicted) and (not actual);fp+=predicted and not actual;fn+=(not predicted) and actual
        total_false_safe+=fp
        detail.append({'domain':d,'predicted_class':pred[d]['response_class'],'observed_class':observed[d]['response_class'],
            'class_match':pred[d]['response_class']==observed[d]['response_class'],'predicted_interval':interval,
            'heldout_rates':observed[d]['rates'],'sensitivity':tp/(tp+fn) if tp+fn else None,
            'specificity':tn/(tn+fp) if tn+fp else None,'false_safe_grid_points':fp})
    fields=sorted({k for r in dev+held for k in r}|{'split'})
    with (OUT/'v44_all_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for split,rows in [('development',dev),('heldout',held)]:
            for r in rows:w.writerow(dict(r,split=split))
    report={'status':'PASS_EXECUTION','stage2_sha256':stage2_hash,'development_rows':len(dev),'heldout_rows':len(held),
            'response_class_matches':sum(x['class_match'] for x in detail),'domains':len(detail),
            'false_safe_grid_points':total_false_safe,'detail':detail,'evidence_class':C['claim_boundary']}
    (OUT/'V44_FROZEN_PREDICTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
