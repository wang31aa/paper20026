#!/usr/bin/env python3
import json,sys,hashlib,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path[:0]=[str(ROOT/'cross_domain_v30'),str(ROOT/'cross_domain_v27')];import run_v30 as run;import run_v27 as b
C=json.loads((R/'V32_TRANSITION_TEST.json').read_text());M=json.loads((ROOT/'cross_domain_v30/V30_FROZEN_MODELS.json').read_text());model=ROOT/'cross_domain_v30/V30_FROZEN_MODELS.json';assert hashlib.sha256(model.read_bytes()).hexdigest()==C['unchanged_model_sha256'];O=R/'results';O.mkdir(exist_ok=True);run.C['rho_grid']=C['rho_grid'];run.C['policies']=C['policies']
rows=run.execute(C['sizes'],C['seeds'],O/'v32_runs.csv')
def feat(x):
 d=x['domain'];n=int(x['n']);rho=float(x['rho']);seed=int(x['seed']);_,_,_,_,p,_,_,_=b.frozen(d,n,rho,seed);lg=math.log(n/5);extra=[max(p['alpha_ratio']),1/min(p['beta_ratio']),min(p['authority_ratio'])] if d=='circuit' else [max(p['tank_area_ratio']),max(p['outflow_ratio']),min(p['pump_gain_ratio'])];return np.array([rho,rho*rho,lg,rho*lg,*extra])
metrics={}
for d,m in M['models'].items():
 z=[x for x in rows if x['domain']==d and x['policy']=='all_coupled'];obs=[];pred=[]
 for x in z:
  xx=(feat(x)-np.array(m['mean']))/np.array(m['scale']);pred.append(float(np.dot(np.array(m['coefficients']),np.r_[1,xx]))>=0);obs.append(bool(int(x['task_success'])))
 tp=sum(a and y for a,y in zip(pred,obs));tn=sum((not a) and (not y) for a,y in zip(pred,obs));p=sum(obs);q=len(obs)-p;metrics[d]={'n':len(obs),'positives':p,'negatives':q,'sensitivity':tp/p if p else None,'specificity':tn/q if q else None}
a=C['acceptance'];passed=all(x['positives']>=a['minimum_positive_events_per_domain'] and x['negatives']>=a['minimum_negative_events_per_domain'] and x['sensitivity']>=a['minimum_sensitivity'] and x['specificity']>=a['minimum_specificity'] for x in metrics.values());out={'status':'PASS_EXECUTION','metrics':metrics,'all_acceptance_pass':passed,'claim_boundary':C['claim_boundary']};(O/'V32_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
