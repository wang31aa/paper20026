#!/usr/bin/env python3
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path[:0]=[str(ROOT/'cross_domain_v30'),str(ROOT/'cross_domain_v36')];import run_v30 as run;import fit_v36 as f
C=json.loads((R/'V39_FROZEN_TEST.json').read_text());mp=ROOT/'cross_domain_v38/V38_FROZEN_MODEL.json';assert hashlib.sha256(mp.read_bytes()).hexdigest()==C['model_sha256'];M=json.loads(mp.read_text());O=R/'results';O.mkdir(exist_ok=True);rows=[]
for d,z in C['design'].items():run.C['domains']=[d];run.C['rho_grid']=z['rho_grid'];run.C['policies']=C['policies'];rows+=run.execute(z['sizes'],C['seeds'],O/f'v39_{d}_runs.csv')
metrics={}
for d,m in M['models'].items():
 obs=[];pred=[]
 for x in [q for q in rows if q['domain']==d and q['policy']=='all_coupled']:
  z=f.expand(f.base(x));s=((z-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['coef']);pred.append(s>=m['threshold']);obs.append(bool(int(x['task_success'])))
 tp=sum(a and q for a,q in zip(pred,obs));tn=sum((not a) and (not q) for a,q in zip(pred,obs));p=sum(obs);n=len(obs)-p;metrics[d]={'n':len(obs),'positives':p,'negatives':n,'sensitivity':tp/p if p else None,'specificity':tn/n if n else None}
a=C['acceptance'];passed=all(x['positives']>=a['minimum_positive_events_per_domain'] and x['negatives']>=a['minimum_negative_events_per_domain'] and x['sensitivity']>=a['minimum_sensitivity'] and x['specificity']>=a['minimum_specificity'] for x in metrics.values());out={'status':'PASS_EXECUTION','metrics':metrics,'all_acceptance_pass':passed,'claim_boundary':C['claim_boundary']};(O/'V39_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
