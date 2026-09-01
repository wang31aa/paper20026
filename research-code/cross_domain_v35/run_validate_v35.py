#!/usr/bin/env python3
import json,sys,hashlib,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path[:0]=[str(ROOT/'cross_domain_v30'),str(ROOT/'cross_domain_v27'),str(ROOT/'cross_domain_v28')];import run_v30 as run;import run_v27 as b;import run_v28 as v
C=json.loads((R/'V35_KERNEL_TEST.json').read_text());mp=ROOT/'cross_domain_v34/V34_FROZEN_KERNEL.json';assert hashlib.sha256(mp.read_bytes()).hexdigest()==C['model_sha256'];M=json.loads(mp.read_text());O=R/'results';O.mkdir(exist_ok=True);rows=[]
for d,design in C['design'].items():run.C['domains']=[d];run.C['rho_grid']=design['rho_grid'];run.C['policies']=C['policies'];rows+=run.execute(design['sizes'],C['seeds'],O/f'v35_{d}_runs.csv')
def feat(x):
 d=x['domain'];n=int(x['n']);rho=float(x['rho']);seed=int(x['seed']);_,_,_,_,p,noise,bias,state=b.frozen(d,n,rho,seed);mu=v.mode(n,0,rho)[2];z=[rho,rho*rho,math.log(n/5),rho*math.log(n/5),mu,float(np.mean(abs(bias))),float(np.max(abs(bias))),float(np.sqrt(np.mean(noise*noise)))]
 for k,_ in b.PARAMS[d]:a=p[k];z += [float(a.mean()),float(a.std()),float(a.min()),float(a.max())]
 if isinstance(state,tuple):state=state[0]
 z += [float(np.max(abs(state))),float(np.mean(abs(state)))];return np.asarray(z)
metrics={}
for d,m in M['models'].items():
 train=np.asarray(m['X']);y=np.asarray(m['y']);obs=[];pred=[]
 for x in [q for q in rows if q['domain']==d and q['policy']=='all_coupled']:
  z=(feat(x)-np.asarray(m['mean']))/np.asarray(m['scale']);dist=np.linalg.norm(train-z,axis=1);idx=np.argsort(dist)[:m['k']];w=1/(dist[idx]+1e-6);pred.append(float(np.sum(w*y[idx])/sum(w))>=.5);obs.append(bool(int(x['task_success'])))
 tp=sum(a and q for a,q in zip(pred,obs));tn=sum((not a) and (not q) for a,q in zip(pred,obs));p=sum(obs);n=len(obs)-p;metrics[d]={'n':len(obs),'positives':p,'negatives':n,'sensitivity':tp/p if p else None,'specificity':tn/n if n else None}
a=C['acceptance'];passed=all(x['positives']>=a['minimum_positive_events_per_domain'] and x['negatives']>=a['minimum_negative_events_per_domain'] and x['sensitivity']>=a['minimum_sensitivity'] and x['specificity']>=a['minimum_specificity'] for x in metrics.values());out={'status':'PASS_EXECUTION','metrics':metrics,'all_acceptance_pass':passed,'claim_boundary':C['claim_boundary']};(O/'V35_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
