#!/usr/bin/env python3
"""Freeze domain-specific theory-structured critical predictors before V29."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path.insert(0,str(ROOT/'cross_domain_v27'));import run_v27 as b
rows=[x for x in csv.DictReader((R/'results/v28_runs.csv').open()) if x['policy']=='all_coupled' and int(x['seed']) in (8101,8102,8103)]
def features(x):
 d=x['domain'];n=int(x['n']);rho=float(x['rho']);seed=int(x['seed']);_,_,_,_,v,_,_,_=b.frozen(d,n,rho,seed)
 spread=float(np.mean([np.ptp(v[k]) for k,_ in b.PARAMS[d]]));mu=float(x['common_metric_mu'])
 return np.array([1.,1/(rho*mu),rho*spread,rho*rho*spread,math.log(n/5)])
models={}
for d in sorted({x['domain'] for x in rows}):
 z=[x for x in rows if x['domain']==d];X=np.vstack([features(x) for x in z]);y=np.array([int(x['task_success']) for x in z],float)
 mean=X[:,1:].mean(0);scale=X[:,1:].std(0);scale[scale<1e-9]=1;Xs=np.c_[np.ones(len(X)),(X[:,1:]-mean)/scale]
 if len(set(y))==1:w=np.array([20. if y[0] else -20.,0,0,0,0])
 else:
  w=np.zeros(5)
  for _ in range(8000):
   p=1/(1+np.exp(-np.clip(Xs@w,-30,30)));w-=.02*((Xs.T@(p-y))/len(y)+np.r_[0,.01*w[1:]])
 models[d]={'coefficients':w.tolist(),'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),'development_n':len(y),'development_positive':int(y.sum())}
payload={'schema_version':'28-critical-1','frozen_before_v29':True,'features':['intercept','1/(rho*mu)','rho*mean_spread','rho^2*mean_spread','log(N/5)'],'development_seeds':[8101,8102,8103],'decision_threshold':0.5,'models':models,'claim_boundary':'domain-specific computational predictor; not exact general-network capability law'}
raw=json.dumps(payload,sort_keys=True,indent=2)+'\n';(R/'V28_FROZEN_CRITICAL_MODELS.json').write_text(raw);print(hashlib.sha256(raw.encode()).hexdigest())
