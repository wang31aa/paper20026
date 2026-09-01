#!/usr/bin/env python3
import csv,json,math,hashlib,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path.insert(0,str(ROOT/'cross_domain_v27'));import run_v27 as b
rows=[x for x in csv.DictReader((R/'results/v30_development_runs.csv').open()) if x['policy']=='all_coupled']
def f(x):
 d=x['domain'];n=int(x['n']);rho=float(x['rho']);seed=int(x['seed']);_,_,_,_,p,_,_,_=b.frozen(d,n,rho,seed);lg=math.log(n/5)
 if d=='circuit':extra=[max(p['alpha_ratio']),1/min(p['beta_ratio']),min(p['authority_ratio'])]
 else:extra=[max(p['tank_area_ratio']),max(p['outflow_ratio']),min(p['pump_gain_ratio'])]
 return np.array([1,rho,rho*rho,lg,rho*lg,*extra])
models={}
for d in ('circuit','water'):
 z=[x for x in rows if x['domain']==d];X=np.vstack([f(x) for x in z]);y=np.array([int(x['task_success']) for x in z],float);mean=X[:,1:].mean(0);scale=X[:,1:].std(0);scale[scale<1e-9]=1;Xs=np.c_[np.ones(len(X)),(X[:,1:]-mean)/scale];w=np.zeros(Xs.shape[1])
 for _ in range(12000):q=1/(1+np.exp(-np.clip(Xs@w,-30,30)));w-=.015*((Xs.T@(q-y))/len(y)+np.r_[0,.006*w[1:]])
 models[d]={'coefficients':w.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'n':len(y),'positives':int(y.sum())}
o={'schema_version':'30-fit-1','frozen_before_v31':True,'features':['rho','rho2','logN','rho_logN','growth_or_area','inverse_time_or_outflow','authority_min'],'threshold':.5,'models':models,'claim_boundary':'repaired two-domain computational predictor'};raw=json.dumps(o,sort_keys=True,indent=2)+'\n';(R/'V30_FROZEN_MODELS.json').write_text(raw);print(hashlib.sha256(raw.encode()).hexdigest())
