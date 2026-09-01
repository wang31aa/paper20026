#!/usr/bin/env python3
"""Freeze a complete-descriptor kNN approximation of the two-layer kernel."""
import csv,json,math,hashlib,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path[:0]=[str(ROOT/'cross_domain_v27'),str(ROOT/'cross_domain_v28')];import run_v27 as b;import run_v28 as v
files=[ROOT/'cross_domain_v30/results/v30_development_runs.csv',ROOT/'cross_domain_v31/results/v31_runs.csv',ROOT/'cross_domain_v32/results/v32_runs.csv',ROOT/'cross_domain_v33/results/v33_circuit_runs.csv',ROOT/'cross_domain_v33/results/v33_water_runs.csv'];rows=[]
for f in files:rows += [x for x in csv.DictReader(f.open()) if x['policy']=='all_coupled']
def feat(x):
 d=x['domain'];n=int(x['n']);rho=float(x['rho']);seed=int(x['seed']);_,_,_,_,p,noise,bias,state=b.frozen(d,n,rho,seed);mu=v.mode(n,0,rho)[2];z=[rho,rho*rho,math.log(n/5),rho*math.log(n/5),mu,float(np.mean(abs(bias))),float(np.max(abs(bias))),float(np.sqrt(np.mean(noise*noise)))]
 for k,_ in b.PARAMS[d]:a=p[k];z += [float(a.mean()),float(a.std()),float(a.min()),float(a.max())]
 if isinstance(state,tuple):state=state[0]
 z += [float(np.max(abs(state))),float(np.mean(abs(state)))];return z
models={}
for d in ('circuit','water'):
 q=[x for x in rows if x['domain']==d];X=np.asarray([feat(x) for x in q]);y=np.asarray([int(x['task_success']) for x in q]);mean=X.mean(0);scale=X.std(0);scale[scale<1e-9]=1;models[d]={'mean':mean.tolist(),'scale':scale.tolist(),'X':((X-mean)/scale).tolist(),'y':y.tolist(),'k':15,'n':len(y),'positives':int(y.sum())}
o={'schema_version':'34.0','frozen_before_v35':True,'model_class':'complete-descriptor distance-weighted k-nearest-neighbour approximation of the system-specific winning set','models':models,'claim_boundary':'nonparametric computational viability approximation; not a universal scalar law or physical validation'};raw=json.dumps(o,sort_keys=True,separators=(',',':'));(R/'V34_FROZEN_KERNEL.json').write_text(raw+'\n');print(hashlib.sha256((raw+'\n').encode()).hexdigest())
