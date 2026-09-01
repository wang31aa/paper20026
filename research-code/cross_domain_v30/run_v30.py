#!/usr/bin/env python3
import csv,json,math,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path[:0]=[str(ROOT/'cross_domain_v27'),str(ROOT/'cross_domain_v28')];import run_v27 as b;import run_v28 as v
C=json.loads((R/'V30_REPAIR_CONTRACT.json').read_text());O=R/'results';O.mkdir(exist_ok=True);DT=v.DT;K=v.K
def water(t):
 d,n,rho,seed,policy=t;_,_,_,_,par,noise,bias,state=b.frozen('water',n,rho,seed);ph=b.digest('water',n,seed,par);p=state.copy();z=np.zeros(n);err=[];margin=[];energy=0.;messages=0
 for k in range(K):
  A,T,mu,ah=v.mode(n,k//v.C['switch_period_steps'],rho);target=.55*math.sin(.18*k*DT);target_dot=.55*.18*math.cos(.18*k*DT);z=target+T@(z-target)
  res=np.abs(bias)+.06*np.abs(p-z);W=b.weights(A,res,policy);messages+=np.count_nonzero(W);coup=W@(p+bias)-W.sum(1)*p;ex=bias if policy=='all_coupled' else bias*(res<.30)
  area,out,auth=par['tank_area_ratio'],.22*par['outflow_ratio'],par['pump_gain_ratio'];ref=1.4+target
  u=out*np.sqrt(np.maximum(ref,0))+area*target_dot+1.05*(ref-p)+.20*rho*coup+1.7*rho*rho*ex+.015*noise[k,:,0]
  u=np.clip(u,0,2.4*auth);p+=DT*(u-out*np.sqrt(np.maximum(p,0)))/area;p=np.maximum(p,0);e=np.max(np.abs(p-ref));mar=min(.55-e,np.min(p)-.25,2.4-np.max(p));energy+=float(np.sum(u*u)*DT);err.append(float(e));margin.append(float(mar))
 q=margin[25:];return {'domain':'water','n':n,'rho':rho,'seed':seed,'policy':policy,'parameter_sha256':ph,'task_success':int(min(q)>=0),'minimum_physical_margin':min(q),'tail_error':float(np.quantile(err[-40:],.95)),'control_energy':energy,'communication_messages':messages,'controller_contract':'equilibrium_feedforward'}
def task(t): return v.simulate(t) if t[0]=='circuit' else water(t)
def execute(sizes,seeds,out):
 tasks=[(d,n,r,s,p) for d in C['domains'] for n in sizes for r in C['rho_grid'] for s in seeds for p in C['policies']]
 with ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(task,tasks,chunksize=8))
 with out.open('w',newline='') as f:w=csv.DictWriter(f,sorted({k for x in rows for k in x}));w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':print(json.dumps({'rows':len(execute(C['development_sizes'],C['development_seeds'],O/'v30_development_runs.csv'))}))
