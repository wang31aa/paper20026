#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
R=Path(__file__).resolve().parent;C=json.loads((R/'V10_FROZEN_CONTRACT.json').read_text());O=R/'results';O.mkdir(exist_ok=True)

def graph(n,k,rng):
 a=np.zeros((n,n));
 for i in range(1,n):a[i,i-1]=1
 if k=='ring':a[0,-1]=1
 elif k=='clustered':
  b=max(2,n//4)
  for i in range(n):
   for j in range(n):
    if i!=j and i//b==j//b:a[i,j]=1/max(b-1,1)
  for i in range(b,n,b):a[i,i-1]=1
 elif k=='random_directed':
  for i in range(n):
   for j in range(n):
    if i!=j and rng.random()<min(.18,3/n):a[i,j]=1
  for i in range(1,n):a[i,i-1]=1
 s=a.sum(1);a[s>0]/=s[s>0,None];return a
@lru_cache(maxsize=None)
def fixed(n,k):
 rng=np.random.default_rng(C['topology_seed']+31*n+list(C['topologies']).index(k));a=graph(n,k,rng);vis=np.zeros(n);vis[:max(1,int(np.ceil(.2*n)))]=1
 h=np.eye(n)-a+np.diag(vis)
 # H is positive stable for the frozen target-rooted graphs.  The Lyapunov
 # equation H^T P+PH=I certifies I >= 2 gamma P with the largest valid
 # scalar returned by this construction.  This is the metric used by TN1;
 # a spectral abscissa alone would not certify contraction for nonnormal H.
 p=solve_continuous_lyapunov(h.T,np.eye(n))
 p=(p+p.T)/2
 residual=h.T@p+p@h-np.eye(n)
 if np.linalg.norm(residual,ord=2)>1e-7 or np.min(np.linalg.eigvalsh(p))<=0:
  raise RuntimeError(f'invalid Lyapunov certificate: n={n}, topology={k}')
 gamma=1/(2*float(np.max(np.linalg.eigvalsh(p))))
 if gamma<=0:raise RuntimeError(f'uncertified graph: n={n}, topology={k}')
 return a,vis,h,gamma
@lru_cache(maxsize=None)
def transition_for(n,k,eta,dt):
 _,_,h,gamma=fixed(n,k)
 return expm(-(float(eta)/gamma)*h*dt)
def trial(n,k,seed):
 a,vis,h,gamma=fixed(n,k);rng=np.random.default_rng(seed+17*n+list(C['topologies']).index(k));het=np.zeros(n,dtype=bool);het[rng.choice(n,max(1,int(np.ceil(.25*n))),replace=False)]=1
 bias=het*rng.choice([-1.,1.],n);noise=rng.normal(0,.02,(400,n));return rng,a,vis,h,gamma,het,bias,noise
def wgt(a,b,p):
 if p=='independent_tracking':return np.zeros_like(a)
 if p=='two_layer_gate':return a*(np.abs(b)<1e-12)[None,:]
 return a
def observer(z,target,transition):
 # The target is held over one numerical sample.  H 1 = visibility, so the
 # exact sampled flow is target*1+exp(-rho H dt)(z-target*1).
 return target+transition@(z-target)
def robot(n,k,eta,seed,p):
 dt=.05;rng,a,vis,h,gamma,het,bias,noise=trial(n,k,seed);rho=eta/gamma;transition=transition_for(n,k,float(eta),dt);eff=np.where(het,rng.uniform(.72,1.18,n),1);x=-np.arange(n,dtype=float);v=np.zeros(n);z=np.zeros(n);w=wgt(a,bias,p);err=[];sep=[];en=0
 for j in range(400):
  target=1.2+.6*np.sin(.25*j*dt);z=observer(z,target,transition);desired=z-np.arange(n);coupling=w@(x+bias+np.arange(n))-w.sum(1)*(x+np.arange(n));u=np.clip(1.4*(desired-x)-1.1*v+.65*rho*coupling+noise[j],-2.5,2.5);v+=dt*eff*u;x+=dt*v;en+=float(np.sum(u*u)*dt);err.append(float(np.max(np.abs(x-(target-np.arange(n))))));sep.append(float(np.min(np.abs(np.diff(np.sort(x))))))
 e=float(np.quantile(err[-80:],.95));m=float(min(sep));return int(e<=.55 and m>=.25),e,m,en,rho,gamma
def motor(n,k,eta,seed,p):
 dt=.05;rng,a,vis,h,gamma,het,bias,noise=trial(n,k,seed);rho=eta/gamma;transition=transition_for(n,k,float(eta),dt);tau=np.where(het,rng.uniform(.35,.95,n),.55);s=np.zeros(n);z=np.zeros(n);w=wgt(a,bias,p);err=[];en=0
 for j in range(400):
  target=1.2+.35*np.sin(.25*j*dt);z=observer(z,target,transition);coupling=w@(s+bias)-w.sum(1)*s;u=np.clip(1.8*(z-s)+.7*rho*coupling+noise[j],-3,3);s+=dt*(-s+u)/tau;en+=float(np.sum(u*u)*dt);err.append(float(np.max(np.abs(s-target))))
 e=float(np.quantile(err[-80:],.95));return int(e<=.45),e,float('nan'),en,rho,gamma
def one(t):
 d,n,k,e,s,p=t;ok,er,m,en,rho,gamma=globals()[d](n,k,float(e),int(s),p);return dict(domain=d,n=n,topology=k,eta=e,rho=rho,gamma=gamma,seed=s,policy=p,success=ok,task_error=er,minimum_task_margin=m,control_energy=en)
def main():
 tasks=[(d,n,k,e,s,p) for d in C['domains'] for n in C['sizes'] for k in C['topologies'] for e in C['eta'] for s in C['heldout_seeds'] for p in C['policies']]
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,tasks,chunksize=20))
 with (O/'v10_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 q={'contract_sha256':hashlib.sha256((R/'V10_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),'runs':len(rows)};(O/'v10_summary.json').write_text(json.dumps(q,indent=2)+'\n');print(q)
if __name__=='__main__':main()
