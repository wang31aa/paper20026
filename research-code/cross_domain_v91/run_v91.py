#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; C=json.loads((ROOT/'V91_FROZEN_CONTRACT.json').read_text()); OUT=ROOT/'results';OUT.mkdir(exist_ok=True)

def graph(n,kind,rng):
 a=np.zeros((n,n));
 for i in range(1,n): a[i,i-1]=1
 if kind=='ring': a[0,-1]=1
 elif kind=='clustered':
  b=max(2,n//4)
  for i in range(n):
   for j in range(n):
    if i!=j and i//b==j//b:a[i,j]=1/max(b-1,1)
  for i in range(b,n,b):a[i,i-1]=1
 elif kind=='random_directed':
  for i in range(n):
   for j in range(n):
    if i!=j and rng.random()<min(.18,3/n):a[i,j]=1
  for i in range(1,n):a[i,i-1]=1
 s=a.sum(1);a[s>0]/=s[s>0,None];return a

def setup(n,kind,seed):
 rng=np.random.default_rng(seed);a=graph(n,kind,rng);visible=np.zeros(n);visible[:max(1,int(np.ceil(.1*n)))]=1
 het=np.zeros(n,dtype=bool);het[rng.choice(n,max(1,int(np.ceil(.25*n))),replace=False)]=1
 bias=het*rng.choice([-1.,1.],n);noise=rng.normal(0,.02,(300,n));return rng,a,visible,het,bias,noise

def obs(z,target,a,visible,rho,dt):return z+dt*rho*(1.4*(a@z-z)+2.2*visible*(target-z))
def weights(a,bias,policy):
 if policy=='independent_tracking':return np.zeros_like(a)
 if policy=='two_layer_gate':return a*(np.abs(bias)<1e-12)[None,:]
 return a
def effective_rho(rho,policy):return min(rho,1.15) if policy=='global_gain_reduction' else rho

def robot(n,kind,rho,seed,policy):
 dt=.05;rng,a,vis,het,bias,noise=setup(n,kind,seed);eff=np.where(het,rng.uniform(.72,1.18,n),1);x=-np.arange(n,dtype=float);v=np.zeros(n);z=np.zeros(n);w=weights(a,bias,policy);deadline=[];sep=[];energy=0
 for k in range(300):
  target=1.2+.6*np.sin(.25*k*dt);z=obs(z,target,a,vis,rho,dt);desired=z-np.arange(n);pr=effective_rho(rho,policy)
  coupling=w@(x+bias+np.arange(n))-w.sum(1)*(x+np.arange(n));u=np.clip(1.4*(desired-x)-1.1*v+.65*pr*coupling+noise[k],-2.5,2.5)
  v+=dt*eff*u;x+=dt*v;energy+=float(np.sum(u*u)*dt);sep.append(float(np.min(np.abs(np.diff(np.sort(x))))))
  if (k+1)*dt>=5:
   true_task=target-np.arange(n)
   deadline.append(float(np.max(np.abs(x-true_task))))
 task=float(max(deadline[:20]));margin=float(min(sep));return int(task<=.55 and margin>=.25),task,margin,energy,int(np.count_nonzero(a))*300

def vehicle(n,kind,rho,seed,policy):
 dt=.05;rng,a,vis,het,bias,noise=setup(n,kind,seed);tau=np.where(het,rng.uniform(.25,.75,n),.4);x=-28*np.arange(n,dtype=float);v=np.full(n,12.);acc=np.zeros(n);z=np.full(n,12.);w=weights(a,bias,policy);margins=[];errs=[];energy=0
 for k in range(300):
  target=12+(2 if k>=40 else 0)+1.2*np.sin(.18*k*dt);z=obs(z,target,a,vis,rho,dt);cmd=np.zeros(n);cmd[0]=.8*(z[0]-v[0]);g=x[:-1]-x[1:];rs=w.sum(1);comm=w@(v+bias);pr=effective_rho(rho,policy)
  cmd[1:]=.22*(g-(4+1.4*v[1:]))+.75*(z[1:]-v[1:])+.45*pr*(comm[1:]-rs[1:]*v[1:]);cmd=np.clip(cmd+noise[k],-3.5,2)
  acc+=dt*(cmd-acc)/tau;v=np.maximum(0,v+dt*acc);x+=dt*v;required=4+1.4*v[1:]+np.maximum((v[1:]**2-v[:-1]**2)/7,0);margins.append(float(np.min(x[:-1]-x[1:]-required)));errs.append(float(np.max(np.abs(v-z))));energy+=float(np.sum(cmd*cmd)*dt)
 tail=float(np.quantile(errs[-80:],.95));margin=float(min(margins));return int(margin>=0 and tail<=1.2),tail,margin,energy,int(np.count_nonzero(a))*300

def execute(t):
 d,n,k,r,s,p=t;ok,e,m,en,msg=globals()[d](n,k,float(r),int(s),p);return dict(domain=d,n=n,topology=k,rho=r,seed=s,policy=p,success=ok,task_error=e,minimum_task_margin=m,control_energy=en,message_count=msg)
def main():
 tasks=[(d,n,k,r,s,p) for d in C['domains'] for n in C['sizes'] for k in C['topologies'] for r in C['rho'] for s in C['heldout_seeds'] for p in C['policies']]
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(execute,tasks,chunksize=20))
 with (OUT/'v91_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 q={'contract_sha256':hashlib.sha256((ROOT/'V91_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),'runs':len(rows),'scope':C['claim_boundary']};(OUT/'v91_summary.json').write_text(json.dumps(q,indent=2)+'\n');print(q)
if __name__=='__main__':main()
