#!/usr/bin/env python3
"""Corrected node-traceable re-execution of the five historical domains."""
from __future__ import annotations
import csv, hashlib, json, math
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent; C=json.loads((R/'V27_FROZEN_CONTRACT.json').read_text())
O=R/'results'; O.mkdir(exist_ok=True); DT=float(C['dt_s']); K=int(C['steps'])
PARAMS={
 'robot': [('mass_ratio','1'),('drag_ratio','1'),('authority_ratio','1')],
 'microgrid':[('inertia_ratio','1'),('damping_ratio','1'),('droop_authority_ratio','1')],
 'circuit':[('alpha_ratio','1'),('beta_ratio','1'),('authority_ratio','1')],
 'water':[('tank_area_ratio','1'),('outflow_ratio','1'),('pump_gain_ratio','1')],
 'structure':[('mass_ratio','1'),('damping_ratio','1'),('stiffness_ratio','1'),('authority_ratio','1')]}
RANGES={
 'robot':[(.7,1.5),(.5,1.75),(.65,1.0)],
 'microgrid':[(.556,1.556),(.5,1.7),(.7,1.15)],
 'circuit':[(.7,1.4),(.7,1.3),(.65,1.1)],
 'water':[(.7,1.5),(.636,1.545),(.65,1.1)],
 'structure':[(.7,1.5),(.5,1.833),(.7,1.5),(.7,1.05)]}

def expm_stable(M):
 """Matrix exponential for the small frozen observer matrices."""
 lam,V=np.linalg.eig(M); X=V@np.diag(np.exp(lam))@np.linalg.inv(V)
 assert np.max(np.abs(np.imag(X)))<1e-9
 X=np.real(X); assert np.isfinite(X).all(); return X

def adjacency(n,seed):
 g=np.random.default_rng(seed); A=np.zeros((n,n))
 for i in range(1,n): A[i,i-1]=1.
 A=np.maximum(A,(g.random((n,n))<min(.18,3/n)).astype(float)); np.fill_diagonal(A,0)
 s=A.sum(1); A[s>0]/=s[s>0,None]; return A

@lru_cache(None)
def frozen(domain,n,rho,seed):
 di=C['domains'].index(domain); rng=np.random.default_rng(seed+97*n+1009*di)
 A=adjacency(n,13001+31*n+di); visible=np.zeros(n); visible[:max(1,math.ceil(.4*n))]=1
 L=np.diag(A.sum(1))-A; H=L+np.diag(visible); T=expm_stable(-rho*H*DT)
 hetero=np.zeros(n,bool); hetero[rng.choice(n,max(1,math.ceil(.5*n)),False)]=1
 vals={}
 for (name,_),(lo,hi) in zip(PARAMS[domain],RANGES[domain]):
  x=np.ones(n); x[hetero]=rng.uniform(lo,hi,hetero.sum()); vals[name]=x
 noise=rng.normal(size=(K,n,3)); bias=np.where(hetero,rng.choice([-1.,1.],n)*rng.uniform(.10,.32,n),0.)
 if domain=='robot': state=(np.c_[-.7*np.arange(n),np.zeros(n)],np.zeros((n,2)))
 elif domain=='circuit': state=np.c_[.1*rng.normal(size=n),np.zeros(n),np.zeros(n)]
 elif domain=='water': state=np.ones(n)
 else: state=(np.zeros(n),np.zeros(n))
 return A,visible,T,hetero,vals,noise,bias,state

def digest(domain,n,seed,vals):
 payload={'domain':domain,'n':n,'seed':seed,'parameters':{k:[float(v) for v in vals[k]] for k,_ in PARAMS[domain]}}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def weights(A,res,policy):
 if policy in ('all_coupled','physical_filter'): return A.copy()
 if policy=='global_gain_reduction': return .55*A
 if policy=='independent_tracking': return np.zeros_like(A)
 W=A*(res<.30)[None,:]
 if policy in ('connectivity_gate','two_layer_gate'):
  for i in range(1,len(A)): W[i,i-1]=max(W[i,i-1],.18)
 return W

def simulate(domain,n,rho,seed,policy):
 A,visible,T,hetero,v,noise,bias,state=frozen(domain,n,rho,seed); ph=digest(domain,n,seed,v)
 z=np.zeros(n); err=[]; margin=[]; energy=0.; messages=0
 if isinstance(state,tuple): p,vel=(x.copy() for x in state)
 else: p=state.copy()
 for k in range(K):
  target=.55*math.sin(.18*k*DT); z=target+T@(z-target)
  x=p[:,0] if p.ndim>1 else p; res=np.abs(bias)+.06*np.abs(x-z); W=weights(A,res,policy); messages+=np.count_nonzero(W)
  coup=W@(x+bias)-W.sum(1)*x
  exposure=np.zeros_like(bias) if policy=='independent_tracking' else (.55*bias if policy=='global_gain_reduction' else bias)
  if policy in ('connectivity_gate','two_layer_gate'): exposure*=res<.30
  if domain=='robot':
   m,d,a=v['mass_ratio'],.16*v['drag_ratio'],v['authority_ratio']; ref=np.c_[z-.7*np.arange(n),.35*np.sin(.11*k*DT+np.arange(n)/n)]
   u=1.4*(ref-p)-vel+.30*rho*coup[:,None]+1.6*rho*rho*exposure[:,None]+.02*noise[k,:,:2]
   if policy=='physical_filter':
    for i in range(n):
     q=p[i]-p; ds=np.linalg.norm(q,axis=1); mask=(ds>0)&(ds<.42)
     if mask.any(): u[i]+=.18*np.sum(q[mask]/(ds[mask,None]**2+.02),axis=0)
   u=np.clip(u,-2.6*a[:,None],2.6*a[:,None]); vel+=DT*(u/m[:,None]-d[:,None]*vel); p+=DT*vel
   e=np.max(np.linalg.norm(p-ref,axis=1)); sep=np.min(np.linalg.norm(p[:,None]-p[None,:]+np.eye(n)[:,:,None]*1e5,axis=2)); mar=min(.75-e,sep-.28)
  elif domain=='microgrid':
   m,d,a=4.5*v['inertia_ratio'],v['damping_ratio'],v['droop_authority_ratio']; u=-1.5*vel-.8*p+.35*(z-vel)+.24*rho*coup+2.2*rho*rho*exposure+.02*noise[k,:,0]
   u=np.clip(u,-2.5*a,2.5*a); flow=A@p-A.sum(1)*p; vel+=DT*(u-d*vel-1.1*p+.38*rho*flow)/m; p+=DT*vel
   e=np.max(np.abs(vel)); mar=min(.65-e,.85-np.max(np.abs(p)))
  elif domain=='circuit':
   al,be,a=v['alpha_ratio'],v['beta_ratio'],v['authority_ratio']; x,y,w=p.T; u=1.1*(z-x)+.25*rho*coup+1.8*rho*rho*exposure+.02*noise[k,:,0]
   u=np.clip(u,-2.4*a,2.4*a); p+=DT*np.c_[al*(y-x**3/3+x)+u,(x-y+w)/be,-.35*y-.18*w]
   e=np.max(np.abs(p[:,0]-z)); mar=min(1.35-e,3.2-np.max(np.abs(p)))
  elif domain=='water':
   ar,out,a=v['tank_area_ratio'],.22*v['outflow_ratio'],v['pump_gain_ratio']; u=.75*(1.4+z-p)+.20*rho*coup+1.7*rho*rho*exposure+.015*noise[k,:,0]
   if policy=='physical_filter': u-=.8*np.maximum(p-2.1,0)
   u=np.clip(u,0,2.4*a); p+=DT*(u-out*np.sqrt(np.maximum(p,0)))/ar; p=np.maximum(p,0)
   e=np.max(np.abs(p-(1.4+target))); mar=min(.55-e,np.min(p)-.25,2.4-np.max(p))
  else:
   m,d,st,a=v['mass_ratio'],.12*v['damping_ratio'],v['stiffness_ratio'],v['authority_ratio']; u=-1.4*p-.8*vel+.30*rho*coup+2*rho*rho*exposure+.02*noise[k,:,0]
   if policy=='physical_filter': u-=.6*np.sign(p)*np.maximum(np.abs(p)-.4,0)
   u=np.clip(u,-2.6*a,2.6*a); vel+=DT*(u-d*vel-st*p)/m; p+=DT*vel
   e=np.max(np.abs(p)); mar=min(.55-e,1.3-np.max(np.abs(vel)))
  energy+=float(np.sum(np.asarray(u)**2)*DT); err.append(float(e)); margin.append(float(mar))
 q=margin[25:]
 return {'domain':domain,'n':n,'topology':'switching','rho':rho,'seed':seed,'policy':policy,'parameter_sha256':ph,
  'task_success':int(min(q)>=0),'minimum_physical_margin':min(q),'tail_error':float(np.quantile(err[-40:],.95)),
  'first_failure_time':next((i*DT for i,x in enumerate(margin[25:],25) if x<0),K*DT),'control_energy':energy,
  'communication_messages':messages,'parameter_evidence_class':C['parameter_evidence_class'],'parameter_unit_class':'NORMALIZED_RATIO'}

def simulate_task(t): return simulate(*t)

def main():
 nodes=[]; seen=set(); tasks=[]
 for d in C['domains']:
  for n in C['sizes']:
   for rho in C['rho_grid']:
    for seed in C['seeds']:
     _,_,_,_,vals,_,_,_=frozen(d,n,rho,seed); ph=digest(d,n,seed,vals)
     key=(d,n,seed,ph)
     if key not in seen:
      seen.add(key)
      for name,unit in PARAMS[d]:
       for node,value in enumerate(vals[name]): nodes.append({'domain':d,'n':n,'seed':seed,'node':node,'parameter':name,'value':float(value),'unit':unit,'evidence_class':C['parameter_evidence_class'],'parameter_sha256':ph})
     for policy in C['policies']: tasks.append((d,n,rho,seed,policy))
 with ThreadPoolExecutor(max_workers=8) as ex: rows=list(ex.map(simulate_task,tasks,chunksize=12))
 for path,data in [(O/'v27_runs.csv',rows),(O/'v27_node_parameters.csv',nodes)]:
  with path.open('w',newline='') as f: w=csv.DictWriter(f,data[0].keys()); w.writeheader(); w.writerows(data)
 print(json.dumps({'trajectory_rows':len(rows),'node_parameter_rows':len(nodes)}))
if __name__=='__main__': main()
