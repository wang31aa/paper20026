#!/usr/bin/env python3
"""Actual common-metric directed switching for the five V27 domains."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parent; sys.path.insert(0,str(ROOT/'cross_domain_v27')); import run_v27 as b
C=json.loads((R/'V28_DYNAMIC_SWITCHING_CONTRACT.json').read_text()); O=R/'results';O.mkdir(exist_ok=True);DT=C['dt_s'];K=C['steps']

@lru_cache(None)
def mode(n,epoch,rho):
 S=np.zeros((n,n))
 for i in range(n): S[i,(i-1)%n]=S[i,(i+1)%n]=.4
 shift=1+epoch%max(1,n-1); P=np.zeros((n,n))
 for i in range(n): P[i,(i+shift)%n]=.25
 A=S+P; L=np.diag(A.sum(1))-A; vis=np.zeros(n);vis[0]=1.;H=L+np.diag(vis);T=b.expm_stable(-rho*H*DT)
 return A,T,float(np.linalg.eigvalsh(H+H.T).min()),hashlib.sha256(A.tobytes()).hexdigest()

def simulate(t):
 d,n,rho,seed,policy=t; _,_,_,het,v,noise,bias,state=b.frozen(d,n,rho,seed);ph=b.digest(d,n,seed,v)
 z=np.zeros(n);err=[];margin=[];energy=0.;messages=0;hashes=[];minimum_mu=1e9
 if isinstance(state,tuple): p,vel=(x.copy() for x in state)
 else:p=state.copy()
 for k in range(K):
  A,T,mu,ah=mode(n,k//C['switch_period_steps'],rho);minimum_mu=min(minimum_mu,mu);hashes.append(ah)
  target=.55*math.sin(.18*k*DT);z=target+T@(z-target);x=p[:,0] if p.ndim>1 else p;res=np.abs(bias)+.06*np.abs(x-z);W=b.weights(A,res,policy);messages+=np.count_nonzero(W)
  coup=W@(x+bias)-W.sum(1)*x;ex=np.zeros_like(bias) if policy=='independent_tracking' else (.55*bias if policy=='global_gain_reduction' else bias)
  if policy in ('connectivity_gate','two_layer_gate'):ex*=res<.30
  if d=='robot':
   mass,drag,auth=v['mass_ratio'],.16*v['drag_ratio'],v['authority_ratio'];ref=np.c_[z-.7*np.arange(n),.35*np.sin(.11*k*DT+np.arange(n)/n)]
   u=1.4*(ref-p)-vel+.30*rho*coup[:,None]+1.6*rho*rho*ex[:,None]+.02*noise[k,:,:2]
   if policy=='physical_filter':
    for i in range(n):
     q=p[i]-p;ds=np.linalg.norm(q,axis=1);mask=(ds>0)&(ds<.42)
     if mask.any():u[i]+=.18*np.sum(q[mask]/(ds[mask,None]**2+.02),axis=0)
   u=np.clip(u,-2.6*auth[:,None],2.6*auth[:,None]);vel+=DT*(u/mass[:,None]-drag[:,None]*vel);p+=DT*vel
   e=np.max(np.linalg.norm(p-ref,axis=1));sep=np.min(np.linalg.norm(p[:,None]-p[None,:]+np.eye(n)[:,:,None]*1e5,axis=2));mar=min(.75-e,sep-.28)
  elif d=='microgrid':
   mass,drag,auth=4.5*v['inertia_ratio'],v['damping_ratio'],v['droop_authority_ratio'];u=-1.5*vel-.8*p+.35*(z-vel)+.24*rho*coup+2.2*rho*rho*ex+.02*noise[k,:,0]
   u=np.clip(u,-2.5*auth,2.5*auth);flow=A@p-A.sum(1)*p;vel+=DT*(u-drag*vel-1.1*p+.38*rho*flow)/mass;p+=DT*vel;e=np.max(np.abs(vel));mar=min(.65-e,.85-np.max(np.abs(p)))
  elif d=='circuit':
   al,be,auth=v['alpha_ratio'],v['beta_ratio'],v['authority_ratio'];x,y,w=p.T;u=1.1*(z-x)+.25*rho*coup+1.8*rho*rho*ex+.02*noise[k,:,0];u=np.clip(u,-2.4*auth,2.4*auth)
   p+=DT*np.c_[al*(y-x**3/3+x)+u,(x-y+w)/be,-.35*y-.18*w];e=np.max(np.abs(p[:,0]-z));mar=min(1.35-e,3.2-np.max(np.abs(p)))
  elif d=='water':
   area,out,auth=v['tank_area_ratio'],.22*v['outflow_ratio'],v['pump_gain_ratio'];u=.75*(1.4+z-p)+.20*rho*coup+1.7*rho*rho*ex+.015*noise[k,:,0]
   if policy=='physical_filter':u-=.8*np.maximum(p-2.1,0)
   u=np.clip(u,0,2.4*auth);p+=DT*(u-out*np.sqrt(np.maximum(p,0)))/area;p=np.maximum(p,0);e=np.max(np.abs(p-(1.4+target)));mar=min(.55-e,np.min(p)-.25,2.4-np.max(p))
  else:
   mass,drag,stiff,auth=v['mass_ratio'],.12*v['damping_ratio'],v['stiffness_ratio'],v['authority_ratio'];u=-1.4*p-.8*vel+.30*rho*coup+2*rho*rho*ex+.02*noise[k,:,0]
   if policy=='physical_filter':u-=.6*np.sign(p)*np.maximum(np.abs(p)-.4,0)
   u=np.clip(u,-2.6*auth,2.6*auth);vel+=DT*(u-drag*vel-stiff*p)/mass;p+=DT*vel;e=np.max(np.abs(p));mar=min(.55-e,1.3-np.max(np.abs(vel)))
  energy+=float(np.sum(np.asarray(u)**2)*DT);err.append(float(e));margin.append(float(mar))
 q=margin[25:];return {'domain':d,'n':n,'topology':'actual_common_metric_switching','rho':rho,'seed':seed,'policy':policy,'parameter_sha256':ph,'distinct_adjacencies':len(set(hashes)),'actual_switch_count':sum(a!=c for a,c in zip(hashes,hashes[1:])),
 'common_metric_mu':minimum_mu,'task_success':int(min(q)>=0),'minimum_physical_margin':min(q),'first_failure_time':next((i*DT for i,x in enumerate(q,25) if x<0),K*DT),'tail_error':float(np.quantile(err[-40:],.95)),'control_energy':energy,'communication_messages':messages,'evidence_class':C['claim_boundary']}

def main():
 tasks=[(d,n,r,s,p) for d in C['domains'] for n in C['sizes'] for r in C['rho_grid'] for s in C['seeds'] for p in C['policies']]
 with ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(simulate,tasks,chunksize=12))
 with (O/'v28_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':main()
