#!/usr/bin/env python3
"""Frozen three-domain computational pressure matrix with a common log schema."""
from __future__ import annotations
import csv, json, math
from itertools import product
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
DOMAINS=('robot','vehicle','uav')
POLICIES=('all_coupled','residual_gate','connectivity_gate','physical_filter','heuristic_high_gain')
FIELDS=('domain','run_id','split','policy_id','paired_replay_id','seed','time','dt','node_id',
'state_true','reference_true','reference_received','reference_estimate','information_error','message_age',
'packet_lost','graph_mode_id','target_reachable','participation_weight','switch_reason','control_requested',
'control_applied','control_saturated','disturbance_injected','physical_margin','task_failed','control_energy',
'communication_messages','theorem_assumptions_pass','physical_mapping_pass')

def graph(n,weight):
 L=np.zeros((n,n))
 for i in range(1,n): L[i,i-1]=-weight;L[i,i]=weight
 L[0,0]+=weight;L[0,n-1]-=weight
 return L

def simulate(domain,cond,policy,seed):
 rng=np.random.default_rng(seed); delay,loss,noise,authority=cond
 n=3 if domain=='vehicle' else 5; dim=1 if domain=='vehicle' else (2 if domain=='robot' else 3)
 dt={'robot':.1,'vehicle':.1,'uav':.05}[domain]; steps=120
 x=np.zeros((n,dim));v=np.zeros_like(x); ref=np.zeros_like(x)
 if domain=='vehicle': x[:,0]=np.array([0,-16,-32]);ref=x.copy()
 else:
  ref[:,0]=np.arange(n)*1.5;x=ref.copy()
 histories=[x.copy() for _ in range(delay+1)]; rows=[]; margins=[]; energy=0; msgs=0
 stress=0.04+0.08*delay+0.35*loss+0.55*noise+0.12/max(authority,0.1)
 for k in range(steps):
  target=ref.copy(); target[:,0]+=0.02*k
  received=histories[max(0,len(histories)-1-delay)].copy()+rng.normal(0,noise,x.shape)
  lost=rng.random(n)<loss; received[lost]=x[lost]
  info=np.linalg.norm(received-target,axis=1); score=info/(.25+noise)
  if policy=='all_coupled': weight=1.
  elif policy=='residual_gate': weight=float(np.mean(score<1.5))
  elif policy=='connectivity_gate': weight=max(.35,float(np.mean(score<1.5)))
  else: weight=max(.5,float(np.mean(score<1.5)))
  L=graph(n,weight)
  kp={'robot':.42,'vehicle':.34,'uav':.55}[domain];kd={'robot':0.,'vehicle':.8,'uav':.45}[domain]
  u=-kp*(x-target)-.12*(L@(x-target))-kd*v
  disturb=rng.normal(0,stress,size=x.shape)
  disturb[-1,0]+=(0.22+0.16*delay+loss+noise)*math.sin(.15*k)
  if policy=='heuristic_high_gain': u+=-.45*(x-target)-.35*v
  if policy=='physical_filter':
   if domain=='vehicle':
    gaps=x[:-1,0]-x[1:,0]
    if np.min(gaps)<10: u[1:,0]-=.8
   else:
    for i in range(n):
     for j in range(i):
      d=x[i]-x[j];q=np.linalg.norm(d)
      if q<1.0: u[i]+=0.5*d/max(q,1e-6)
  limit=authority*({'robot':.5,'vehicle':2.,'uav':1.5}[domain]);requested=u+disturb
  applied=np.clip(requested,-limit,limit); saturated=np.any(abs(requested)>limit+1e-12,axis=1)
  if domain=='vehicle' and 28<=k<48:
   applied[0,0]-=1.1*(1+delay+loss)
  if domain=='robot': x=x+dt*applied
  else: v=v+dt*applied;x=x+dt*v
  histories.append(x.copy());energy+=float(np.sum(applied**2)*dt);msgs+=int(weight*n*(n-1))
  if domain=='vehicle': margin=float(np.min(x[:-1,0]-x[1:,0]-12.5))
  else:
   sep=min(np.linalg.norm(x[i]-x[j]) for i in range(n) for j in range(i));track=1.5-np.max(np.linalg.norm(x-target,axis=1));margin=float(min(sep-.75,track))
  margins.append(margin)
  for i in range(n): rows.append(dict(zip(FIELDS,(domain,'_'.join(map(str,cond)),'exploratory_synthetic',policy,f'{domain}-{cond}',seed,k*dt,dt,i,json.dumps(x[i].tolist()),json.dumps(target[i].tolist()),json.dumps(received[i].tolist()),json.dumps(received[i].tolist()),float(info[i]),delay*dt,int(lost[i]),f'w{weight:.2f}',1,weight,'residual_or_physical',json.dumps(requested[i].tolist()),json.dumps(applied[i].tolist()),int(saturated[i]),json.dumps(disturb[i].tolist()),margin,int(margin<0),float(np.sum(applied[i]**2)*dt),int(weight*(n-1)),0,0))))
 return rows,{'domain':domain,'condition':'_'.join(map(str,cond)),'policy':policy,'failure':int(min(margins)<0),'minimum_margin':min(margins),'tail_margin':float(np.quantile(margins[-30:],.05)),'control_energy':energy,'communication_messages':msgs,'risk_score':(delay+.001)+5*loss+2*noise+1/max(authority,.01)}

def main():
 # This confirmation grid was fixed after a separately retained range-finding
 # run. Range-finding outputs are never pooled with these evaluations.
 conditions=list(product((0,1,3),(0.,.2),(0.03,.18),(.5,1.0))); logs=[];summary=[]
 for di,d in enumerate(DOMAINS):
  for ci,c in enumerate(conditions):
   for pi,p in enumerate(POLICIES):
    a,b=simulate(d,c,p,20260812+1000*di+10*ci);logs+=a;summary.append(b)
 with (OUT/'unified_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(logs)
 with (OUT/'pressure_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 print(json.dumps({'log_rows':len(logs),'run_policy_units':len(summary),'domains':3,'conditions_per_domain':len(conditions),'policies':len(POLICIES)},indent=2))
if __name__=='__main__':main()
