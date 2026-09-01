#!/usr/bin/env python3
"""Frozen V9 heterogeneous controller in an EPFL-scale reference environment."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
Q=json.loads((HERE/'UAV_V9_REFERENCE_ENVIRONMENT_PROTOCOL.json').read_text())
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
N=Q['agents']; DT=Q['dt_s']; STEPS=round(Q['duration_s']/DT)
B=np.array(Q['heterogeneous_plant']['acceleration_effectiveness'])
D=np.array(Q['heterogeneous_plant']['linear_drag_per_s'])
TAU=np.array(Q['heterogeneous_plant']['actuator_time_constant_s'])
TRIM=np.array(Q['heterogeneous_plant']['persistent_trim_mps2'])
LANE=np.linspace(-1.25,1.25,N)

ENVS={
 'development':[('sparse',[(5.0,.70,.40),(9.5,-.72,.40)],.06,.03),
                ('dense',[(4.0,.70,.40),(7.0,-.70,.42),(10.0,.72,.40),(12.0,-.72,.40)],.10,.08)],
 'heldout':[('crosswind',[(5.0,-.68,.42),(9.0,.70,.42),(11.8,-.72,.40)],.15,.05),
            ('lossy_dense',[(3.8,.68,.40),(6.2,-.68,.42),(8.6,.70,.42),(11.0,-.70,.42)],.11,.16)]}

def streams(seed,wind,loss):
 g=np.random.default_rng(seed)
 return g.normal(0,wind,(STEPS,N,2)),g.normal(0,.012,(STEPS,N,2)),g.random((STEPS,N,N))>=loss

def graph(packet):
 A=np.zeros((N,N))
 for i in range(1,N): A[i,i-1]=packet[i,i-1]
 A[0,N-1]=packet[0,N-1]
 return A

def observer_step(z,A,rho):
 target=np.full(N,Q['task']['goal_x_m']); target[1:]=z[1:]
 dz=A@(z)-A.sum(1)*z
 dz[0]+=Q['information']['observer_gain']*(Q['task']['goal_x_m']-z[0])
 return z+DT*rho*dz

def barrier(u,p,v,obstacles):
 # Early reciprocal-style projection: retain the physical threshold and act
 # before the boundary rather than changing the threshold after failure.
 for i in range(N):
  for j in range(i+1,N):
   d=p[i]-p[j]; r=np.linalg.norm(d)
   if 1e-9<r<1.0:
    push=3.2*(1.0-r)*d/r; u[i]+=push; u[j]-=push
  for ox,oy,rad in obstacles:
   d=p[i]-np.array([ox,oy]); r=np.linalg.norm(d); activation=rad+.35+1.15
   if 1e-9<r<activation:
    radial=d/r
    # A deterministic tangential bias prevents a head-on equilibrium.
    tangent=np.array([-radial[1],radial[0]])
    if tangent[1]*np.sign(LANE[i]-oy)<0:tangent=-tangent
    u[i]+=4.2*(activation-r)*radial+1.15*(activation-r)*tangent
 return u

def shape_reference(goal,p,v,obstacles):
 # V9 identity hook. Later, separately frozen controller protocols may replace
 # it without changing this historical experiment or its physical thresholds.
 return goal

def run(policy,rho,env,seed):
 name,obstacles,wind,loss=env; W,NZ,PK=streams(seed,wind,loss)
 p=np.c_[np.zeros(N),LANE]; v=np.zeros((N,2)); applied=np.zeros((N,2)); z=np.zeros(N)
 min_pair=min_obs=np.inf; energy=messages=0.; first_fail=None; recovered=0.; gated=False
 for k in range(STEPS):
  A=graph(PK[k]); messages+=np.count_nonzero(A); z=observer_step(z,A,rho)
  measured=p+NZ[k]; goal=shape_reference(np.c_[z,LANE],p,v,obstacles)
  u=1.25*(goal-measured)-1.45*v
  if policy=='pf':u=.72*(goal-measured)-.95*v
  residual=np.abs(TRIM)+.06*np.linalg.norm(v,axis=1)+.08*np.abs(z-Q['task']['goal_x_m'])
  physical=A.copy()
  if policy=='two_layer_heterogeneity_gate':
   keep=residual<np.quantile(residual,.65); physical*=keep[None,:]; gated|=not np.array_equal(physical,A)
  coupling=physical@(p+TRIM[:,None])-physical.sum(1)[:,None]*p
  u+=.28*rho*coupling
  if policy in ('new_mpc_cbf','two_layer_heterogeneity_gate'):u=barrier(u,p,v,obstacles)
  norm=np.linalg.norm(u,axis=1); lim=Q['task']['maximum_acceleration_mps2']; m=norm>lim;u[m]*=(lim/norm[m])[:,None]
  applied+=DT*(u-applied)/TAU[:,None]
  v+=DT*(B[:,None]*applied-D[:,None]*v+TRIM[:,None]*np.array([1.,.25])+W[k]);p+=DT*v
  speed=np.linalg.norm(v,axis=1);m=speed>Q['task']['maximum_speed_mps'];v[m]*=(Q['task']['maximum_speed_mps']/speed[m])[:,None]
  pair=min(np.linalg.norm(p[i]-p[j]) for i in range(N) for j in range(i+1,N))
  obs=min(np.linalg.norm(p[i]-[ox,oy])-rad for i in range(N) for ox,oy,rad in obstacles)
  min_pair=min(min_pair,pair);min_obs=min(min_obs,obs);energy+=DT*np.sum(applied*applied)
  feasible=pair>=Q['task']['minimum_pair_clearance_m'] and obs>=Q['task']['minimum_obstacle_clearance_m']
  if not feasible and first_fail is None:first_fail=k*DT
  if feasible and first_fail is not None:recovered+=DT
 completion=np.mean(p[:,0]>=Q['task']['completion_x_m'])
 success=completion>=Q['task']['completion_fraction'] and min_pair>=Q['task']['minimum_pair_clearance_m'] and min_obs>=Q['task']['minimum_obstacle_clearance_m']
 return dict(environment=name,seed=seed,rho=rho,policy=policy,success=int(success),completion_fraction=completion,minimum_pair_clearance_m=min_pair,minimum_obstacle_clearance_m=min_obs,control_energy=energy,communication_messages=int(messages),first_failure_time_s=first_fail if first_fail is not None else Q['duration_s'],recovery_dwell_s=recovered,gate_changed_physical_graph=int(gated))

def main():
 rows=[]
 for split in ('development','heldout'):
  for env in ENVS[split]:
   for seed in Q[f'{split}_seeds']:
    for rho in Q['participation_grid']:
     for policy in Q['policies']:
      r=run(policy,float(rho),env,int(seed));r['split']=split;rows.append(r)
 path=OUT/'uav_v9_reference_runs.csv'
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 dev=[r for r in rows if r['split']=='development' and r['policy']=='two_layer_heterogeneity_gate']
 rates={str(x):float(np.mean([r['success'] for r in dev if r['rho']==x])) for x in Q['participation_grid']}
 window=[float(x) for x,y in rates.items() if y>=.8]
 test=[r for r in rows if r['split']=='heldout' and r['policy']=='two_layer_heterogeneity_gate']
 tp=tn=fp=fn=0
 for r in test:
  pred=r['rho'] in window; actual=bool(r['success']);tp+=pred and actual;tn+=(not pred) and (not actual);fp+=pred and not actual;fn+=(not pred) and actual
 summary={'protocol_sha256':hashlib.sha256((HERE/'UAV_V9_REFERENCE_ENVIRONMENT_PROTOCOL.json').read_bytes()).hexdigest(),'rows':len(rows),'heldout_rows':len([r for r in rows if r['split']=='heldout']),'development_gate_success_by_rho':rates,'frozen_feasible_rho':window,'heldout_confusion':{'tp':tp,'tn':tn,'fp':fp,'fn':fn},'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'evidence_class':Q['evidence_class']}
 (OUT/'uav_v9_reference_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
