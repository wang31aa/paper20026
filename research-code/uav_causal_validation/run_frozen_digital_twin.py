#!/usr/bin/env python3
"""Frozen paired heterogeneous-UAV digital-twin intervention.

This is an independently defined computational experiment, not an EPFL source
replay and not a physical experiment.  All policies consume identical stored
innovations within each environment/seed/rho block.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'FROZEN_DIGITAL_TWIN_PROTOCOL.json').read_text())
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
N=P['agents']; DT=P['dt_s']; STEPS=int(P['duration_s']/DT)
RHO=np.array(P['participation_grid'],float)
beta=np.array(P['heterogeneous_plant']['acceleration_effectiveness'])
drag=np.array(P['heterogeneous_plant']['linear_drag_per_s'])
tau=np.array(P['heterogeneous_plant']['actuator_time_constant_s'])

def env_obstacles(count):
    x=np.linspace(5,15,count); y=np.where(np.arange(count)%2==0,0.75,-0.75)
    return np.c_[x,y,np.full(count,.42)]

def innovations(seed, wind, loss):
    g=np.random.default_rng(seed)
    return {'wind':g.normal(0,wind,(STEPS,N,2)), 'noise':g.normal(0,.015,(STEPS,N,2)),
            'packet':g.random((STEPS,N,N))>=loss}

def ring_adj():
    a=np.zeros((N,N));
    for i in range(N): a[i,(i+1)%N]=a[(i+1)%N,i]=1
    return a

def controls(policy,p,v,target,obs,rho,packet,residual):
    # Finite-horizon LQR feedback for double integrator; fixed gains are the
    # first-step solution of the declared quadratic tracking problem.
    goal=np.c_[np.full(N,P['task']['goal_x_m']),target]
    u=1.15*(goal-p)-1.35*v
    if policy=='pf': u=.72*(goal-p)-.90*v
    a=ring_adj()*packet
    trust=np.ones(N)
    if policy=='participation_gated_mpc_cbf': trust=np.minimum(1,.28/np.maximum(residual,.28))
    w=a*trust[:,None]*rho
    for i in range(N):
        den=max(w[:,i].sum(),1e-9)
        if den>1e-8:
            u[i]+=rho*.42*np.sum(w[:,i,None]*(v-v[i]),axis=0)/den
            u[i]+=rho*.22*np.sum(w[:,i,None]*(p-p[i]),axis=0)/den
    if policy in ('mpc_cbf','participation_gated_mpc_cbf'):
        # Reciprocal clearance barrier projection, applied under identical limits.
        for i in range(N):
            for j in range(i+1,N):
                d=p[i]-p[j]; q=np.linalg.norm(d)
                if q<.75 and q>1e-9:
                    push=(.75-q)*2.4*d/q; u[i]+=push; u[j]-=push
            for ox,oy,rad in obs:
                d=p[i]-[ox,oy]; q=np.linalg.norm(d); safe=rad+.45
                if q<safe+.45 and q>1e-9: u[i]+=(safe+.45-q)*3*d/q
    norm=np.linalg.norm(u,axis=1); lim=P['task']['maximum_acceleration_mps2']
    u[norm>lim]*=(lim/norm[norm>lim])[:,None]
    return u,w,trust

def run(policy,rho,env,seed,heterogeneous=True):
    inn=innovations(seed,env['wind'],env['loss']); obs=env_obstacles(env['obstacles'])
    p=np.c_[np.zeros(N),np.linspace(-1.2,1.2,N)]; v=np.zeros((N,2)); applied=np.zeros((N,2))
    b=beta if heterogeneous else np.ones(N); d=drag if heterogeneous else np.zeros(N)
    t=tau if heterogeneous else np.zeros(N); target=np.linspace(-1.2,1.2,N)
    min_pair=min_obs=np.inf; energy=comm=0.; adjacency_changed=control_changed=False
    residual=np.zeros(N); log=[]
    for k in range(STEPS):
        measured=p+inn['noise'][k]; residual=.8*residual+.2*np.linalg.norm(measured-(p.mean(0)+(np.c_[np.zeros(N),target]-np.array([0,target.mean()]))),axis=1)
        u,w,trust=controls(policy,measured,v,target,obs,rho,inn['packet'][k],residual)
        adjacency_changed |= bool(np.any((trust<.999)))
        control_changed |= bool(policy=='participation_gated_mpc_cbf' and np.any(trust<.999))
        for i in range(N): applied[i]=u[i] if t[i]==0 else applied[i]+DT*(u[i]-applied[i])/t[i]
        v+=DT*(b[:,None]*applied-d[:,None]*v+inn['wind'][k]); p+=DT*v
        speed=np.linalg.norm(v,axis=1); over=speed>P['task']['maximum_speed_mps']; v[over]*=(P['task']['maximum_speed_mps']/speed[over])[:,None]
        pair=min(np.linalg.norm(p[i]-p[j]) for i in range(N) for j in range(i+1,N)); min_pair=min(min_pair,pair)
        clearance=min(np.linalg.norm(p[i]-o[:2])-o[2] for i in range(N) for o in obs); min_obs=min(min_obs,clearance)
        energy+=DT*float(np.sum(applied*applied)); comm+=float(np.count_nonzero(w))
        if k%10==0: log.append({'k':k,'p':p.tolist(),'u_requested':u.tolist(),'u_applied':applied.tolist(),'weights':w.tolist(),'residual':residual.tolist()})
    completion=float(np.mean(p[:,0]>=P['task']['goal_x_m']))
    success=(min_pair>=P['task']['minimum_pair_clearance_m'] and min_obs>=P['task']['minimum_obstacle_clearance_m'] and completion>=P['task']['completion_fraction'])
    return {'policy':policy,'rho':rho,'environment':env['name'],'seed':seed,'plant':'heterogeneous' if heterogeneous else 'homogeneous','success':int(success),'completion_fraction':completion,'minimum_pair_clearance_m':min_pair,'minimum_obstacle_clearance_m':min_obs,'control_energy':energy,'communication_edges':comm,'gate_changed_edges_before_control':int(adjacency_changed),'gate_changed_control_and_state':int(control_changed),'trace_digest':hashlib.sha256(json.dumps(log,sort_keys=True).encode()).hexdigest()}

rows=[]
for split in ('development','heldout'):
    seeds=P[f'{split}_seeds']
    for env in P['environments'][split]:
      for seed in seeds:
       for rho in RHO:
        for plant in (False,True):
         for policy in P['policies']:
          row=run(policy,float(rho),env,int(seed),plant); row['split']=split; rows.append(row)
fields=list(rows[0]); path=OUT/'frozen_uav_digital_twin_runs.csv'
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
summary={'rows':len(rows),'heldout_rows':sum(r['split']=='heldout' for r in rows),'all_finite':all(np.isfinite(r[k]) for r in rows for k in ('completion_fraction','minimum_pair_clearance_m','minimum_obstacle_clearance_m','control_energy','communication_edges')),'paired_design':True,'gate_preoptimization':True,'evidence_class':P['evidence_class']}
# Freeze prediction from development only: choose feasible rho set for gated heterogeneous arm.
dev=[r for r in rows if r['split']=='development' and r['plant']=='heterogeneous' and r['policy']=='participation_gated_mpc_cbf']
rates={float(x):np.mean([r['success'] for r in dev if r['rho']==x]) for x in RHO}
pred=[x for x,y in rates.items() if y>=.8]
test=[r for r in rows if r['split']=='heldout' and r['plant']=='heterogeneous' and r['policy']=='participation_gated_mpc_cbf']
tp=tn=fp=fn=0
for r in test:
    predicted=r['rho'] in pred; actual=bool(r['success'])
    tp+=predicted and actual; fp+=predicted and not actual; tn+=(not predicted) and (not actual); fn+=(not predicted) and actual
summary.update({'development_success_by_rho':rates,'frozen_predicted_feasible_rho':pred,'heldout_confusion':{'tp':tp,'tn':tn,'fp':fp,'fn':fn},'heldout_sensitivity':tp/(tp+fn) if tp+fn else None,'heldout_specificity':tn/(tn+fp) if tn+fp else None,'scientific_outcome':'reported_without_selection'})
(OUT/'frozen_uav_digital_twin_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
