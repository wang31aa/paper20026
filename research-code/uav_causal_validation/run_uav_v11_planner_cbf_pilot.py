#!/usr/bin/env python3
"""Frozen V11 pilot: obstacle-aware reference plus lag-robust CBF projection."""
import csv, json
from pathlib import Path
import numpy as np
import run_uav_v9_reference_controller as base

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'UAV_V11_PLANNER_CBF_PILOT_PROTOCOL.json').read_text())
C=P['cbf']; PLAN=P['planner']

def shape_reference(goal,p,v,obstacles):
 out=goal.copy()
 for i in range(base.N):
  candidates=[]
  for ox,oy,rad in obstacles:
   ahead=ox-p[i,0]
   if 0.0<ahead<PLAN['lookahead_m']:
    side=np.sign(base.LANE[i]-oy)
    if side==0: side=1 if i%2 else -1
    desired=oy+side*(rad+base.Q['task']['minimum_obstacle_clearance_m']+PLAN['lateral_buffer_m'])
    candidates.append((ahead,desired))
  if candidates: out[i,1]=min(candidates)[1]
 return out

def project_halfspace(u,a,b):
 gap=b-float(np.sum(a*u))
 den=float(np.sum(a*a))
 if gap>0 and den>1e-12: u=u+gap*a/den
 return u

def barrier(u,p,v,obstacles):
 lim=base.Q['task']['maximum_acceleration_mps2']
 dp=base.Q['task']['minimum_pair_clearance_m']+C['pair_robustness_buffer_m']
 do=base.Q['task']['minimum_obstacle_clearance_m']+C['obstacle_robustness_buffer_m']
 for _ in range(C['projection_passes']):
  for i in range(base.N):
   for j in range(i+1,base.N):
    d=p[i]-p[j]; vr=v[i]-v[j]; h=float(d@d-dp*dp)
    rhs=-float(vr@vr)-C['k1']*float(d@vr)-.5*C['k0']*h
    a=np.zeros_like(u); a[i]=d; a[j]=-d
    u=project_halfspace(u,a,rhs)
   for ox,oy,rad in obstacles:
    d=p[i]-np.array([ox,oy]); safe=rad+do; h=float(d@d-safe*safe)
    rhs=-float(v[i]@v[i])-C['k1']*float(d@v[i])-.5*C['k0']*h
    a=np.zeros_like(u); a[i]=d
    u=project_halfspace(u,a,rhs)
  n=np.linalg.norm(u,axis=1); m=n>lim
  u[m]*=(lim/n[m])[:,None]
 return u

def main():
 base.shape_reference=shape_reference
 base.barrier=barrier
 rows=[]
 for env in base.ENVS['development']:
  for seed in P['pilot_seeds']:
   for rho in P['participation_grid']:
    for policy in P['policies']:
     rows.append(base.run(policy,float(rho),env,int(seed)))
 path=base.OUT/'uav_v11_planner_cbf_pilot_runs.csv'
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 rates={p:{str(r):float(np.mean([x['success'] for x in rows if x['policy']==p and x['rho']==r])) for r in P['participation_grid']} for p in P['policies']}
 out={
  'rows':len(rows),
  'success_by_policy_rho':rates,
  'minimum_pair_clearance_m':min(x['minimum_pair_clearance_m'] for x in rows),
  'minimum_obstacle_clearance_m':min(x['minimum_obstacle_clearance_m'] for x in rows),
  'scope':'development pilot only; not heldout confirmation'
 }
 (base.OUT/'uav_v11_planner_cbf_pilot_summary.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))

if __name__=='__main__': main()
