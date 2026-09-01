#!/usr/bin/env python3
"""Frozen pilot of a relative-degree-two CBF projection for the V9 plant."""
import csv,json
from pathlib import Path
import numpy as np
import run_uav_v9_reference_controller as base

H=Path(__file__).resolve().parent
P=json.loads((H/'UAV_V10_CBF_PILOT_PROTOCOL.json').read_text())
k0=P['cbf']['k0'];k1=P['cbf']['k1'];passes=P['cbf']['projection_passes']

def project_halfspace(u,a,b):
 val=float(np.sum(a*u))
 if val<b:
  den=float(np.sum(a*a))
  if den>1e-12:u=u+(b-val)*a/den
 return u

def cbf(u,p,v,obstacles):
 # Project the nominal acceleration onto pair- and obstacle-barrier halfspaces.
 # Repeated projections solve the convex feasibility part; the final run is
 # not certified unless the independently recomputed violation is below tol.
 lim=base.Q['task']['maximum_acceleration_mps2'];dmin=base.Q['task']['minimum_pair_clearance_m']
 for _ in range(passes):
  for i in range(base.N):
   for j in range(i+1,base.N):
    d=p[i]-p[j];vr=v[i]-v[j];h=float(d@d-dmin*dmin)
    rhs=-float(vr@vr)-k1*float(d@vr)-.5*k0*h
    a=np.zeros_like(u);a[i]=d;a[j]=-d;u=project_halfspace(u,a,rhs)
   for ox,oy,rad in obstacles:
    d=p[i]-np.array([ox,oy]);safe=rad+base.Q['task']['minimum_obstacle_clearance_m'];h=float(d@d-safe*safe)
    rhs=-float(v[i]@v[i])-k1*float(d@v[i])-.5*k0*h
    a=np.zeros_like(u);a[i]=d;u=project_halfspace(u,a,rhs)
  n=np.linalg.norm(u,axis=1);m=n>lim;u[m]*=(lim/n[m])[:,None]
 return u

base.barrier=cbf
rows=[]
for env in base.ENVS['development']:
 for seed in P['pilot_seeds']:
  for rho in P['participation_grid']:
   for policy in P['policies']:
    rows.append(base.run(policy,float(rho),env,int(seed)))
path=base.OUT/'uav_v10_cbf_pilot_runs.csv'
with path.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={p:{str(r):float(np.mean([x['success'] for x in rows if x['policy']==p and x['rho']==r])) for r in P['participation_grid']} for p in P['policies']}
out={'rows':len(rows),'success_by_policy_rho':summary,'minimum_pair_clearance_m':min(x['minimum_pair_clearance_m'] for x in rows),'minimum_obstacle_clearance_m':min(x['minimum_obstacle_clearance_m'] for x in rows),'scope':'development pilot only; not heldout confirmation'}
(base.OUT/'uav_v10_cbf_pilot_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
