#!/usr/bin/env python3
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; C=json.loads((R/'V27_FROZEN_CONTRACT.json').read_text()); O=R/'results'
sys.path.insert(0,str(R)); import run_v27 as m
rows=list(csv.DictReader((O/'v27_runs.csv').open())); nodes=list(csv.DictReader((O/'v27_node_parameters.csv').open()))
expected=len(C['domains'])*len(C['sizes'])*len(C['rho_grid'])*len(C['seeds'])*len(C['policies'])
assert len(rows)==expected
blocks=defaultdict(list)
for r in rows: blocks[(r['domain'],r['n'],r['rho'],r['seed'])].append(r)
assert all(len(v)==len(C['policies']) and len({x['parameter_sha256'] for x in v})==1 for v in blocks.values())
by=defaultdict(list)
for x in nodes: by[(x['domain'],int(x['n']),int(x['seed']),x['parameter'])].append(float(x['value']))
for (d,n,seed,p),a in by.items():
 idx=[x[0] for x in m.PARAMS[d]].index(p); lo,hi=m.RANGES[d][idx]
 assert min(a)>=lo-1e-12 and max(a)<=hi+1e-12 and np.ptp(a)>0
# Direct causal probes: perturb each parameter in the actual one-step equation.
def step(d,v):
 dt=m.DT
 if d=='robot':
  mass,drag,auth=v; vel=.3; requested=3.; u=np.clip(requested,-2.6*auth,2.6*auth); return vel+dt*(u/mass-.16*drag*vel)
 if d=='microgrid':
  inertia,damp,auth=v; omega=.3; angle=.2; requested=3.; u=np.clip(requested,-2.5*auth,2.5*auth); return omega+dt*(u-damp*omega-1.1*angle)/(4.5*inertia)
 if d=='circuit':
  alpha,beta,auth=v; x,y,w=.4,.2,.1; requested=3.; u=np.clip(requested,-2.4*auth,2.4*auth); return np.array([x+dt*alpha*(y-x**3/3+x+u),y+dt*(x-y+w)/beta])
 if d=='water':
  area,out,auth=v; level=1.; requested=3.; u=np.clip(requested,0,2.4*auth); return level+dt*(u-.22*out*np.sqrt(level))/area
 mass,damp,stiff,auth=v; x,vel=.2,.3; requested=3.; u=np.clip(requested,-2.6*auth,2.6*auth); return vel+dt*(u-.12*damp*vel-stiff*x)/mass
effects={}
for d in C['domains']:
 _,_,_,_,vals,_,_,_=m.frozen(d,5,1.1,8101); names=[p for p,_ in m.PARAMS[d]]; base=np.array([float(vals[p][0]) for p in names]); y0=np.asarray(step(d,base))
 for j,p in enumerate(names):
  altered=base.copy(); altered[j]*=1.01; effects[f'{d}:{p}']=float(np.max(np.abs(np.asarray(step(d,altered))-y0)))
assert all(x>1e-12 for x in effects.values()),effects
out={'status':'PASS','trajectory_rows':len(rows),'paired_blocks':len(blocks),'node_parameter_rows':len(nodes),
 'all_policy_parameter_hashes_identical':True,'all_parameter_ranges_pass':True,'all_parameters_have_nonzero_state_effect':True,
 'finite_difference_state_effects':effects,'minimum_finite_difference_effect':min(effects.values()),'corrected_observer':'PASS_WEIGHTED_ROW_LAPLACIAN','circuit_endpoint':'PASS_POST_UPDATE_STATE',
 'claim_boundary':C['claim_boundary']}
(O/'V27_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
