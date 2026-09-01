#!/usr/bin/env python3
"""Fail-closed registry for the V11 development-only controller pilot."""
import csv, hashlib, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
protocol=HERE/'UAV_V11_PLANNER_CBF_PILOT_PROTOCOL.json'
runs=HERE/'results/uav_v11_planner_cbf_pilot_runs.csv'
summary=HERE/'results/uav_v11_planner_cbf_pilot_summary.json'
P=json.loads(protocol.read_text())
R=list(csv.DictReader(runs.open()))
expected=len(P['pilot_seeds'])*len(P['participation_grid'])*len(P['policies'])*2
keys=[(x['environment'],x['seed'],x['rho'],x['policy']) for x in R]
numeric=['success','completion_fraction','minimum_pair_clearance_m','minimum_obstacle_clearance_m','control_energy','communication_messages']
finite=all(math.isfinite(float(x[k])) for x in R for k in numeric)
rates={p:{str(r):sum(int(x['success']) for x in R if x['policy']==p and float(x['rho'])==r)/4 for r in P['participation_grid']} for p in P['policies']}
checks={
 'protocol_frozen':P['status']=='FROZEN_DEVELOPMENT_PILOT',
 'row_count':len(R)==expected,
 'unique_keys':len(keys)==len(set(keys)),
 'finite_endpoints':finite,
 'no_heldout_or_physical_claim':'Development-only' in P['claim_boundary'],
 'unchanged_task_thresholds_declared':'physical task thresholds' in P['unchanged'],
 'candidate_success_observed':max(rates['two_layer_heterogeneity_gate'].values())==1.0,
 'stable_finite_window_confirmed':False,
 'cbf_feasibility_residual_certified':False
}
registry={
 'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
 'checks':checks,
 'success_by_policy_rho':rates,
 'status':'DEVELOPMENT_CANDIDATE_NOT_HELDOUT_QUALIFIED',
 'causal_validation_qualified':False,
 'reason':'A candidate success point exists, but the pilot has four trials per policy-rho, no independent heldout contract, and no logged post-saturation CBF residual certificate.'
}
out=HERE/'results/UAV_V11_QUALIFICATION_REGISTRY.json'
out.write_text(json.dumps(registry,indent=2)+'\n')
assert all(v for k,v in checks.items() if k not in {'stable_finite_window_confirmed','cbf_feasibility_residual_certified'})
assert not registry['causal_validation_qualified']
print(json.dumps(registry,indent=2))
