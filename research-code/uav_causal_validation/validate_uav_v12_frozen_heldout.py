#!/usr/bin/env python3
"""Fail-closed validation of the pre-frozen V12 heldout experiment."""
import csv, hashlib, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
contract=HERE/'UAV_V12_FROZEN_HELDOUT_CONTRACT.json'
runs=HERE/'results/uav_v12_frozen_heldout_runs.csv'
P=json.loads(contract.read_text())
R=list(csv.DictReader(runs.open()))
expected=len(P['heldout_environments'])*len(P['heldout_seeds'])*len(P['rho_witnesses'])*len(P['policies'])
keys=[(x['environment'],x['seed'],x['rho'],x['policy']) for x in R]
numeric=['success','completion_fraction','minimum_pair_clearance_m','minimum_obstacle_clearance_m','control_energy','communication_messages']
finite=all(math.isfinite(float(x[k])) for x in R for k in numeric)
primary=P['primary_policy']
rates={name:sum(int(x['success']) for x in R if x['policy']==primary and float(x['rho'])==rho)/(len(P['heldout_environments'])*len(P['heldout_seeds'])) for name,rho in P['rho_witnesses'].items()}
F=P['frozen_predictions']
prediction={
 'low':rates['low']<=F['low_success_rate_max'],
 'middle':rates['middle']>=F['middle_success_rate_min'],
 'high':rates['high']<=F['high_success_rate_max'],
 'ordering':rates['middle']>rates['low'] and rates['middle']>rates['high']
}
checks={
 'contract_hash_matches_execution':hashlib.sha256(contract.read_bytes()).hexdigest()=='dc61db4d1278aa070f87323fa02310441478982e03b5aab9a8e9442330dca7ba',
 'complete_matrix':len(R)==expected,
 'unique_keys':len(keys)==len(set(keys)),
 'finite_endpoints':finite,
 'prediction_passed':all(prediction.values())
}
registry={
 'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),
 'checks':checks,
 'primary_success_rates':rates,
 'prediction_checks':prediction,
 'status':'FROZEN_HELDOUT_PREDICTION_FAILED',
 'causal_validation_qualified':False,
 'interpretation':'Low-participation failure and middle feasibility transferred, but the preregistered high-participation failure did not; the heldout high point succeeded in all ten trials.',
 'claim_boundary':P['claim']
}
(HERE/'results/UAV_V12_QUALIFICATION_REGISTRY.json').write_text(json.dumps(registry,indent=2)+'\n')
assert all(v for k,v in checks.items() if k!='prediction_passed')
assert not checks['prediction_passed'] and not registry['causal_validation_qualified']
print(json.dumps(registry,indent=2))
