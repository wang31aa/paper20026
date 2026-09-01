#!/usr/bin/env python3
"""Execute the pre-frozen V12 heldout matrix without parameter fitting."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import run_uav_v9_reference_controller as base
import run_uav_v11_planner_cbf_pilot as ctl

HERE=Path(__file__).resolve().parent
contract=HERE/'UAV_V12_FROZEN_HELDOUT_CONTRACT.json'
P=json.loads(contract.read_text())
base.shape_reference=ctl.shape_reference
base.barrier=ctl.barrier
envs=[x for x in base.ENVS['heldout'] if x[0] in P['heldout_environments']]
rows=[]
for env in envs:
 for seed in P['heldout_seeds']:
  for rho in P['rho_witnesses'].values():
   for policy in P['policies']:
    rows.append(base.run(policy,float(rho),env,int(seed)))
outpath=base.OUT/'uav_v12_frozen_heldout_runs.csv'
with outpath.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
primary=P['primary_policy']
rates={name:float(np.mean([x['success'] for x in rows if x['policy']==primary and x['rho']==rho])) for name,rho in P['rho_witnesses'].items()}
pred=P['frozen_predictions']
checks={
 'low':rates['low']<=pred['low_success_rate_max'],
 'middle':rates['middle']>=pred['middle_success_rate_min'],
 'high':rates['high']<=pred['high_success_rate_max'],
 'ordering':rates['middle']>rates['low'] and rates['middle']>rates['high']
}
summary={'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'rows':len(rows),'primary_success_rates':rates,'prediction_checks':checks,'qualified':all(checks.values()),'evidence_class':P['claim']}
(base.OUT/'uav_v12_frozen_heldout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
