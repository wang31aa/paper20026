#!/usr/bin/env python3
"""Fail-closed validation of the V9 reference-environment controller study."""
import csv, hashlib, json, math
from collections import Counter
from pathlib import Path

H=Path(__file__).resolve().parent; R=H/'results'
q=json.loads((H/'UAV_V9_REFERENCE_ENVIRONMENT_PROTOCOL.json').read_text())
s=json.loads((R/'uav_v9_reference_summary.json').read_text())
archive=json.loads((R/'archived_nmpc_causal_rollout_qualification.json').read_text())
rows=list(csv.DictReader((R/'uav_v9_reference_runs.csv').open()))
keys=[(r['split'],r['environment'],r['seed'],r['rho'],r['policy']) for r in rows]
blocks=Counter((r['split'],r['environment'],r['seed'],r['rho']) for r in rows)
finite=all(math.isfinite(float(r[k])) for r in rows for k in (
 'completion_fraction','minimum_pair_clearance_m','minimum_obstacle_clearance_m',
 'control_energy','communication_messages','first_failure_time_s','recovery_dwell_s'))
integration=max(x['velocity_transition_rmse'] for x in archive['records'])
checks={
 'protocol_hash':s['protocol_sha256']==hashlib.sha256((H/'UAV_V9_REFERENCE_ENVIRONMENT_PROTOCOL.json').read_bytes()).hexdigest(),
 'row_count':len(rows)==1408,
 'heldout_count':s['heldout_rows']>=q['promotion']['minimum_heldout_runs'],
 'unique_keys':len(keys)==len(set(keys)),
 'complete_policy_blocks':all(n==len(q['policies']) for n in blocks.values()),
 'finite_endpoints':finite,
 'reference_dynamics_reintegration':integration<1e-12,
}
task_pass=bool(s['frozen_feasible_rho']) and s['sensitivity'] is not None and s['specificity'] is not None and s['sensitivity']>=q['promotion']['minimum_sensitivity'] and s['specificity']>=q['promotion']['minimum_specificity']
out={'schema':'UAV-V9-QUALIFICATION-R1','pipeline_integrity_pass':all(checks.values()),'task_prediction_pass':task_pass,'checks':checks,'reference_environment_qualified':checks['reference_dynamics_reintegration'],'new_controller_qualified_for_task_claim':task_pass,'status':'REFERENCE_ENVIRONMENT_QUALIFIED_CONTROLLER_NOT_TASK_QUALIFIED' if all(checks.values()) and not task_pass else ('QUALIFIED' if all(checks.values()) and task_pass else 'PIPELINE_INVALID'),'failure_attribution':'completion was achieved at moderate/high participation, but pair-clearance and obstacle-clearance constraints failed; low participation preserved clearance but did not complete the task','claim_boundary':'new heterogeneous controller in an archive-scaled, dynamics-qualified computational environment; not official controller replay, public-flight counterfactual, HIL or hardware'}
(R/'UAV_V9_QUALIFICATION_REGISTRY.json').write_text(json.dumps(out,indent=2)+'\n')
if not out['pipeline_integrity_pass']:raise SystemExit('FAIL: V9 pipeline integrity')
print(out['status'])
