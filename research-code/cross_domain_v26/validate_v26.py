#!/usr/bin/env python3
"""Fail-closed validator for node-level parameter traceability."""
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT/'audit'))
import parameter_flow_audit_v26 as audit
C=json.loads((P/'V26_PARAMETER_TRACE_CONTRACT.json').read_text())
rows=list(csv.DictReader((P/'results/v26_runs.csv').open()))
ledger=list(csv.DictReader((P/'results/v26_node_parameters.csv').open()))
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*(len(C['development_seeds'])+len(C['heldout_seeds']))*len(C['policies'])
assert len(rows)==expected
groups=defaultdict(list)
for r in rows:groups[r['paired_replay_id']].append(r)
assert all({r['policy'] for r in x}==set(C['policies']) for x in groups.values())
assert all(len({r['parameter_sha256'] for r in x})==1 for x in groups.values())

lg=defaultdict(list)
for r in ledger:lg[r['paired_replay_id']].append(r)
assert set(lg)==set(groups)
for block,rr in lg.items():
    params=defaultdict(dict)
    for r in rr:params[r['parameter']][int(r['node'])]=float(r['value'])
    payload={k:[v[i] for i in sorted(v)] for k,v in sorted(params.items())}
    # Re-map motor ledger names to the sampler keys before hashing.
    if rr[0]['domain']=='motor':
        back={'resistance_ratio':'resistance_ratio','inductance_ratio':'inductance_ratio','inertia_ratio':'inertia_ratio','friction_ratio':'friction_ratio','torque_constant_ratio':'torque_constant_ratio'}
        payload={back[k]:v for k,v in payload.items()}
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert h==rr[0]['parameter_sha256']==groups[block][0]['parameter_sha256']
    assert all(np.ptp(v)>0 for v in payload.values())
    audit.range_checks(rr[0]['domain'], {k:np.asarray(v,dtype=float) for k,v in payload.items()})
assert all(r['parameter_unit_class']=='NORMALIZED_RATIO_MODEL_NOT_SI' for r in rows if r['domain']=='motor')

# The trace instrumentation must be outcome-neutral relative to the frozen V25
# parent design.
parent=list(csv.DictReader((ROOT/'cross_domain_v25/results/v25_runs.csv').open()))
key=lambda r:(r['domain'],r['n'],r['topology'],r['rho'],r['seed'],r['policy'])
pa={key(r):r for r in parent}; ch={key(r):r for r in rows}
assert pa.keys()==ch.keys()
outcome_fields=['task_success','minimum_physical_margin','tail_error','control_energy','communication_messages','first_failure_time']
maximum_parent_difference=max(abs(float(pa[k][f])-float(ch[k][f])) for k in pa for f in outcome_fields)
assert maximum_parent_difference==0

effects=audit.finite_difference_checks()
rates={}
for domain in C['domains']:
    rates[domain]={}
    for policy in C['policies']:
        sample=[int(r['task_success']) for r in rows if r['domain']==domain and r['policy']==policy and int(r['seed']) in C['heldout_seeds']]
        rates[domain][policy]=sum(sample)/len(sample)
out={'status':'PASS','trajectory_rows':len(rows),'paired_blocks':len(groups),'node_parameter_rows':len(ledger),'all_policy_parameter_hashes_identical':True,'all_parameter_ranges_pass':True,'all_parameters_have_nonzero_state_effect':True,'minimum_finite_difference_effect':min(effects.values()),'motor_unit_rule_enforced':True,'maximum_parent_outcome_difference':maximum_parent_difference,'heldout_policy_success_rates':rates,'claim_boundary':C['claim_boundary']}
(P/'results/V26_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
