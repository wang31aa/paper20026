#!/usr/bin/env python3
"""Fail-closed audit for the future eight-domain OOD kernel experiment."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
contract_path=HERE/'V16_OOD_KERNEL_CONTRACT.json'
contract=json.loads(contract_path.read_text())
evidence=json.loads((ROOT/'cross_domain_literature_v15/DOMAIN_EVIDENCE_REGISTRY.json').read_text())
physical=json.loads((ROOT/'physical_platform_closure/PLATFORM_QUALIFICATION_REGISTRY.json').read_text())

rows={}
for domain in contract['domains']:
    item=evidence['domains'][domain]
    source_model_and_bounds=item['parameter_identification_status']=='FULL'
    physical_task_endpoint=any(
        key in ' '.join(item['source_qualified']).lower()
        for key in ('collision endpoint','failure endpoint','task completion endpoint'))
    # No retained platform manifest currently closes observer-controller-gate
    # intervention in any of the eight domain computations.
    paired_closed_loop_intervention=False
    predecessor_enclosure=False
    heldout_kernel_prediction=False
    qualified=all((source_model_and_bounds,physical_task_endpoint,
                   paired_closed_loop_intervention,predecessor_enclosure,
                   heldout_kernel_prediction))
    rows[domain]={
        'source_model_and_bounds':source_model_and_bounds,
        'physical_task_endpoint':physical_task_endpoint,
        'paired_closed_loop_intervention':paired_closed_loop_intervention,
        'certified_predecessor_enclosure':predecessor_enclosure,
        'heldout_kernel_prediction':heldout_kernel_prediction,
        'status':'OOD_KERNEL_QUALIFIED' if qualified else 'NOT_QUALIFIED',
        'permitted_current_label':item['permitted_simulation_label']}

payload={
    'schema':'CROSS-DOMAIN-KERNEL-V16-QUALIFICATION-1',
    'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    'domains':rows,
    'qualified_domain_count':sum(v['status']=='OOD_KERNEL_QUALIFIED' for v in rows.values()),
    'eight_domain_ood_claim_allowed':all(v['status']=='OOD_KERNEL_QUALIFIED' for v in rows.values()),
    'physical_platform_claim_allowed':physical['physical_claim_allowed'],
    'interpretation':'This is a prerequisite audit, not an execution result.'}
(HERE/'V16_OOD_KERNEL_QUALIFICATION.json').write_text(json.dumps(payload,indent=2)+'\n')
assert payload['qualified_domain_count']==0
assert payload['eight_domain_ood_claim_allowed'] is False
assert payload['physical_platform_claim_allowed'] is False
print('PASS: V16 qualification is fail-closed; 0/8 domains currently qualify for an OOD kernel claim')
