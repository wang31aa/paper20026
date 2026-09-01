#!/usr/bin/env python3
"""Fail closed unless every platform has genuine physical execution evidence."""
import hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
contract=HERE/'MASTER_FROZEN_CONTRACT.json'
C=json.loads(contract.read_text())
statuses={}
for domain,spec in C['platforms'].items():
 manifest=HERE/'evidence'/domain/'PHYSICAL_RUN_MANIFEST.json'
 if not manifest.exists():
  statuses[domain]={'qualified':False,'status':'NOT_EXECUTED','reason':'missing physical run manifest'}
  continue
 M=json.loads(manifest.read_text())
 required=set(C['shared_design']['required_logs'])
 checks={
  'external_device_connected':bool(M.get('external_device_connected')),
  'node_count':int(M.get('node_count',0))>=spec['minimum_nodes'],
  'required_logs':required.issubset(set(M.get('logged_fields',[]))),
  'applied_input_measured':bool(M.get('applied_input_measured')),
  'heldout_runs':int(M.get('independent_heldout_runs_per_class',0))>=C['shared_design']['minimum_independent_heldout_runs_per_class'],
  'operator_attestation':bool(M.get('operator_attestation')),
  'raw_hash_manifest':bool(M.get('raw_hash_manifest'))
 }
 statuses[domain]={'qualified':all(checks.values()),'status':'QUALIFIED' if all(checks.values()) else 'INCOMPLETE','checks':checks}
out={
 'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),
 'platforms':statuses,
 'qualified_platform_count':sum(x['qualified'] for x in statuses.values()),
 'all_platforms_qualified':all(x['qualified'] for x in statuses.values()),
 'physical_claim_allowed':all(x['qualified'] for x in statuses.values())
}
(HERE/'PLATFORM_QUALIFICATION_REGISTRY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
