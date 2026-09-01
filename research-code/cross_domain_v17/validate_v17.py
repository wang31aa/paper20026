#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
C=json.loads((HERE/'V17_FROZEN_CONTRACT.json').read_text())
M=json.loads((OUT/'v17_manifest.json').read_text()); A=json.loads((OUT/'v17_analysis.json').read_text())
rows=list(csv.DictReader((OUT/'v17_runs.csv').open()))
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*len(C['heldout_seeds'])*len(C['policies'])
assert M['contract_sha256']==hashlib.sha256((HERE/'V17_FROZEN_CONTRACT.json').read_bytes()).hexdigest()
assert len(rows)==expected==M['rows']==M['unique_rows']==5400
assert all(float(r['minimum_parameter_spread'])>0 for r in rows)
assert all(r['evidence_class']=='domain_specific_heterogeneous_computational_intervention' for r in rows)
assert all(A['checks'].values())
qualification={
    'schema':'V17-QUALIFICATION-1',
    'contract_integrity':True,
    'row_and_pair_integrity':True,
    'explicit_dynamical_heterogeneity':True,
    'computational_intervention_qualified':True,
    'source_model_reproduction_qualified':False,
    'hil_qualified':False,
    'physical_experiment_qualified':False,
    'universal_policy_dominance_supported':False,
    'reason':'All frozen computational checks pass; source identification, HIL and physical execution are outside this matrix, and retained outcomes include null and adverse effects.'
}
(OUT/'V17_QUALIFICATION_REGISTRY.json').write_text(json.dumps(qualification,indent=2)+'\n')
print('PASS: 5,400 frozen V17 heterogeneous computational interventions; all pairs and parameter spreads verified')
