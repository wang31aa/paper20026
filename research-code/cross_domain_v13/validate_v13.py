#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent;C=json.loads((R/'V13_FROZEN_CONTRACT.json').read_text());P=json.loads((R/'V13_FROZEN_PARAMETERS.json').read_text());A=json.loads((R/'results/v13_analysis.json').read_text());errors=[]
if P['contract_sha256']!=hashlib.sha256((R/'V13_FROZEN_CONTRACT.json').read_bytes()).hexdigest():errors.append('contract hash mismatch')
for phase in ('development','confirmation'):
 d=pd.read_csv(R/f'results/v13_{phase}_runs.csv');expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*len(C[phase+'_seeds'])*len(C['policies'])
 if len(d)!=expected:errors.append(f'{phase} rows {len(d)} != {expected}')
 keys=['domain','n','topology','rho','seed','policy']
 if d.duplicated(keys).any():errors.append(f'{phase} duplicate keys')
 if not set(d.domain)==set(C['domains']):errors.append(f'{phase} domain omission')
 if not (d.parameter_spread.map(lambda x:all(float(v)>0 for v in json.loads(x).values()))).all():errors.append(f'{phase} at least one declared heterogeneous parameter has zero spread')
 a=d[d.policy.eq('all_coupled')].set_index(keys[:-1]);b=d[d.policy.eq('two_layer_gate')].set_index(keys[:-1])
 if not a.index.equals(b.index):errors.append(f'{phase} unpaired conditions')
if len(A['prediction'])!=len(C['domains']):errors.append('analysis omitted domain')
I=json.loads((R/'results/v13_feature_identifiability.json').read_text())
if I['groups']!=700 or I['complete_five_domain_groups']!=700:errors.append('feature-fibre coverage incomplete')
if I['mixed_outcome_groups']<1:errors.append('claimed feature non-identifiability lacks a mixed fibre')
print('PASS' if not errors else 'FAIL: '+'; '.join(errors));sys.exit(bool(errors))
