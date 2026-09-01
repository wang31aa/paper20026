#!/usr/bin/env python3
import json,sys
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent;C=json.loads((R/'V14_BASELINE_CONTRACT.json').read_text());d=pd.read_csv(R/'results/v14_baseline_runs.csv');errors=[]
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*len(C['seeds'])*len(C['policies'])
keys=['domain','n','topology','rho','seed','policy']
if len(d)!=expected:errors.append('row count')
if d.duplicated(keys).any():errors.append('duplicate keys')
if set(d.policy)!=set(C['policies']):errors.append('policy omission')
counts=d.groupby(keys[:-1]).policy.nunique()
if not (counts==len(C['policies'])).all():errors.append('incomplete pairing')
print('PASS' if not errors else 'FAIL: '+'; '.join(errors));sys.exit(bool(errors))
