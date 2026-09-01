#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent
C=json.loads((R/'V11_1_CONFIRMATORY_CONTRACT.json').read_text());P=json.loads((R/'V11_1_FROZEN_PARAMETERS.json').read_text());A=json.loads((R/'results/v11_1_analysis.json').read_text());d=pd.read_csv(R/'results/v11_1_confirmatory_runs.csv')
assert len(d)==5400 and len(d.drop_duplicates(['domain','n','topology','rho','seed','policy']))==5400
assert set(d.domain)==set(C['domains']) and set(d.policy)==set(C['policies'])
assert d.select_dtypes('number').notna().all().all()
assert hashlib.sha256((R/'V11_1_FROZEN_PARAMETERS.json').read_bytes()).hexdigest()==json.loads((R/'results/v11_1_manifest.json').read_text())['parameter_sha256']
assert A['status']=='FROZEN_CONFIRMATORY_FAILURE' and A['passing_domains']==1
print('PASS: 5,400 frozen paired runs; confirmatory failure retained without threshold revision')
