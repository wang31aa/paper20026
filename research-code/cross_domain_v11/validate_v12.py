#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;C=json.loads((R/'V12_CONFIRMATORY_CONTRACT.json').read_text());P=json.loads((R/'V12_FROZEN_PARAMETERS.json').read_text());M=json.loads((R/'results/v12_manifest.json').read_text());A=json.loads((R/'results/v12_analysis.json').read_text());d=pd.read_csv(R/'results/v12_confirmatory_runs.csv')
assert M['contract_sha256']==hashlib.sha256((R/'V12_CONFIRMATORY_CONTRACT.json').read_bytes()).hexdigest()
assert M['parameter_sha256']==hashlib.sha256((R/'V12_FROZEN_PARAMETERS.json').read_bytes()).hexdigest()
assert P['status']=='FROZEN_BEFORE_V12_EXECUTION';assert len(d)==10800 and M['unique_keys']==10800
assert np.isfinite(d.select_dtypes('number').to_numpy()).all();assert set(d.domain)==set(C['domains'])
assert set(d.policy)==set(C['policies']);assert len(A['prediction'])==3
print(f"PASS: 10,800 frozen heterogeneous V12 runs; status={A['status']}; no threshold revision")
