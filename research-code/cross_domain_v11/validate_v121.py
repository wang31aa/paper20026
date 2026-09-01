#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;C=json.loads((R/'V12_1_FAULT_CONFIRMATORY_CONTRACT.json').read_text());M=json.loads((R/'results/v12_1_manifest.json').read_text());A=json.loads((R/'results/v12_1_analysis.json').read_text());d=pd.read_csv(R/'results/v12_1_fault_confirmatory_runs.csv')
assert M['contract_sha256']==hashlib.sha256((R/'V12_1_FAULT_CONFIRMATORY_CONTRACT.json').read_bytes()).hexdigest();assert M['parameter_sha256']==hashlib.sha256((R/'V12_FROZEN_PARAMETERS.json').read_bytes()).hexdigest();assert len(d)==10800 and M['unique_keys']==10800;assert np.isfinite(d.select_dtypes('number').to_numpy()).all();assert set(d.fault)==set(C['fault_rotation']);assert all(v>0 for v in M['fault_counts'].values());assert len(A['prediction'])==3
print(f"PASS: 10,800 V12.1 fault-rotated heterogeneous runs; status={A['status']}; all four fault families present")
