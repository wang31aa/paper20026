#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
import numpy as np

root=Path(__file__).parent; p=root/'results'
r=list(csv.DictReader((p/'raw_runs.csv').open()))
assert len(r)==500
assert len({x['topology'] for x in r})==5
assert len({x['heterogeneity'] for x in r})==5
assert len({x['seed'] for x in r})==20
assert all(x['finite'] in ('True','False') for x in r)
c=list(csv.DictReader((p/'dt_convergence.csv').open()))
assert len(c)==20 and {float(x['dt']) for x in c}=={.008,.004,.002,.001}
z=np.load(p/'topologies.npz')
assert len([k for k in z.files if k.startswith('random_tree_')])==5
m=json.loads((p/'metadata.json').read_text())
assert m['gain_search'] is False and m['state_clipping'] is False
assert m['old_artifacts_read'] is False and m['published_delta_used'] is False
print('PASS: dimensions, provenance flags, topology count, and dt grid validated')
