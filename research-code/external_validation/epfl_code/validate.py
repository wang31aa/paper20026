#!/usr/bin/env python3
import hashlib,json,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
record=json.loads((HERE/'source_record.json').read_text())
p=Path(record['local_archive'])
assert p.exists(), 'official EPFL code archive is not present'
h=hashlib.md5(p.read_bytes()).hexdigest()
assert h==record['md5'] and p.stat().st_size==record['bytes']
with zipfile.ZipFile(p) as z:
 names=set(z.namelist())
 required={
  'matlab_code/README.txt','matlab_code/requirements.txt',
  'matlab_code/LICENSE.txt','matlab_code/examples/example_mpc.m',
  'matlab_code/mpc_functions/swarming_model.m',
  'matlab_code/mpc_functions/compute_cost_offline.m',
  'matlab_code/swarming_core/cl_run.m'}
 assert required<=names, sorted(required-names)
 req=z.read('matlab_code/requirements.txt').decode(errors='replace')
 for token in ('Ubuntu 16.04','Matlab >= R2019b','91067daebe12c07d76d32a6aed0b8db00b3a54e1','swarmlab v1.0'):
  assert token in req, token
print(f'PASS: official EPFL source archive {p.stat().st_size} bytes, MD5 {h}; original-stack replay remains open')
