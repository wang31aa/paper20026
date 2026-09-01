#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,sys
R=Path(__file__).resolve().parent;ROOT=R.parent
sys.path[:0]=[str(ROOT/'cross_domain_v26'),str(ROOT/'cross_domain_v27')]
import run_v26 as v26
import run_v27 as v27
C=json.loads((R/'V42_PARAMETER_LEDGER_CORRECTION_CONTRACT.json').read_text());src=ROOT/'cross_domain_v41/results/v41_all_policy_runs.csv';manifest=ROOT/'cross_domain_v41/results/v41_manifest.json'
assert hashlib.sha256(src.read_bytes()).hexdigest()==C['source_v41_runs_sha256'];assert hashlib.sha256(manifest.read_bytes()).hexdigest()==C['source_v41_manifest_sha256']
rows=list(csv.DictReader(src.open()));main={'uav6dof','vehicle','motor'};cache={}
for x in rows:
 key=(x['domain'],int(x['n']),float(x['rho']),int(x['seed']))
 if key not in cache:
  d,n,rho,seed=key
  if d in main:_,h=v26.parameters(d,n,'certified_switching',rho,seed)
  else:*_,p,noise,bias,state=v27.frozen(d,n,rho,seed);h=v27.digest(d,n,seed,p)
  cache[key]=h
 x['parameter_sha256']=cache[key];x['protocol_pair_id']=x.pop('v40_pair_id')
O=R/'results';O.mkdir(exist_ok=True);fields=sorted({k for x in rows for k in x})
with (O/'v42_all_policy_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
out={'status':'PASS','source_rows':len(rows),'parameter_blocks':len(cache),'source_v41_runs_sha256':C['source_v41_runs_sha256'],'v42_runs_sha256':hashlib.sha256((O/'v42_all_policy_runs.csv').read_bytes()).hexdigest(),'trajectory_endpoints_changed':False,'claim_boundary':C['claim_boundary']};(O/'v42_manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
