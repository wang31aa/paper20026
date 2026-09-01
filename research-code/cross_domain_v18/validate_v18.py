#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
C=json.loads((HERE/'V18_FROZEN_CONTRACT.json').read_text()); P=json.loads((HERE/'V18_FROZEN_PREDICTOR.json').read_text())
M=json.loads((OUT/'v18_manifest.json').read_text()); rows=list(csv.DictReader((OUT/'v18_heldout_runs.csv').open()))
expected=len(C['domains'])*len(C['heldout_sizes'])*len(C['heldout_topologies'])*len(C['heldout_rho'])*len(C['heldout_seeds'])*len(C['policies'])
assert len(rows)==expected==M['rows']==M['unique_rows']==1152
assert M['contract_sha256']==hashlib.sha256((HERE/'V18_FROZEN_CONTRACT.json').read_bytes()).hexdigest()
assert M['predictor_sha256']==hashlib.sha256((HERE/'V18_FROZEN_PREDICTOR.json').read_bytes()).hexdigest()

def raw(r):
    p=r['policy']; n=float(r['n']); rho=float(r['rho'])
    return np.asarray([math.log(n/5.),rho,rho*rho,float(r['gamma']),float(r['visible_fraction']),
      float(r['heterogeneous_fraction']),float(p=='independent_tracking'),float(p=='two_layer_gate'),0.,0.])

results={}
for d in C['domains']:
    rr=[r for r in rows if r['domain']==d]; model=P['models'][d]
    mean=np.asarray(model['mean']); scale=np.asarray(model['scale']); coef=np.asarray(model['coefficients'])
    tp=tn=fp=fn=0
    for r in rr:
        z=np.r_[1.,(raw(r)-mean)/scale]; pred=int(1/(1+np.exp(-np.clip(z@coef,-30,30)))>=.5); y=int(r['task_success'])
        tp+=pred==1 and y==1;tn+=pred==0 and y==0;fp+=pred==1 and y==0;fn+=pred==0 and y==1
    sensitivity=tp/(tp+fn) if tp+fn else None; specificity=tn/(tn+fp) if tn+fp else None
    results[d]={'n':len(rr),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'sensitivity':sensitivity,'specificity':specificity,
                'promotion_pass':sensitivity is not None and specificity is not None and sensitivity>=.8 and specificity>=.8}
payload={'schema':'V18-RESULT-1','rows':len(rows),'domains':results,
 'qualified_domain_count':sum(v['promotion_pass'] for v in results.values()),
 'all_eight_qualified':all(v['promotion_pass'] for v in results.values()),
 'interpretation':C['claim_boundary']}
(OUT/'V18_QUALIFICATION_REGISTRY.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
