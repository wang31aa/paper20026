#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V19_FROZEN_CONTRACT.json').read_text())

def vector(r):
    policies=C['policies']; tops=['chain','ring','random_directed','clustered','switching']
    faults=['development','delay','packet_loss','mixed','sensor_bias','actuator_loss']
    base=[math.log(float(r['n'])),float(r['rho']),float(r['rho'])**2,float(r['gamma']),
          float(r['visible_fraction']),float(r['heterogeneous_fraction']),
          float(r.get('minimum_parameter_spread') or 0.)]
    return base+[float(r['policy']==x) for x in policies[1:]]+[
        float(r['topology']==x) for x in tops[1:]]+[float(r['fault']==x) for x in faults[1:]]

rows=[]; hashes={}
for rel in C['development_sources']:
    p=ROOT/rel; hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.extend(csv.DictReader(p.open()))
models={}
for d in C['domains']:
    rr=[r for r in rows if r['domain']==d and r['policy'] in C['policies']]
    X=np.asarray([vector(r) for r in rr]); y=np.asarray([int(r['task_success']) for r in rr])
    mean=X.mean(0); scale=X.std(0); scale[scale<1e-12]=1
    Z=(X-mean)/scale
    models[d]={'mean':mean.tolist(),'scale':scale.tolist(),'x':Z.tolist(),'y':y.tolist(),
               'rows':len(rr),'prevalence':float(y.mean())}
payload={'schema':'V19-FROZEN-PREDICTOR-1','contract_sha256':hashlib.sha256((HERE/'V19_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
         'development_sha256':hashes,'models':models,'heldout_opened':False}
(HERE/'V19_FROZEN_PREDICTOR.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
print(json.dumps({'domains':len(models),'rows':sum(m['rows'] for m in models.values()),'heldout_opened':False}))
