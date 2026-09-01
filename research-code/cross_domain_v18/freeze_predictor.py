#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V18_FROZEN_CONTRACT.json').read_text())
rows=list(csv.DictReader((ROOT/C['development_source']).open()))

def features(r):
    p=r['policy']; t=r['topology']; n=float(r['n']); rho=float(r['rho'])
    return [1.,math.log(n/5.),rho,rho*rho,float(r['gamma']),float(r['visible_fraction']),
            float(r['heterogeneous_fraction']),float(p=='independent_tracking'),
            float(p=='two_layer_gate'),float(t=='random_directed'),float(t=='switching')]

models={}
for domain in C['domains']:
    selected=[r for r in rows if r['domain']==domain and r['policy'] in C['policies']]
    X=np.asarray([features(r) for r in selected]); y=np.asarray([float(r['task_success']) for r in selected])
    mean=X[:,1:].mean(0); scale=X[:,1:].std(0); scale[scale<1e-12]=1
    Z=np.c_[np.ones(len(X)),(X[:,1:]-mean)/scale]; w=np.zeros(Z.shape[1])
    for _ in range(C['prediction']['iterations']):
        s=np.clip(Z@w,-30,30); prob=1/(1+np.exp(-s))
        grad=Z.T@(prob-y)/len(y); grad[1:]+=C['prediction']['ridge']*w[1:]
        w-=C['prediction']['learning_rate']*grad
    models[domain]={'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':w.tolist(),
                    'development_rows':len(selected),'development_prevalence':float(y.mean())}

payload={'schema':'V18-FROZEN-PREDICTOR-1',
         'contract_sha256':hashlib.sha256((HERE/'V18_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
         'development_sha256':hashlib.sha256((ROOT/C['development_source']).read_bytes()).hexdigest(),
         'models':models,'heldout_opened':False}
(HERE/'V18_FROZEN_PREDICTOR.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'domains':len(models),'rows':sum(x['development_rows'] for x in models.values()),'heldout_opened':False}))
