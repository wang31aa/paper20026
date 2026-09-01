#!/usr/bin/env python3
"""Freeze a domain-parameterized critical function before V12 execution."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

R=Path(__file__).resolve().parent
C=json.loads((R/'V12_CONFIRMATORY_CONTRACT.json').read_text())
frames=[]
for rel in C['development_sources']:
    d=pd.read_csv(R/rel)
    frames.append(d[d.policy.eq('all_coupled')])
d=pd.concat(frames,ignore_index=True)

def metrics(y,p):
    tp=((y==1)&(p==1)).sum(); fn=((y==1)&(p==0)).sum()
    tn=((y==0)&(p==0)).sum(); fp=((y==0)&(p==1)).sum()
    se=tp/max(tp+fn,1); sp=tn/max(tn+fp,1)
    return float(se),float(sp),float((se+sp)/2)

out=[]
for domain,g in d.groupby('domain'):
    y=g.task_success.to_numpy(int)
    x=np.c_[1/(g.rho*g.gamma),g.rho*g.heterogeneous_fraction,
            g.rho**2*g.heterogeneous_fraction,np.log(g.n/5)]
    def objective(theta):
        p=(x@theta<=1).astype(int); se,sp,ba=metrics(y,p)
        return -(ba-.25*abs(se-sp))
    fit=differential_evolution(objective,[(0,1),(0,8),(0,5),(0,2)],seed=12015,
                               popsize=18,tol=1e-9,polish=True,workers=1)
    p=(x@fit.x<=1).astype(int); se,sp,ba=metrics(y,p)
    out.append(dict(domain=domain,A_information=float(fit.x[0]),
                    B_mismatch=float(fit.x[1]),C_constraint=float(fit.x[2]),
                    D_scale=float(fit.x[3]),development_sensitivity=se,
                    development_specificity=sp,development_balanced_accuracy=ba,
                    rows=int(len(g))))
payload={'schema_version':'12.0','contract_sha256':hashlib.sha256((R/'V12_CONFIRMATORY_CONTRACT.json').read_bytes()).hexdigest(),
         'critical_function':C['critical_function'],'parameters':out,
         'status':'FROZEN_BEFORE_V12_EXECUTION'}
(R/'V12_FROZEN_PARAMETERS.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
