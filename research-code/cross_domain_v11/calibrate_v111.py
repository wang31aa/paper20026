#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np, pandas as pd
R=Path(__file__).resolve().parent
d=pd.read_csv(R/'results/v11_heldout_runs.csv')
d=d[d.policy.eq('all_coupled')].copy()
out=[]
for domain,g in d.groupby('domain'):
    y=g.task_success.to_numpy(int); eta=g.eta.to_numpy(float); rho=g.rho.to_numpy(float); hf=g.heterogeneous_fraction.to_numpy(float)
    best=None
    for A in np.geomspace(1e-5,.2,90):
      for B in np.geomspace(.01,8,90):
        psi=A/eta+B*rho*hf; pred=psi<=1
        tp=((pred==1)&(y==1)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum(); fp=((pred==1)&(y==0)).sum()
        sens=tp/max(tp+fn,1);spec=tn/max(tn+fp,1);score=(sens+spec)/2
        cand=(score,-fn,A,B,sens,spec)
        if best is None or cand>best:best=cand
    _,_,A,B,sens,spec=best
    out.append({'domain':domain,'A_information':A,'B_mismatch':B,'development_sensitivity':sens,'development_specificity':spec})
p={'schema_version':'11.1','source_v11_sha256':hashlib.sha256((R/'results/v11_heldout_runs.csv').read_bytes()).hexdigest(),'critical_function':'Psi=A/(rho*gamma)+B*rho*f_het; feasible iff Psi<=1','parameters':out,'status':'FROZEN_BEFORE_V11_1'}
(R/'V11_1_FROZEN_PARAMETERS.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps(p,indent=2))
