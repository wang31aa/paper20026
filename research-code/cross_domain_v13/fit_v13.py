#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import differential_evolution
R=Path(__file__).resolve().parent;C=json.loads((R/'V13_FROZEN_CONTRACT.json').read_text());d=pd.read_csv(R/'results/v13_development_runs.csv');d=d[d.policy.eq('all_coupled')]
def met(y,p):
 tp=((y==1)&(p==1)).sum();fn=((y==1)&(p==0)).sum();tn=((y==0)&(p==0)).sum();fp=((y==0)&(p==1)).sum();se=tp/max(tp+fn,1);sp=tn/max(tn+fp,1);return se,sp,(se+sp)/2
out=[]
for domain,g in d.groupby('domain'):
 y=g.task_success.to_numpy(int);x=np.c_[1/(g.rho*g.gamma),g.rho*g.heterogeneous_fraction,g.rho**2*g.heterogeneous_fraction,np.log(g.n/5)]
 def obj(t):
  se,sp,ba=met(y,(x@t<=1).astype(int));return -(ba-.25*abs(se-sp))
 fit=differential_evolution(obj,[(0,1),(0,8),(0,5),(0,2)],seed=13015,popsize=15,tol=1e-8,polish=True)
 se,sp,ba=met(y,(x@fit.x<=1).astype(int));out.append({'domain':domain,'A_information':fit.x[0],'B_mismatch':fit.x[1],'C_constraint':fit.x[2],'D_scale':fit.x[3],'development_sensitivity':se,'development_specificity':sp,'development_balanced_accuracy':ba,'rows':len(g)})
payload={'contract_sha256':hashlib.sha256((R/'V13_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),'critical_function':C['critical_function'],'parameters':out,'status':'FROZEN_BEFORE_CONFIRMATION'}
(R/'V13_FROZEN_PARAMETERS.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
