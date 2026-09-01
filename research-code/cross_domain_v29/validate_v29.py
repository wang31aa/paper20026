#!/usr/bin/env python3
import csv,json,math,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path[:0]=[str(ROOT/'cross_domain_v27')];import run_v27 as b
C=json.loads((R/'V29_OOD_CONTRACT.json').read_text());M=json.loads((ROOT/'cross_domain_v28/V28_FROZEN_CRITICAL_MODELS.json').read_text());rows=list(csv.DictReader((R/'results/v29_runs.csv').open()))
assert len(rows)==5*2*7*3*2
metrics={}
for d,model in M['models'].items():
 z=[x for x in rows if x['domain']==d and x['policy']=='all_coupled'];pred=[];obs=[]
 for x in z:
  n=int(x['n']);rho=float(x['rho']);seed=int(x['seed']);_,_,_,_,v,_,_,_=b.frozen(d,n,rho,seed);sp=float(np.mean([np.ptp(v[k]) for k,_ in b.PARAMS[d]]));mu=float(x['common_metric_mu'])
  raw=np.array([1/(rho*mu),rho*sp,rho*rho*sp,math.log(n/5)]);xs=(raw-np.array(model['feature_mean']))/np.array(model['feature_scale']);score=float(np.dot(np.array(model['coefficients']),np.r_[1,xs]));pred.append(score>=0);obs.append(bool(int(x['task_success'])))
 tp=sum(a and b0 for a,b0 in zip(pred,obs));tn=sum((not a) and (not b0) for a,b0 in zip(pred,obs));p=sum(obs);q=len(obs)-p
 metrics[d]={'n':len(obs),'positives':p,'sensitivity':tp/p if p else None,'specificity':tn/q if q else None,'balanced_accuracy':((tp/p if p else 0)+(tn/q if q else 0))/2 if p and q else None}
out={'status':'PASS_EXECUTION','trajectory_rows':len(rows),'metrics':metrics,'all_domains_meet_0_8':all(v['sensitivity'] is not None and v['specificity'] is not None and v['sensitivity']>=.8 and v['specificity']>=.8 for v in metrics.values()),'claim_boundary':C['claim_boundary']}
(R/'results/V29_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
