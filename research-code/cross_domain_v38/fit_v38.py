#!/usr/bin/env python3
"""Freeze a batch-robust domain-coordinate viability classifier."""
import csv,glob,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path.insert(0,str(ROOT/'cross_domain_v36'));import fit_v36 as f
def main():
 files=[]
 for version in range(30,38):files+=glob.glob(str(ROOT/f'cross_domain_v{version}/results/*runs.csv'))
 rows=[]
 for path in sorted(set(files)):
  for x in csv.DictReader(open(path)):
   if x['policy']=='all_coupled':x['_batch']=Path(path).parts[-3];rows.append(x)
 models={};diag={}
 for d in ('circuit','water'):
  q=[x for x in rows if x['domain']==d];X=np.asarray([f.expand(f.base(x)) for x in q]);y=np.asarray([int(x['task_success']) for x in q]);mean=X.mean(0);scale=X.std(0);scale[scale<1e-10]=1;Z=(X-mean)/scale
  wo=np.where(y,.5/max(y.sum(),1),.5/max((1-y).sum(),1));coef=np.linalg.solve(Z.T@(wo[:,None]*Z)+.08*np.eye(Z.shape[1]),Z.T@(wo*(2*y-1)));score=Z@coef
  groups=[]
  for g in sorted(set(x['_batch'] for x in q)):
   ii=np.array([x['_batch']==g for x in q]);p=int(y[ii].sum());n=int(ii.sum()-p)
   if p>=20 and n>=20:groups.append((g,ii))
  best=None
  for t in np.unique(score):
   vals=[]
   for _,ii in groups:
    yy=y[ii];pp=score[ii]>=t;vals += [float(pp[yy==1].mean()),float((~pp[yy==0]).mean())]
   pp=score>=t;vals += [float(pp[y==1].mean()),float((~pp[y==0]).mean())]
   key=(min(vals),sum(vals)/len(vals))
   if best is None or key>best[:2]:best=(key[0],key[1],float(t),vals)
  models[d]={'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'threshold':best[2],'n':len(y),'positives':int(y.sum()),'eligible_batches':[g for g,_ in groups],'feature_contract':'complete domain descriptors; quadratic lift'}
  diag[d]={'worst_group_metric':best[0],'mean_group_metric':best[1]}
 out={'schema_version':'38.0','frozen_before_v39':True,'model_class':'batch-robust domain-coordinate quadratic approximation of the system-specific task winning set','models':models,'diagnostics':diag,'claim_boundary':'computational approximation subordinate to set-valued viability theory; failed V29/V32/V33/V35/V37 rounds retained'};raw=json.dumps(out,sort_keys=True,separators=(',',':'))+'\n';(R/'V38_FROZEN_MODEL.json').write_text(raw);print(json.dumps({'sha256':hashlib.sha256(raw.encode()).hexdigest(),'diagnostics':diag},indent=2))

if __name__=='__main__': main()
