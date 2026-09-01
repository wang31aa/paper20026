#!/usr/bin/env python3
"""Freeze domain-coordinate approximations of the task winning sets."""
import csv, glob, hashlib, json, math, sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parent
sys.path[:0]=[str(ROOT/'cross_domain_v27'),str(ROOT/'cross_domain_v28')]
import run_v27 as b
import run_v28 as v

def base(x):
 d=x['domain']; n=int(x['n']); rho=float(x['rho']); seed=int(x['seed'])
 _,_,_,_,p,noise,bias,state=b.frozen(d,n,rho,seed); mu=v.mode(n,0,rho)[2]
 z=[rho,rho*rho,math.log(n),rho*math.log(n),mu,float(np.mean(abs(bias))),
    float(np.max(abs(bias))),float(np.sqrt(np.mean(noise*noise)))]
 for k,_ in b.PARAMS[d]:
  a=p[k]; z += [float(a.mean()),float(a.std()),float(a.min()),float(a.max()),
                 float(np.quantile(a,.25)),float(np.quantile(a,.75))]
 if isinstance(state,tuple): state=state[0]
 z += [float(np.max(abs(state))),float(np.mean(abs(state)))]
 return np.asarray(z)

def expand(z):
 # Physics descriptors plus curvature and selected pairwise couplings.
 return np.r_[z,z*z,[z[i]*z[j] for i in range(len(z)) for j in range(i+1,len(z))]]

def main():
 files=[]
 for version in range(30,36): files += glob.glob(str(ROOT/f'cross_domain_v{version}/results/*runs.csv'))
 rows=[]
 for path in sorted(set(files)):
  for x in csv.DictReader(open(path)):
   if x['policy']=='all_coupled': x['_source']=Path(path).parts[-3]; rows.append(x)
 models={}; diagnostics={}
 for d in ('circuit','water'):
  q=[x for x in rows if x['domain']==d]; X=np.asarray([expand(base(x)) for x in q]); y=np.asarray([int(x['task_success']) for x in q],float)
  mean=X.mean(0); scale=X.std(0); scale[scale<1e-10]=1.; Z=(X-mean)/scale
  # Class-balanced ridge least-squares score; lambda fixed before V37.
  wobs=np.where(y>0,.5/max(y.sum(),1),.5/max((1-y).sum(),1)); A=Z.T@(wobs[:,None]*Z)+.08*np.eye(Z.shape[1]); coef=np.linalg.solve(A,Z.T@(wobs*(2*y-1)))
  score=Z@coef; candidates=np.unique(score); best=None
  for t in candidates:
   pred=score>=t; sens=np.mean(pred[y==1]) if np.any(y==1) else 0; spec=np.mean(~pred[y==0]) if np.any(y==0) else 0
   val=min(sens,spec)
   if best is None or (val,sens+spec)>(best[0],best[1]): best=(val,sens+spec,float(t),float(sens),float(spec))
  models[d]={'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'threshold':best[2],
             'n':len(y),'positives':int(y.sum()),'feature_contract':'complete domain descriptors; quadratic lift'}
  diagnostics[d]={'development_sensitivity':best[3],'development_specificity':best[4]}
 out={'schema_version':'36.0','frozen_before_v37':True,'model_class':'domain-coordinate quadratic approximation of the system-specific task winning set','models':models,'diagnostics':diagnostics,'claim_boundary':'computational approximation subordinate to the exact set-valued viability theorem; not a universal scalar law and not physical validation'}
 raw=json.dumps(out,sort_keys=True,separators=(',',':'))+'\n'; (R/'V36_FROZEN_MODEL.json').write_text(raw)
 print(json.dumps({'sha256':hashlib.sha256(raw.encode()).hexdigest(),'diagnostics':diagnostics},indent=2))

if __name__=='__main__': main()
