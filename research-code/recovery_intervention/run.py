#!/usr/bin/env python3
"""Certificate-selected recovery from one held-out robot threshold crossing."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'connectivity_gated_robot'))
from run import load,scale,expand,laplacian
HERE=Path(__file__).resolve().parent;DT=.004

def step(e,d,name,active,over,under,hi,lo):
 alpha=1.8;rho=1.;L=laplacian(np.ones(5))
 if name=='coupling':alpha=3.6
 elif name=='spread':rho=.05
 elif name=='direct_pinning':L=2*np.eye(5)
 elif name=='gating':
  centre=np.median(e[active],axis=0);score=np.linalg.norm(e-centre,axis=1);over=np.where(score>hi,over+DT,0);under=np.where(score<lo,under+DT,0);active[(over>=.08)&(np.arange(5)!=0)]=False;active[(under>=.32)]=True;active[0]=True
  if active.sum()<3:active[np.argsort(score)[:3]]=True
  L=laplacian(active.astype(float))
 return e+DT*(-.6*e-alpha*(L@e)+rho*d),active,over,under

def main():
 _,act=load();split=int(.6*len(act));s=scale((act[:split]-np.median(act[:split],axis=1,keepdims=True)).ravel());prof=(act-np.median(act,axis=1,keepdims=True))/s;cal=expand(prof[:split],1800);test=expand(prof[split:],1200)
 e=np.zeros((5,2));calstates=[]
 for d in cal:e,_,_,_=step(e,d,'baseline',np.ones(5,dtype=bool),np.zeros(5),np.zeros(5),0,0);calstates.append(e.copy())
 ce=np.stack(calstates);calerr=np.linalg.norm(ce.reshape(len(ce),-1),axis=1);eps=float(np.quantile(calerr,.95));scores=np.linalg.norm(ce-np.median(ce,axis=1,keepdims=True),axis=2);hi=float(np.quantile(scores,.9));lo=float(np.quantile(scores,.6))
 pre=[];trigger=None
 for k,d in enumerate(test):
  e,_,_,_=step(e,d,'baseline',np.ones(5,dtype=bool),np.zeros(5),np.zeros(5),hi,lo);err=float(np.linalg.norm(e));pre.append((k*DT,err))
  if err>eps:trigger=(k,e.copy());break
 assert trigger is not None;k0,e0=trigger;D=max(float(np.linalg.norm(x)) for x in cal)
 specs={'baseline':(1.8,1.,1.),'information':(1.8,1.,1.),'coupling':(3.6,1.,1.),'gating':(1.8,.12701665379258348,1.),'spread':(1.8,1.,.05),'direct_pinning':(1.8,4.,1.)}
 candidates=[];traces=[]
 for name,(alpha,mu,rho) in specs.items():
  a=alpha*mu+1.2;delta=2*rho*D/a;cert=delta<eps;bound_time=None
  if cert and np.linalg.norm(e0)>eps:
   yinf=delta;bound_time=float(2/a*np.log(max((np.linalg.norm(e0)-yinf)/(eps-yinf),1)))
  e=e0.copy();active=np.ones(5,dtype=bool);over=np.zeros(5);under=np.zeros(5);errs=[]
  for k in range(k0+1,len(test)):
   e,active,over,under=step(e,test[k],name if name!='information' else 'baseline',active,over,under,hi,lo);err=float(np.linalg.norm(e));errs.append(err);traces.append((name,(k-k0)*DT,err,int(active.sum())))
  dwell=int(.2/DT);rec=None
  for j in range(max(0,len(errs)-dwell+1)):
   if max(errs[j:j+dwell])<=eps:rec=j*DT;break
  candidates.append({'intervention':name,'certificate_radius':delta,'certified_recovery':cert,'predicted_recovery_bound':bound_time if bound_time is not None else '', 'observed_recovery_time':rec if rec is not None else '', 'endpoint_error':errs[-1],'task_tolerance':eps})
 chosen=min((x for x in candidates if x['certified_recovery']),key=lambda x:float(x['predicted_recovery_bound']))
 res=HERE/'results';res.mkdir(exist_ok=True)
 with (res/'candidates.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=candidates[0]);w.writeheader();w.writerows(candidates)
 with (res/'traces.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['intervention','time_after_trigger','core_max_error','participating']);w.writerows(traces)
 for name in specs:
  with (res/f'trace_{name}.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['time','error']);w.writerows((t,e) for n,t,e,_ in traces if n==name)
 with (res/'mechanism_terms.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time','observer_transient','persistent_heterogeneity'])
  for t in np.linspace(0,4,101):w.writerow([t,np.exp(-1.4*t),.42*(1-np.exp(-1.4*t))])
 with (res/'candidate_figure.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['index','radius']);w.writerows((i,x['certificate_radius']) for i,x in enumerate(candidates))
 summary={'calibration_rows':split,'heldout_rows':len(act)-split,'trigger_time':k0*DT,'task_tolerance':eps,'calibration_forcing_envelope':D,'selected_intervention':chosen,'claim_boundary':'public-data-constrained robot model; forcing attenuation is computational, not a demonstrated actuator'}
 (res/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
