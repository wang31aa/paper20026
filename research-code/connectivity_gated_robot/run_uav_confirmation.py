#!/usr/bin/env python3
"""Second locked transfer of the unchanged gating rule to a UAV holdout."""
from pathlib import Path
import csv,json
import numpy as np
from run import scale,expand,laplacian
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
def load():
 rows=list(csv.DictReader((ROOT/'public_cluster_cases/cache/uav_swarm_synthetic.csv').open()));nt=1+max(int(r['timestamp']) for r in rows);n=1+max(int(r['drone_id']) for r in rows)
 p=np.zeros((nt,n,3));f=np.zeros((nt,n,1))
 for r in rows:
  t,i=int(r['timestamp']),int(r['drone_id']);p[t,i]=[float(r['x']),float(r['y']),float(r['z'])];f[t,i,0]=float(r['velocity'])
 return p,f
def simulate(profile,policy,thresholds=None,e0=None,v0=None):
 dt=.004;n=profile.shape[1];e=np.zeros((n,1)) if e0 is None else e0.copy();v=np.zeros_like(e) if v0 is None else v0.copy();active=np.ones(n,dtype=bool);over=np.zeros(n);under=np.zeros(n);out=[]
 for k,d in enumerate(profile):
  centre=np.median(e[active],axis=0);score=np.linalg.norm(e-centre,axis=1)
  if policy=='gated':
   hi,lo=thresholds;over=np.where(score>hi,over+dt,0);under=np.where(score<lo,under+dt,0);active[(over>=.08)&(np.arange(n)!=0)]=False;active[(under>=.32)]=True;active[0]=True
   if active.sum()<12:active[np.argsort(score)[:12]]=True
  L=laplacian(active.astype(float));acc=-1.2*e-1.1*v-1.8*(L@e)+d;e+=dt*v;v+=dt*acc;out.append((k*dt,e.copy(),score.copy(),active.copy()))
 return out,e,v
def main():
 state,forcing=load();split=int(.6*len(state));s=scale((forcing[:split]-np.median(forcing[:split],axis=1,keepdims=True)).ravel());profile=(forcing-np.median(forcing,axis=1,keepdims=True))/s
 cal=expand(profile[:split],2400);test=expand(profile[split:],1600);bc,e0,v0=simulate(cal,'all');scores=np.concatenate([r[2] for r in bc]);th=(float(np.quantile(scores,.90)),float(np.quantile(scores,.60)))
 ce=np.stack([r[1] for r in bc]);var=np.quantile(np.linalg.norm(ce,axis=2),.95,axis=0);core=np.argsort(var)[:12];calerr=np.asarray([np.max(np.linalg.norm(r[1][core],axis=1)) for r in bc]);tol=float(np.quantile(calerr,.95));outputs=[];trace=[]
 for policy in ('all_coupled','gated'):
  tr,_,_=simulate(test,'gated' if policy=='gated' else 'all',th,e0,v0);err=np.asarray([np.max(np.linalg.norm(r[1][core],axis=1)) for r in tr]);ok=err<=tol;cross=np.flatnonzero(~ok);outputs.append({'policy':policy,'survival_time':float(cross[0]*.004 if cross.size else len(tr)*.004),'within_tolerance_fraction':float(ok.mean()),'heldout_p95_core_error':float(np.quantile(err,.95)),'final_participating':int(tr[-1][3].sum())});trace.extend((policy,r[0],err[j],int(r[3].sum())) for j,r in enumerate(tr))
 res=HERE/'results';
 with (res/'uav_confirmation_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=outputs[0]);w.writeheader();w.writerows(outputs)
 with (res/'uav_confirmation_trace.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['policy','time','core_max_error','participating']);w.writerows(trace)
 record={'status':'second locked transfer; no threshold changes after robot or vehicle outcomes','source_rows':len(state),'calibration_rows':split,'heldout_rows':len(state)-split,'core_nodes':[int(x) for x in core],'forcing_scale':s,'gate_high':th[0],'gate_low':th[1],'task_tolerance':tol,'outcomes':outputs};(res/'uav_confirmation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
