#!/usr/bin/env python3
"""Calibration/held-out robot-domain test of connectivity-safe residual gating."""
from pathlib import Path
import csv, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def load():
    rows=list(csv.DictReader((ROOT/'public_cluster_cases/cache/robot_swarm_validation.csv').open()))
    rows=[r for r in rows if r['run']=='0']; nt=1+max(int(r['step']) for r in rows); n=1+max(int(r['robot_idx']) for r in rows)
    act=np.zeros((nt,n,2)); pos=np.zeros_like(act)
    for r in rows:
        t,i=int(r['step']),int(r['robot_idx']); act[t,i]=[float(r['action_x']),float(r['action_y'])]; pos[t,i]=[float(r['pos_x']),float(r['pos_y'])]
    return pos,act

def scale(x):
    q=np.quantile(x,[.25,.75]); return max(float((q[1]-q[0])/1.349),1e-9)

def expand(x,count):
    old=np.linspace(0,1,len(x)); new=np.linspace(0,1,count); flat=x.reshape(len(x),-1)
    return np.column_stack([np.interp(new,old,flat[:,j]) for j in range(flat.shape[1])]).reshape(count,*x.shape[1:])

def laplacian(active):
    # Source gating: excluded robots still receive but cannot influence others.
    n=len(active); w=np.tile(active,(n,1)); np.fill_diagonal(w,0)
    return np.diag(w.sum(1))-w + 0.5*np.eye(n)

def simulate(profile,policy,thresholds=None,e0=None):
    dt=.004; n=profile.shape[1]; e=np.zeros((n,2)) if e0 is None else e0.copy()
    active=np.ones(n,dtype=bool); over=np.zeros(n); under=np.zeros(n)
    trace=[]
    for k,d in enumerate(profile):
        centre=np.median(e[active],axis=0) if active.any() else np.zeros(2)
        score=np.linalg.norm(e-centre,axis=1)
        if policy=='gated':
            high,low=thresholds
            over=np.where(score>high,over+dt,0); under=np.where(score<low,under+dt,0)
            active[(over>=.08)&(np.arange(n)!=0)]=False
            active[(under>=.32)]=True; active[0]=True
            if active.sum()<3:
                order=np.argsort(score); active[order[:3]]=True
        L=laplacian(active.astype(float))
        e += dt*(-0.6*e-1.8*(L@e)+d)
        trace.append((k*dt,e.copy(),score.copy(),active.copy()))
    return trace,e

def main():
    pos,act=load(); split=int(.6*len(act))
    s=scale((act[:split]-np.median(act[:split],axis=1,keepdims=True)).ravel())
    prof=(act-np.median(act,axis=1,keepdims=True))/s
    cal=expand(prof[:split],1800); test=expand(prof[split:],1200)
    base_cal,e0=simulate(cal,'all')
    scores=np.concatenate([r[2] for r in base_cal]); high=float(np.quantile(scores,.90)); low=float(np.quantile(scores,.60))
    # Fixed core and task tolerance use calibration data only.
    cal_node=np.stack([r[1] for r in base_cal]); variability=np.quantile(np.linalg.norm(cal_node,axis=2),.95,axis=0)
    core=np.argsort(variability)[:3]
    cal_core=np.array([np.max(np.linalg.norm(r[1][core],axis=1)) for r in base_cal])
    tolerance=float(np.quantile(cal_core,.95))
    out=[]; traces={}
    for policy in ('all_coupled','gated'):
        tr,_=simulate(test,'gated' if policy=='gated' else 'all',(high,low),e0)
        err=np.array([np.max(np.linalg.norm(r[1][core],axis=1)) for r in tr]); ok=err<=tolerance; cross=np.flatnonzero(~ok)
        out.append({'policy':policy,'survival_time':float(cross[0]*.004 if cross.size else len(tr)*.004),
                    'within_tolerance_fraction':float(ok.mean()),'heldout_p95_core_error':float(np.quantile(err,.95)),
                    'final_participating':int(tr[-1][3].sum())})
        traces[policy]=[(r[0],err[j],int(r[3].sum())) for j,r in enumerate(tr)]
    res=HERE/'results'; res.mkdir(exist_ok=True)
    with (res/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0]); w.writeheader(); w.writerows(out)
    with (res/'trace.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['policy','time','core_max_error','participating'])
        for p in traces:
            w.writerows((p,*r) for r in traces[p])
    protocol={'source_rows':len(act),'calibration_rows':split,'heldout_rows':len(act)-split,
      'core_nodes':[int(x) for x in core],'forcing_scale':s,'gate_high':high,'gate_low':low,
      'task_tolerance':tolerance,'minimum_participating':3,'target_root_node':0,'outcomes':out}
    (res/'protocol_and_results.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol,indent=2))
if __name__=='__main__': main()
