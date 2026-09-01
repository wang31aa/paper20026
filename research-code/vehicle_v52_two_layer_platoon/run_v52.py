#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;sys.path.insert(0,str(ROOT/"vehicle_v51_lag_refinement"))
from run_v51 import lag_distance
C=json.loads((H/"V52_FROZEN_CONTRACT.json").read_text());O=H/"results";DT=C["sample_time_s"]
B=np.array(C["braking_limits_mps2"]);TAU=np.array(C["actuator_lags_s"]);G0=C["clearance_m"]

def pair_distance(c,q,tau,D,Bi):
    # Reuse the exact formula after temporarily replacing the module bounds.
    A=Bi-D;E=Bi-q
    if c<0:return 0.0
    if c==0 and q>=D:return 0.0
    def f(t):return c-A*t+E*tau*(1-np.exp(-t/tau))
    lo=1e-12 if c==0 else 0.;hi=max(1.,(c+E*tau)/A+2*tau)
    while f(hi)>0:hi*=2
    for _ in range(32):
        m=(lo+hi)/2
        if f(m)>0:lo=m
        else:hi=m
    T=(lo+hi)/2
    return float(c*T-A*T*T/2+E*tau*(T-tau*(1-np.exp(-T/tau))))

def viable(g,c,q,i):return g+1e-9>=G0+pair_distance(c,q,TAU[i],B[i-1],B[i])
def reduced(g,c,i):return g+1e-9>=G0+max(c,0)**2/(2*(B[i]-B[i-1]))

def simulate(seed,delay,error_bound,policy):
    rng=np.random.default_rng(seed+int(delay*100)+int(error_bound*1000));n=C["vehicles"];steps=int(C["horizon_s"]/DT)
    v=np.array([15.,15.5,16.,16.5])+rng.normal(0,.15,n);q=np.zeros(n);z=np.full(n,v[0]);
    gaps=np.zeros(n-1)
    for i in range(1,n):
        c=v[i]-v[i-1];gaps[i-1]=G0+pair_distance(c,q[i],TAU[i],B[i-1],B[i])+rng.uniform(.2,2.)
    hist=[z.copy() for _ in range(int(round(delay/DT))+1)];safe=True;energy=0;comm=0;obs=[];minmargin=1e9
    for k in range(steps):
        ref=15+1.2*np.sin(.35*k*DT);z[0]=ref;old=hist[max(0,len(hist)-1-int(round(delay/DT)))]
        for i in range(1,n):
            if rng.random()>.08:
                z[i]+=DT*C["observer_gain"]*(old[i-1]-z[i]);comm+=1
        command=np.zeros(n);command[0]=np.clip(max(v[0]-ref,0)*1.2,0,B[0])
        for i in range(1,n):
            base=np.clip(max(v[i]-z[i],0)*1.1,0,B[i]);c_true=v[i]-v[i-1]
            noise=float(rng.uniform(-error_bound,error_bound));c_hat=c_true+noise
            if policy=="two_layer_lag_kernel":
                c_use=max(c_hat+error_bound,0.);q_use=max(q[i]-0.05,0.)
                candidates=np.linspace(base,B[i],17);chosen=B[i]
                for cand in candidates:
                    qn=q_use+DT*(cand-q_use)/TAU[i];cn=c_use+DT*(B[i-1]-q_use);gn=gaps[i-1]-DT*c_use-.5*DT*DT*(B[i-1]-q_use)
                    if viable(gn,cn,qn,i):chosen=float(cand);break
                command[i]=chosen
            elif policy=="unbuffered_kernel":
                command[i]=B[i] if not reduced(gaps[i-1]-DT*c_hat,c_hat,i) else base
            elif policy=="maximum_braking":command[i]=B[i]
            else:command[i]=base
        oldv=v.copy();oldq=q.copy();q+=DT*(command-q)/TAU;v=np.maximum(0,v-DT*oldq)
        for i in range(1,n):gaps[i-1]+=DT*(oldv[i-1]-oldv[i])+.5*DT*DT*(oldq[i]-oldq[i-1])
        margins=[gaps[i-1]-G0 for i in range(1,n)];minmargin=min(minmargin,min(margins));safe &= min(margins)>=-1e-8
        energy+=float(np.sum(command[1:]**2)*DT);obs.append(float(np.sqrt(np.mean((z[1:]-ref)**2))));hist.append(z.copy())
    return {"seed":seed,"delay":delay,"error_bound":error_bound,"policy":policy,"success":int(safe),"minimum_margin_m":minmargin,"energy":energy,"observer_rmse":float(np.mean(obs)),"messages":comm}

def main():
    O.mkdir(exist_ok=True);rows=[simulate(s,d,e,p) for s in C["heldout_seeds"] for d in C["message_delay_s"] for e in C["observer_error_bounds_mps"] for p in C["policies"]]
    summary={p:{"success_rate":float(np.mean([r["success"] for r in rows if r["policy"]==p])),"median_energy":float(np.median([r["energy"] for r in rows if r["policy"]==p])),"median_minimum_margin_m":float(np.median([r["minimum_margin_m"] for r in rows if r["policy"]==p]))} for p in C["policies"]}
    report={"contract_sha256":hashlib.sha256((H/"V52_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"runs":len(rows),"summary":summary,"claim_boundary":C["claim_boundary"]}
    (O/"V52_REPORT.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
