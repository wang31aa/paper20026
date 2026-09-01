#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent; C=json.loads((H/"V51_FROZEN_CONTRACT.json").read_text()); O=H/"results"
G0=C["clearance_m"]; B=C["commanded_max_braking_mps2"]; D=C["leader_max_braking_mps2"]; A=B-D

def closing(t,c,q,tau):
    E=B-q
    return c-A*t+E*tau*(1-math.exp(-t/tau))

def stop_time(c,q,tau):
    if c<0:return 0.0
    # At c=0, insufficient realized braking can first create positive closing
    # speed.  The relevant stopping time is the later positive root.
    if c==0 and q>=D:return 0.0
    lo,hi=0.0,max(1.0,(c+(B-q)*tau)/A+2*tau)
    if c==0:lo=1e-12
    while closing(hi,c,q,tau)>0: hi*=2
    for _ in range(48):
        mid=(lo+hi)/2
        if closing(mid,c,q,tau)>0:lo=mid
        else:hi=mid
    return (lo+hi)/2

def lag_distance(c,q,tau):
    if c<0:return 0.0
    T=stop_time(c,q,tau);E=B-q
    return c*T-A*T*T/2+E*tau*(T-tau*(1-math.exp(-T/tau)))

def reduced_distance(c): return max(c,0)**2/(2*A)
def beta(c,q,tau): return lag_distance(c,q,tau)-reduced_distance(c)
def full_viable(g,c,q,tau): return g+1e-10>=G0+lag_distance(c,q,tau)
def reduced_viable(g,c): return g+1e-10>=G0+reduced_distance(c)

def equivalence():
    rng=np.random.default_rng(20260825); mismatches=0; max_beta=0; min_beta=1e9
    for _ in range(C["equivalence_random_cases"]):
        tau=float(rng.uniform(.08,.9)); c=float(rng.uniform(0,8)); q=float(rng.uniform(0,B)); g=float(rng.uniform(0,40))
        z=beta(c,q,tau); max_beta=max(max_beta,z);min_beta=min(min_beta,z)
        mismatches+=int(full_viable(g,c,q,tau)!=reduced_viable(g-z,c))
    return {"cases":C["equivalence_random_cases"],"mismatches":mismatches,"minimum_beta":min_beta,"maximum_beta":max_beta}

def simulate(seed,tau,policy):
    rng=np.random.default_rng(seed+int(1000*tau));dt=C["sample_time_s"];steps=int(C["horizon_s"]/dt)
    c=float(rng.uniform(.5,6.5));q=float(rng.uniform(0,1.2));g=float(G0+lag_distance(c,q,tau)+rng.uniform(.02,2.0));safe=True;energy=0
    for _ in range(steps):
        d=float(np.clip(rng.normal(1.4,.8),0,D))
        if policy=="lag_kernel":
            # Minimum command on a fixed grid whose worst-case short-step successor remains viable.
            candidates=np.linspace(0,B,17);lo,hi=0,len(candidates)-1
            while lo<hi:
                mid=(lo+hi)//2;cand=float(candidates[mid])
                qn=q+dt*(cand-q)/tau;cn=c+dt*(D-q);gn=g-dt*c-.5*dt*dt*(D-q)
                if full_viable(gn,cn,qn,tau):hi=mid
                else:lo=mid+1
            command=float(candidates[lo])
        elif policy=="reduced_kernel":
            command=B if g<=G0+reduced_distance(c)+0.5 else 0.0
        elif policy=="maximum_braking":command=B
        else:command=0.0
        old_c,old_q=c,q
        qn=old_q+dt*(command-old_q)/tau
        c=old_c+dt*(d-old_q)
        g=g-dt*old_c-.5*dt*dt*(d-old_q)
        q=qn
        energy+=command*command*dt;safe &= g>=G0-1e-9
    return {"seed":seed,"tau":tau,"policy":policy,"success":int(safe),"energy":energy}

def main():
    O.mkdir(exist_ok=True); eq=equivalence(); table=[]
    for tau in C["actuator_lag_s"]:
      for c in C["closing_speed_mps"]:
       for q in C["initial_realized_braking_mps2"]:
        table.append({"tau":tau,"closing":c,"realized_braking":q,"lag_distance":lag_distance(c,q,tau),"reduced_distance":reduced_distance(c),"buffer":beta(c,q,tau)})
    frozen={"contract_sha256":hashlib.sha256((H/"V51_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"policy":"least 17-point command preserving the lag kernel under worst leader braking"}
    fp=O/"V51_FROZEN_POLICY.json";fp.write_text(json.dumps(frozen,indent=2)+"\n")
    policies=("lag_kernel","reduced_kernel","maximum_braking","none")
    rows=[simulate(s,t,p) for s in C["heldout_seeds"] for t in C["actuator_lag_s"] for p in policies]
    summary={p:{"success_rate":float(np.mean([r["success"] for r in rows if r["policy"]==p])),"median_energy":float(np.median([r["energy"] for r in rows if r["policy"]==p]))} for p in policies}
    report={"contract_sha256":frozen["contract_sha256"],"prediction_sha256":hashlib.sha256(fp.read_bytes()).hexdigest(),"equivalence":eq,"grid":table,"heldout_runs":len(rows),"heldout":summary,"claim_boundary":C["claim_boundary"]}
    (O/"V51_REPORT.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
