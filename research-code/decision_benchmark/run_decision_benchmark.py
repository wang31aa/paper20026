#!/usr/bin/env python3
"""Pre-registered synthetic high-fidelity frequency-protection benchmark.

No parameter in this file is fitted to the generated outcomes.  See
PREREGISTRATION.md.  The model is intentionally labelled synthetic.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

N = 5
DT = 0.0025
T_END = 12.0
SAFETY = 0.50                 # Hz, synthetic decision boundary
ALARM = 0.42                  # Hz
CERT_MARGIN = 0.08            # Hz, fixed before simulations
ACTION_DELAY = 0.12           # s
ACTION_FRACTION = 0.72
MIN_HOLD = 1.0
NOISE_SD = 0.003              # Hz
SEEDS = list(range(30))
PHASE_B_SEEDS = list(range(1000,1030))
SCENARIOS = ("step", "ramp", "pulse", "double")
STRATEGIES = ("oracle", "full", "no_model", "uncertified")

# Fixed, connected, heterogeneous synthetic network. Values are per-unit-like,
# selected as a plausible stress-test range, not calibrated to a named grid.
EDGES = ((0,1,7.0),(1,2,5.5),(2,3,6.5),(3,4,5.0),(4,0,4.5),(1,3,3.0))
M = np.array([7.2, 5.8, 6.5, 4.9, 6.0])
D = np.array([1.10, .95, 1.20, .90, 1.05])
RINV = np.array([1.9, 2.1, 1.8, 2.2, 2.0])
TAU = np.array([.38, .44, .35, .48, .40])

def laplacian(scale: float) -> np.ndarray:
    L=np.zeros((N,N))
    for i,j,b in EDGES:
        b*=scale; L[i,i]+=b; L[j,j]+=b; L[i,j]-=b; L[j,i]-=b
    return L

def disturbance(t: float, kind: str, amp: float, node: int) -> np.ndarray:
    p=np.zeros(N)
    if kind=="step" and t>=1: p[node]=-amp
    elif kind=="ramp" and 1<=t<3: p[node]=-amp*(t-1)/2
    elif kind=="ramp" and t>=3: p[node]=-amp
    elif kind=="pulse" and 1<=t<3.2: p[node]=-amp
    elif kind=="double":
        if t>=1: p[node]-=.62*amp
        if t>=2.4: p[(node+2)%N]-=.55*amp
    return p

def rhs(x, p, action, L):
    th=x[:N]; f=x[N:2*N]; q=x[2*N:]
    # protection offsets a known fraction of the instantaneous lost power
    protected=p*(1-ACTION_FRACTION*action)
    return np.r_[2*np.pi*f, (q+protected-D*f-L@th)/M, (-q-RINV*f)/TAU]

def rk4(x, dt, p, action, L):
    k1=rhs(x,p,action,L); k2=rhs(x+dt*k1/2,p,action,L)
    k3=rhs(x+dt*k2/2,p,action,L); k4=rhs(x+dt*k3,p,action,L)
    return x+dt*(k1+2*k2+2*k3+k4)/6

def discrete_plant(dt, L):
    """Exact zero-order-hold discretization of the linear swing plant."""
    A=np.zeros((3*N,3*N)); Bp=np.zeros((3*N,N))
    A[:N,N:2*N]=2*np.pi*np.eye(N)
    A[N:2*N,:N]=-np.diag(1/M)@L
    A[N:2*N,N:2*N]=-np.diag(D/M)
    A[N:2*N,2*N:]=np.diag(1/M)
    A[2*N:,N:2*N]=-np.diag(RINV/TAU)
    A[2*N:,2*N:]=-np.diag(1/TAU)
    Bp[N:2*N,:]=np.diag(1/M)
    aug=np.zeros((4*N,4*N)); aug[:3*N,:3*N]=A; aug[:3*N,3*N:]=Bp
    E=expm(aug*dt)
    return E[:3*N,:3*N],E[:3*N,3*N:]

def case_parameters(seed, kind, phase="A"):
    rng=np.random.default_rng(19073+seed+1000*SCENARIOS.index(kind))
    amp_range=(2.2,5.2) if phase=="A" else (4.5,10.5)
    return dict(amp=float(rng.uniform(*amp_range)), node=int(rng.integers(N)),
                bscale=float(rng.uniform(.82,1.18)))

def simulate(seed, kind, strategy, dt=DT, forced_no_action=False, phase="A", oracle_required=None):
    par=case_parameters(seed,kind,phase); L=laplacian(par["bscale"])
    Phi,Gamma=discrete_plant(dt,L)
    rng=np.random.default_rng(44021+seed+1000*SCENARIOS.index(kind))
    x=np.zeros(3*N); times=np.arange(0,T_END+dt/2,dt)
    active=False; scheduled=np.inf; action_time=np.nan; hold_until=np.inf
    max_abs=0.; unsafe_time=0.; alarm_time=np.nan; max_est=0.
    # distributed observer states: consensus-filtered measured frequency
    z=np.zeros(N); z_prev=np.zeros(N); alpha=13.; beta=8.
    # oracle label is calculated separately using a no-action counterfactual.
    for k,t in enumerate(times):
        p=disturbance(t,kind,par["amp"],par["node"])
        f=x[N:2*N]
        y=f+rng.normal(0,NOISE_SD,N)
        z_prev[:]=z
        z += dt*(-alpha*(z-y)-beta*(L@z))
        est=float(np.max(np.abs(z)))
        trend=float(max(0.,-(np.mean(z)-np.mean(z_prev))/dt))
        # fixed 0.35 s look-ahead proxy; same coefficients in every run
        risk=est+.35*trend
        max_est=max(max_est,risk)
        if not active and not np.isfinite(scheduled) and not forced_no_action:
            trigger=False
            if strategy=="oracle":
                if phase=="B":
                    # held-out information upper comparator: paired future label;
                    # trigger once the disturbance begins, never before it exists.
                    trigger=bool(oracle_required and np.any(p))
                else:
                    oracle_risk=float(np.max(np.abs(f)))+.35*max(0.,-np.mean(rhs(x,p,0.,L)[N:2*N]))
                    trigger=oracle_risk>=ALARM
            elif strategy=="full": trigger=risk+CERT_MARGIN>=SAFETY
            elif strategy=="uncertified": trigger=risk>=SAFETY
            elif strategy=="no_model": trigger=float(np.max(np.abs(y)))>=ALARM
            if trigger:
                alarm_time=t; scheduled=t+ACTION_DELAY; hold_until=scheduled+MIN_HOLD
        if not active and t+1e-12>=scheduled:
            active=True; action_time=t
        # irreversible action within the event horizon (hold is documented minimum)
        x=Phi@x+Gamma@(p*(1-ACTION_FRACTION*(1. if active else 0.)))
        af=float(np.max(np.abs(x[N:2*N])))
        max_abs=max(max_abs,af); unsafe_time += dt*(af>SAFETY)
    return dict(seed=seed,scenario=kind,strategy=strategy,dt=dt,
        amplitude=par["amp"],disturbance_node=par["node"],bscale=par["bscale"],
        action=int(active),alarm_time=None if np.isnan(alarm_time) else alarm_time,
        action_time=None if np.isnan(action_time) else action_time,
        max_abs_frequency=max_abs,unsafe_time=unsafe_time,max_risk_estimate=max_est)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="raw")
    ap.add_argument("--quick",action="store_true"); ap.add_argument("--phase",choices=("A","B"),default="A"); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    seeds=(SEEDS if a.phase=="A" else PHASE_B_SEEDS); seeds=seeds[:4] if a.quick else seeds
    rows=[]
    for kind in SCENARIOS:
      for seed in seeds:
        cf=simulate(seed,kind,"oracle",forced_no_action=True,phase=a.phase)
        required=cf["max_abs_frequency"]>SAFETY
        for strategy in STRATEGIES:
          r=simulate(seed,kind,strategy,phase=a.phase,oracle_required=required); r["counterfactual_max_abs_frequency"]=cf["max_abs_frequency"]
          r["protection_required"]=int(required)
          r["false_alarm"]=int(r["action"] and not required)
          r["missed_violation"]=int((not r["action"] or r["unsafe_time"]>0) and required)
          # fixed utility: avoided unsafe exposure minus actuation and classification costs
          avoided=max(0.,cf["unsafe_time"]-r["unsafe_time"])
          r["utility"]=10*avoided-0.45*r["action"]-2*r["false_alarm"]-15*r["missed_violation"]
          rows.append(r)
    keys=list(rows[0]);
    with (out/"decision_runs.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary={}
    for s in STRATEGIES:
      rr=[r for r in rows if r["strategy"]==s]
      summary[s]={"n":len(rr),"false_alarm_rate":np.mean([r["false_alarm"] for r in rr]),
        "missed_violation_rate":np.mean([r["missed_violation"] for r in rr]),
        "unsafe_case_rate":np.mean([r["unsafe_time"]>0 for r in rr]),
        "mean_unsafe_time":np.mean([r["unsafe_time"] for r in rr]),
        "mean_utility":np.mean([r["utility"] for r in rr]),
        "action_rate":np.mean([r["action"] for r in rr])}
    # dt audit on a fixed, predeclared 16-case subset and all strategies
    audit=[]
    for kind in SCENARIOS:
      for seed in seeds[:4]:
       for s in STRATEGIES:
        req=simulate(seed,kind,"oracle",DT,True,a.phase)["max_abs_frequency"]>SAFETY
        base=simulate(seed,kind,s,DT,phase=a.phase,oracle_required=req); fine=simulate(seed,kind,s,DT/2,phase=a.phase,oracle_required=req)
        audit.append(dict(seed=seed,scenario=kind,strategy=s,coarse_dt=DT,fine_dt=DT/2,
          coarse_max=base["max_abs_frequency"],fine_max=fine["max_abs_frequency"],
          abs_difference=abs(base["max_abs_frequency"]-fine["max_abs_frequency"]),
          action_agreement=int(base["action"]==fine["action"])))
    with (out/"dt_audit.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)
    payload={"label":"synthetic high-fidelity benchmark; not field data","phase":a.phase,
      "frozen_constants":dict(dt=DT,safety=SAFETY,alarm=ALARM,certificate_margin=CERT_MARGIN,
      action_delay=ACTION_DELAY,action_fraction=ACTION_FRACTION,noise_sd=NOISE_SD),
      "summary":summary,"dt_audit":{"n":len(audit),"max_abs_difference":max(x["abs_difference"] for x in audit),
      "action_agreement_rate":np.mean([x["action_agreement"] for x in audit])}}
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
