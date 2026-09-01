#!/usr/bin/env python3
"""V3.1 dimensionally homogeneous calibration and critical-function freeze."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from uav_v3_engine import P, simulate, adjacency

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"; OUT.mkdir(exist_ok=True)
Q=json.loads((HERE/"UAV_V31_PREREGISTRATION.json").read_text())
if set(Q["development_seeds"]) & set(Q["future_heldout_seeds"]): raise SystemExit("seed leakage")
rows=[]
for env in P["development_environments"]:
  for seed in Q["development_seeds"]:
    for rho in Q["rho_grid"]:
      rows.append(simulate("all_coupled_with_trim_sharing",float(rho),env,int(seed),"all_heterogeneous"))
with (OUT/"uav_v31_development_runs.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
q=float(Q["observer_confidence_quantile"])
Vph=float(np.quantile([r["rho"]*r["peak_position_observer_error_m"] for r in rows],q))
Vvh=float(np.quantile([r["rho"]*r["peak_velocity_observer_error_mps"] for r in rows],q))

# Acceleration budgets are kept in acceleration units.  U=0 is the conservative
# value because the implemented feedback does not reveal/allocate a separate
# pure-cancellation reserve.
trim=np.asarray(P["declared_trim_acceleration_mps2"],float)*1.05
D0=float(np.max(np.linalg.norm(trim,axis=1)))
A=adjacency(); Db=float(P["trim_sharing_gain"]*np.max(np.linalg.norm(A@trim,axis=1)))
U=0.0

lams=[1.381966011250105,1.381966011250105,3.618033988749895,3.618033988749895]
plants=list(zip([.88,.94,1.,1.07,1.14],[.04,.06,.08,.10,.12],[.22,.30,.40,.56,.72]))

def gains(rho,lam):
    kp=Q["position_feedback_gain_s-2"]+rho*Q["position_coupling_gain_s-2"]*lam
    kd=Q["velocity_feedback_gain_s-1"]+rho*Q["velocity_coupling_gain_s-1"]*lam
    k0=0.; kpeak=0.; dt=.002; steps=int(40/dt)
    for beta,drag,tau in plants:
        # exact steady position gain for unit additive acceleration forcing
        k0=max(k0,1/(beta*kp))
        p=v=a=0.; peak=0.
        alpha=np.exp(-dt/tau)
        for _ in range(steps):
            u=-kp*p-kd*v; a=alpha*a+(1-alpha)*u
            v+=dt*(beta*a-drag*v+1.0); p+=dt*v; peak=max(peak,abs(p))
        # tail must agree with the analytic DC value to within 2%; otherwise
        # the finite horizon is not accepted as a peak-gain calculation.
        if abs(abs(p)-1/(beta*kp))>0.02*(1/(beta*kp)): raise RuntimeError("step-response horizon too short")
        kpeak=max(kpeak,peak)
    return k0,kpeak

crit=[]; gain_rows=[]
for rho in Q["rho_grid"]:
    branches=[]
    for mode,lam in enumerate(lams):
        K0,Kp=gains(rho,lam); R=max(D0+rho*Db-U,0.)
        direct=Vph/rho
        velocity=Kp*Q["velocity_feedback_gain_s-1"]*Vvh/rho
        peak_m=Kp*R+direct+velocity; ultimate_m=K0*R
        psi=max(peak_m/Q["epsilon_peak_m"],ultimate_m/Q["epsilon_ultimate_m"])
        branches.append((psi,mode,peak_m,ultimate_m)); gain_rows.append({"rho":rho,"mode":mode,"lambda":lam,"K0_s2":K0,"Kpeak_s2":Kp})
    active=max(branches)
    crit.append({"rho":rho,"psi_output":active[0],"active_mode":active[1],"peak_bound_m":active[2],"ultimate_bound_m":active[3],"predicted_feasible":int(active[0]<=1)})
with (OUT/"uav_v31_output_gains.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(gain_rows[0])); w.writeheader(); w.writerows(gain_rows)
with (OUT/"uav_v31_critical_function.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(crit[0])); w.writeheader(); w.writerows(crit)
contract={"schema_version":"3.1","status":"FROZEN_DIMENSIONALLY_CLOSED_BEFORE_HELDOUT",
 "preregistration_sha256":hashlib.sha256((HERE/"UAV_V31_PREREGISTRATION.json").read_bytes()).hexdigest(),
 "parent_v3_contract_sha256":Q["parent_v3_contract_sha256"],"development_seeds":Q["development_seeds"],
 "future_heldout_seed_count_read":0,"Vph_upper_m":Vph,"Vvh_upper_mps":Vvh,"D0_upper_mps2":D0,"Db_upper_mps2":Db,"U_lower_mps2":U,
 "epsilon_peak_m":Q["epsilon_peak_m"],"epsilon_ultimate_m":Q["epsilon_ultimate_m"],"critical_function":crit,"units":Q["units"]}
raw=json.dumps(contract,sort_keys=True,separators=(",",":")).encode(); contract["contract_sha256"]=hashlib.sha256(raw).hexdigest()
(OUT/"UAV_V31_FROZEN_CONTRACT.json").write_text(json.dumps(contract,indent=2)+"\n")
print(json.dumps(contract,indent=2))
