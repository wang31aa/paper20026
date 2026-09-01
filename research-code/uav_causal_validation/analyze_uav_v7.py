#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;O=H/"results";Q=json.loads((H/"UAV_V7_TARGET_ROOTED_PREREGISTRATION.json").read_text());R=list(csv.DictReader((O/"uav_v7_heldout_runs.csv").open()))
numeric=lambda r,k:float(r[k]); lo,mi,hi=Q["witness_rho"].values(); policies=Q["policies"]
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
rows=[]
for rho in (lo,mi,hi):
 for policy in policies:
  x=[r for r in R if numeric(r,"rho")==rho and r["policy"]==policy];k=sum(int(r["task_success"]) for r in x);a,b=wilson(k,len(x));rows.append({"rho":rho,"policy":policy,"n":len(x),"successes":k,"success_rate":k/len(x),"wilson95_low":a,"wilson95_high":b,"median_tail_error_m":float(np.median([numeric(r,"tail_tracking_error_m") for r in x])),"median_target_observer_error_m":float(np.median([numeric(r,"tail_target_observer_error_m") for r in x])),"median_control_energy":float(np.median([numeric(r,"control_energy") for r in x])),"median_communication_messages":float(np.median([numeric(r,"communication_messages") for r in x]))})
with (O/"uav_v7_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
by={(r["environment"],r["seed"],r["rho"],r["policy"]):r for r in R};paired=[]
for env in Q["heldout_environments"]:
 for seed in map(str,Q["heldout_seeds"]):
  a=by[env,seed,str(hi),policies[0]];g=by[env,seed,str(hi),policies[1]];paired.append({"environment":env,"seed":seed,"rho":hi,"gate_minus_all_tail_error_m":numeric(g,"tail_tracking_error_m")-numeric(a,"tail_tracking_error_m"),"gate_minus_all_energy":numeric(g,"control_energy")-numeric(a,"control_energy"),"gate_minus_all_communication":numeric(g,"communication_messages")-numeric(a,"communication_messages")})
with (O/"uav_v7_high_participation_paired_effects.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
print(json.dumps({"summary_rows":len(rows),"paired_rows":len(paired),"median_high_gate_tail_change_m":float(np.median([x["gate_minus_all_tail_error_m"] for x in paired])),"median_high_gate_energy_change":float(np.median([x["gate_minus_all_energy"] for x in paired]))},indent=2))
