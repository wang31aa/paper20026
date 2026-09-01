#!/usr/bin/env python3
"""Reset-aware, dimensionally homogeneous V3.2 output certificate."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from uav_v3_engine import P,simulate
HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
Q=json.loads((HERE/"UAV_V32_PREREGISTRATION.json").read_text()); old=json.loads((OUT/"UAV_V31_FROZEN_CONTRACT.json").read_text())
if set(Q["development_seeds"])&set(Q["future_heldout_seeds"]): raise SystemExit("seed leakage")
rows=[]
for env in P["development_environments"]:
 for seed in Q["development_seeds"]:
  for rho in Q["rho_grid"]: rows.append(simulate("all_coupled_with_trim_sharing",rho,env,seed,"all_heterogeneous"))
with (OUT/"uav_v32_development_runs.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
q=Q["confidence_quantile"]
Ep=float(np.quantile([r["peak_post_reset_position_error_m"] for r in rows],q)); Ev=float(np.quantile([r["peak_post_reset_velocity_error_mps"] for r in rows],q))
Vph=float(np.quantile([r["rho"]*max(r["peak_position_observer_error_m"]-r["peak_post_reset_position_error_m"],0) for r in rows],q))
Vvh=float(np.quantile([r["rho"]*max(r["peak_velocity_observer_error_mps"]-r["peak_post_reset_velocity_error_mps"],0) for r in rows],q))
gains=list(csv.DictReader((OUT/"uav_v31_output_gains.csv").open())); gain={(float(r["rho"]),int(r["mode"])):r for r in gains}
D0=old["D0_upper_mps2"];Db=old["Db_upper_mps2"];U=old["U_lower_mps2"];kv=1.45
crit=[];parts=[]
for rho in Q["rho_grid"]:
 branches=[]
 for mode in range(4):
  g=gain[(rho,mode)];K0=float(g["K0_s2"]);Kp=float(g["Kpeak_s2"]);R=max(D0+rho*Db-U,0)
  force=Kp*R;pos=Ep+Vph/rho;vel=Kp*kv*(Ev+Vvh/rho);peak=force+pos+vel;ultimate=K0*R;psi=max(peak/Q["epsilon_peak_m"],ultimate/Q["epsilon_ultimate_m"])
  branches.append((psi,mode,peak,ultimate));parts.append({"rho":rho,"mode":mode,"forcing_m":force,"position_information_m":pos,"velocity_information_m":vel,"peak_total_m":peak,"ultimate_m":ultimate})
 a=max(branches);crit.append({"rho":rho,"psi_output":a[0],"active_mode":a[1],"peak_bound_m":a[2],"ultimate_bound_m":a[3],"predicted_feasible":int(a[0]<=1)})
with (OUT/"uav_v32_term_breakdown.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(parts[0]));w.writeheader();w.writerows(parts)
with (OUT/"uav_v32_critical_function.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(crit[0]));w.writeheader();w.writerows(crit)
C={"schema_version":"3.2","status":"FROZEN_RESET_AWARE_DIMENSIONALLY_CLOSED_BEFORE_HELDOUT","parent_v31_contract_sha256":Q["parent_v31_contract_sha256"],"future_heldout_seed_count_read":0,"Ep_reset_upper_m":Ep,"Vph_upper_m":Vph,"Ev_reset_upper_mps":Ev,"Vvh_upper_mps":Vvh,"D0_upper_mps2":D0,"Db_upper_mps2":Db,"U_lower_mps2":U,"critical_function":crit,"units":Q["units"]}
raw=json.dumps(C,sort_keys=True,separators=(",",":")).encode();C["contract_sha256"]=hashlib.sha256(raw).hexdigest();(OUT/"UAV_V32_FROZEN_CONTRACT.json").write_text(json.dumps(C,indent=2)+"\n");print(json.dumps(C,indent=2))
