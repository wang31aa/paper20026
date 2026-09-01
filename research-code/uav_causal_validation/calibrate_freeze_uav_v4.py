#!/usr/bin/env python3
"""Calibrate and freeze V4 with full-reset observer and explicit reserve."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from uav_v3_engine import P,simulate,adjacency
H=Path(__file__).resolve().parent;O=H/"results";Q=json.loads((H/"UAV_V4_PREREGISTRATION.json").read_text())
if set(Q["development_seeds"])&set(Q["heldout_seeds"]):raise SystemExit("seed leakage")
rows=[]
for env in P["development_environments"]:
 for seed in Q["development_seeds"]:
  for rho in Q["rho_grid"]:rows.append(simulate("connectivity_constrained_observer_gate",rho,env,seed,"all_heterogeneous",Q["observer_mode"],Q["command_reserve_mps2"]))
with (O/"uav_v4_development_runs.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
quant=.95;Ep=float(np.quantile([r["peak_post_reset_position_error_m"] for r in rows],quant));Ev=float(np.quantile([r["peak_post_reset_velocity_error_mps"] for r in rows],quant));Vph=float(np.quantile([r["rho"]*max(r["peak_position_observer_error_m"]-r["peak_post_reset_position_error_m"],0) for r in rows],quant));Vvh=float(np.quantile([r["rho"]*max(r["peak_velocity_observer_error_mps"]-r["peak_post_reset_velocity_error_mps"],0) for r in rows],quant))
trim=np.asarray(P["declared_trim_acceleration_mps2"],float)*1.05;D0=float(np.max(np.linalg.norm(trim,axis=1)));Db=float(P["trim_sharing_gain"]*np.max(np.linalg.norm(adjacency()@trim,axis=1)));U=Q["certified_acceleration_reserve_mps2"]
gain={(float(r["rho"]),int(r["mode"])):r for r in csv.DictReader((O/"uav_v31_output_gains.csv").open())};crit=[];parts=[]
for rho in Q["rho_grid"]:
 bs=[]
 for m in range(4):
  g=gain[(rho,m)];K0=float(g["K0_s2"]);Kp=float(g["Kpeak_s2"]);R=max(D0+rho*Db-U,0);pos=Ep+Vph/rho;vel=Kp*1.45*(Ev+Vvh/rho);force=Kp*R;peak=force+pos+vel;ult=K0*R;psi=max(peak/.14,ult/.14);bs.append((psi,m,peak,ult));parts.append({"rho":rho,"mode":m,"force_m":force,"position_info_m":pos,"velocity_info_m":vel,"peak_m":peak,"ultimate_m":ult})
 a=max(bs);crit.append({"rho":rho,"psi":a[0],"active_mode":a[1],"peak_bound_m":a[2],"ultimate_bound_m":a[3],"predicted_feasible":int(a[0]<=1)})
with (O/"uav_v4_term_breakdown.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(parts[0]));w.writeheader();w.writerows(parts)
with (O/"uav_v4_critical_function.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(crit[0]));w.writeheader();w.writerows(crit)
feas=[r for r in crit if r["predicted_feasible"]];low=[r for r in crit if not r["predicted_feasible"] and feas and r["rho"]<min(x["rho"] for x in feas)];high=[r for r in crit if not r["predicted_feasible"] and feas and r["rho"]>max(x["rho"] for x in feas)]
C={"schema_version":"4.0","status":"FROZEN_BEFORE_V4_HELDOUT","preregistration_sha256":hashlib.sha256((H/"UAV_V4_PREREGISTRATION.json").read_bytes()).hexdigest(),"heldout_seed_count_read":0,"Ep_m":Ep,"Vph_m":Vph,"Ev_mps":Ev,"Vvh_mps":Vvh,"D0_mps2":D0,"Db_mps2":Db,"U_mps2":U,"critical_function":crit,"feasible_rho":[r["rho"] for r in feas],"low_witness":low[-1]["rho"] if low else None,"interior_witness":min(feas,key=lambda r:r["psi"])["rho"] if feas else None,"high_witness":high[0]["rho"] if high else None}
raw=json.dumps(C,sort_keys=True,separators=(",",":")).encode();C["contract_sha256"]=hashlib.sha256(raw).hexdigest();(O/"UAV_V4_FROZEN_CONTRACT.json").write_text(json.dumps(C,indent=2)+"\n");print(json.dumps(C,indent=2))
