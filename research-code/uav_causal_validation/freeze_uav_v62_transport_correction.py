#!/usr/bin/env python3
"""Freeze the held-out transport contract before reading any held-out seed."""
import csv,hashlib,json
from pathlib import Path
from uav_v3_engine import DT
H=Path(__file__).resolve().parent;O=H/"results"
Q=json.loads((H/"UAV_V61_PREREGISTRATION.json").read_text());P=json.loads((O/"UAV_V61_FROZEN_CONTRACT.json").read_text())
assert P["heldout_seed_count_read"]==0
delay=.06
gain={(float(r["rho"]),int(r["mode"])):r for r in csv.DictReader((O/"uav_v31_output_gains.csv").open())};crit=[];parts=[]
for rho in Q["rho_grid"]:
 T=(Q["max_missed_updates"]+1)*max(1,round(Q["base_update_interval_s"]/(rho*DT)))*DT+delay;bounds=[]
 for m in range(4):
  g=gain[(rho,m)];K0=float(g["K0_s2"]);Kp=float(g["Kpeak_s2"]);R=max(P["D0_mps2"]+rho*P["Db_mps2"]-P["U_mps2"],0);force=Kp*R;pos=P["Ep_m"]+P["Ev_mps"]*T+.5*P["Ap_mps2"]*T*T;vel=Kp*1.45*(P["Ev_mps"]+P["Av_mps2"]*T);peak=force+pos+vel;ult=K0*R;psi=max(peak/Q["epsilon_peak_m"],ult/Q["epsilon_ultimate_m"]);bounds.append((psi,m,peak,ult));parts.append({"rho":rho,"mode":m,"message_age_s":T,"force_m":force,"position_info_m":pos,"velocity_info_m":vel,"peak_m":peak,"ultimate_m":ult})
 w=max(bounds);crit.append({"rho":rho,"psi":w[0],"active_mode":w[1],"peak_bound_m":w[2],"ultimate_bound_m":w[3],"predicted_feasible":int(w[0]<=1)})
for name,data in (("uav_v62_term_breakdown.csv",parts),("uav_v62_critical_function.csv",crit)):
 with (O/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
feas=[r for r in crit if r["predicted_feasible"]];low=[r for r in crit if feas and not r["predicted_feasible"] and r["rho"]<min(x["rho"] for x in feas)];high=[r for r in crit if feas and not r["predicted_feasible"] and r["rho"]>max(x["rho"] for x in feas)];C={k:P[k] for k in ("Ep_m","Ev_mps","Ap_mps2","Av_mps2","D0_mps2","Db_mps2","U_mps2")};C.update({"schema_version":"6.2","status":"FROZEN_BEFORE_V62_HELDOUT","correction":"maximum declared held-out transport delay included","maximum_transport_delay_s":delay,"heldout_seed_count_read":0,"critical_function":crit,"feasible_rho":[r["rho"] for r in feas],"low_witness":low[-1]["rho"] if low else None,"interior_witness":min(feas,key=lambda r:r["psi"])["rho"] if feas else None,"high_witness":high[0]["rho"] if high else None,"heldout_authorized":bool(feas and low and high)});raw=json.dumps(C,sort_keys=True,separators=(",",":")).encode();C["contract_sha256"]=hashlib.sha256(raw).hexdigest();(O/"UAV_V62_FROZEN_CONTRACT.json").write_text(json.dumps(C,indent=2)+"\n");print(json.dumps(C,indent=2))
