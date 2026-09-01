#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
from uav_v3_engine import simulate
H=Path(__file__).resolve().parent;O=H/"results";Q=json.loads((H/"UAV_V7_TARGET_ROOTED_PREREGISTRATION.json").read_text());rows=[]
for env in Q["development_environments"]:
 for seed in Q["development_seeds"]:
  for rho in Q["witness_rho"].values():
   for policy in Q["policies"]: rows.append(simulate(policy,rho,env,seed,"all_heterogeneous",Q["observer_mode"],Q["command_reserve_mps2"],Q["max_missed_updates"],Q["base_update_interval_s"],Q["target_visibility"],Q["target_manoeuvre_amplitude_m"],Q["target_observer_gain"]))
with (O/"uav_v7_development_runs.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rate(rho,policy):
 z=[r["task_success"] for r in rows if r["rho"]==rho and r["policy"]==policy];return sum(z)/len(z)
low=Q["witness_rho"]["low"];mid=Q["witness_rho"]["middle"];high=Q["witness_rho"]["high"]
summary={"low_all_success_rate":rate(low,Q["policies"][0]),"middle_all_success_rate":rate(mid,Q["policies"][0]),"middle_gate_success_rate":rate(mid,Q["policies"][1]),"high_all_success_rate":rate(high,Q["policies"][0]),"high_gate_success_rate":rate(high,Q["policies"][1])}
C={"schema_version":"7.0","status":"FROZEN_BEFORE_V7_HELDOUT","heldout_seed_count_read":0,"preregistration_sha256":hashlib.sha256((H/"UAV_V7_TARGET_ROOTED_PREREGISTRATION.json").read_bytes()).hexdigest(),"development_summary":summary,"witness_rho":Q["witness_rho"],"heldout_authorized":summary["middle_gate_success_rate"]>=.9 and summary["high_all_success_rate"]<=.5 and summary["high_gate_success_rate"]>=.8 and summary["low_all_success_rate"]<=.5};raw=json.dumps(C,sort_keys=True,separators=(",",":")).encode();C["contract_sha256"]=hashlib.sha256(raw).hexdigest();(O/"UAV_V7_FROZEN_CONTRACT.json").write_text(json.dumps(C,indent=2)+"\n");print(json.dumps(C,indent=2))
