#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
from uav_v3_engine import P,simulate
H=Path(__file__).resolve().parent;O=H/"results";Q=json.loads((H/"UAV_V61_PREREGISTRATION.json").read_text());C=json.loads((O/"UAV_V62_FROZEN_CONTRACT.json").read_text())
assert C["heldout_authorized"] and C["heldout_seed_count_read"]==0
rhos=[C["low_witness"],C["interior_witness"],C["high_witness"]]
policies=["all_coupled_with_trim_sharing","connectivity_constrained_observer_gate","independent_constrained_tracking"]
rows=[]
for env in P["heldout_environments"]:
 for seed in Q["heldout_seeds"]:
  for rho in rhos:
   for policy in policies:
    rows.append(simulate(policy,rho,env,seed,"all_heterogeneous",Q["observer_mode"],Q["command_reserve_mps2"],Q["max_missed_updates"],Q["base_update_interval_s"]))
with (O/"uav_v62_heldout_runs.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
g=[r for r in rows if r["policy"]=="connectivity_constrained_observer_gate"]
pred={x["rho"]:bool(x["predicted_feasible"]) for x in C["critical_function"]}
tp=sum(pred[r["rho"]] and r["task_success"] for r in g);fp=sum(pred[r["rho"]] and not r["task_success"] for r in g);tn=sum((not pred[r["rho"]]) and not r["task_success"] for r in g);fn=sum((not pred[r["rho"]]) and r["task_success"] for r in g)
feasible=[r for r in g if pred[r["rho"]]];outside=[r for r in g if not pred[r["rho"]]]
def med(xs):return float(np.median(xs))
by={(r["environment"],r["seed"],r["rho"],r["policy"]):r for r in rows};paired=[]
for env in P["heldout_environments"]:
 for seed in Q["heldout_seeds"]:
  for rho in rhos:
   a=by[env,seed,rho,"all_coupled_with_trim_sharing"];b=by[env,seed,rho,"connectivity_constrained_observer_gate"]
   paired.append({"environment":env,"seed":seed,"rho":rho,"tail_improvement_m":a["tail_tracking_error_m"]-b["tail_tracking_error_m"],"energy_change":b["control_energy"]-a["control_energy"],"communication_change":b["communication_messages"]-a["communication_messages"]})
with (O/"uav_v62_paired_effects.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
checks={"frozen_contract_hash_present":bool(C.get("contract_sha256")),"all_declared_heldout_environments_executed":set(r["environment"] for r in rows)==set(P["heldout_environments"]),"paired_random_stream_complete":len(rows)==len(P["heldout_environments"])*len(Q["heldout_seeds"])*len(rhos)*len(policies),"feasible_certificate_zero_task_failures":all(r["task_success"] for r in feasible),"low_middle_high_executed":set(r["rho"] for r in g)==set(rhos),"both_outside_witnesses_contain_failure":all(any((r["rho"]==rho and not r["task_success"]) for r in outside) for rho in (C["low_witness"],C["high_witness"])),"gating_has_positive_paired_task_effect":any(x["tail_improvement_m"]>0 for x in paired)}
qualified=all(checks.values())
R={"schema_version":"6.2","contract_sha256":C["contract_sha256"],"status":"CAUSAL_VALIDATION_QUALIFIED" if qualified else "NOT_QUALIFIED","n_runs":len(rows),"confusion_counts":{"tp_safe":tp,"fp_predicted_safe_but_failed":fp,"tn_predicted_outside_and_failed":tn,"fn_outside_but_safe":fn},"feasible_success_rate":sum(r["task_success"] for r in feasible)/len(feasible),"outside_success_rate":sum(r["task_success"] for r in outside)/len(outside),"median_gated_minus_all_coupled_tail_improvement_m":med([x["tail_improvement_m"] for x in paired]),"median_gated_minus_all_coupled_energy":med([x["energy_change"] for x in paired]),"checks":checks,"interpretation":"The frozen feasible set is a sufficient certificate. Outside-window successes do not falsify it, but absence of outside failures prevents promotion as a predictive task-failure boundary."};(O/"uav_v62_validation.json").write_text(json.dumps(R,indent=2)+"\n");print(json.dumps(R,indent=2))
