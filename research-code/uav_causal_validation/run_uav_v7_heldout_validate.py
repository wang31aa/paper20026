#!/usr/bin/env python3
import csv,json
from pathlib import Path
from uav_v3_engine import simulate
H=Path(__file__).resolve().parent;O=H/"results";Q=json.loads((H/"UAV_V7_TARGET_ROOTED_PREREGISTRATION.json").read_text());C=json.loads((O/"UAV_V7_FROZEN_CONTRACT.json").read_text());assert C["heldout_authorized"] and C["heldout_seed_count_read"]==0
rows=[]
for env in Q["heldout_environments"]:
 for seed in Q["heldout_seeds"]:
  for rho in Q["witness_rho"].values():
   for policy in Q["policies"]:rows.append(simulate(policy,rho,env,seed,"all_heterogeneous",Q["observer_mode"],Q["command_reserve_mps2"],Q["max_missed_updates"],Q["base_update_interval_s"],Q["target_visibility"],Q["target_manoeuvre_amplitude_m"],Q["target_observer_gain"]))
with (O/"uav_v7_heldout_runs.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rate(rho,policy):
 z=[r["task_success"] for r in rows if r["rho"]==rho and r["policy"]==policy];return sum(z)/len(z)
lo,mi,hi=Q["witness_rho"].values();a,g,ind=Q["policies"];metrics={"low_all_failure_rate":1-rate(lo,a),"low_gate_failure_rate":1-rate(lo,g),"middle_all_success_rate":rate(mi,a),"middle_gate_success_rate":rate(mi,g),"high_all_failure_rate":1-rate(hi,a),"high_gate_success_rate":rate(hi,g),"high_independent_success_rate":rate(hi,ind)}
pc=Q["promotion_checks"];checks={"all_720_runs_complete":len(rows)==720,"middle_gate_success":metrics["middle_gate_success_rate"]>=pc["middle_gated_success_rate_min"],"low_information_failure":metrics["low_all_failure_rate"]>=pc["low_all_coupled_failure_rate_min"],"high_heterogeneity_failure":metrics["high_all_failure_rate"]>=pc["high_all_coupled_failure_rate_min"],"high_gate_recovery":metrics["high_gate_success_rate"]>=pc["high_gate_success_rate_min"],"all_environments_present":set(r["environment"] for r in rows)==set(Q["heldout_environments"])};qualified=all(checks.values());R={"schema_version":"7.0","contract_sha256":C["contract_sha256"],"status":"V7_COMPUTATIONAL_CAUSAL_VALIDATION_QUALIFIED" if qualified else "NOT_QUALIFIED","n_runs":len(rows),"metrics":metrics,"checks":checks,"scope":"paired held-out heterogeneous low-order closed-loop model; not official NMPC source replay, HIL or hardware"};(O/"uav_v7_validation.json").write_text(json.dumps(R,indent=2)+"\n");print(json.dumps(R,indent=2))
