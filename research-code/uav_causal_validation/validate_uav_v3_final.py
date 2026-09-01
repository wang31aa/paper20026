#!/usr/bin/env python3
"""Independent fail-closed V3 endpoint and intervention audit."""
import csv, hashlib, json, math, statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
P=json.loads((HERE/"UAV_V3_PREREGISTRATION.json").read_text())
C=json.loads((OUT/"UAV_V3_FROZEN_CONTRACT.json").read_text())
W=json.loads((OUT/"uav_v3_frozen_window.json").read_text())
rows=list(csv.DictReader((OUT/"uav_v3_heldout_runs.csv").open()))
expected=len(P["heldout_environments"])*len(P["heldout_seeds"])*len(P["rho_grid"])*len(P["policies"])
primary=[r for r in rows if r["policy"]=="connectivity_constrained_observer_gate"]
tp=tn=fp=fn=0
for r in primary:
    pred=bool(int(r["predicted_feasible"])); actual=bool(int(r["task_success"]))
    tp+=pred and actual; fp+=pred and not actual; tn+=(not pred) and (not actual); fn+=(not pred) and actual
sens=tp/(tp+fn) if tp+fn else None; spec=tn/(tn+fp) if tn+fp else None; fnr=fn/(tp+fn) if tp+fn else None

idx={(r["environment"],r["seed"],r["rho"],r["policy"]):r for r in rows}
paired=[]
for r in primary:
    b=idx[(r["environment"],r["seed"],r["rho"],"all_coupled_with_trim_sharing")]
    paired.append({"survival":int(r["task_success"])-int(b["task_success"]),
                   "recovery":float(r["recovery_time_s"])-float(b["recovery_time_s"]),
                   "energy":float(r["control_energy"])-float(b["control_energy"]),
                   "communication":float(r["communication_messages"])-float(b["communication_messages"])})

def witness(rho, expected_success):
    if rho is None: return False
    q=[r for r in primary if float(r["rho"])==float(rho)]
    rate=sum(int(r["task_success"]) for r in q)/len(q)
    return rate>=.8 if expected_success else rate<=.2

checks={
 "contract_hash":all(r["contract_sha256"]==C["contract_sha256"] for r in rows),
 "window_hash":all(r["window_sha256"]==W["window_sha256"] for r in rows),
 "row_count":len(rows)==expected,
 "all_finite":all(math.isfinite(float(r[k])) for r in rows for k in ("tail_tracking_error_m","control_energy","communication_messages")),
 "sensitivity":sens is not None and sens>=P["promotion_gates"]["sensitivity_min"],
 "specificity":spec is not None and spec>=P["promotion_gates"]["specificity_min"],
 "false_negative_rate":fnr is not None and fnr<=P["promotion_gates"]["false_negative_rate_max"],
 "low_witness":witness(W["low_failure_witness"],False),
 "interior_witness":witness(W["interior_success_witness"],True),
 "high_witness":witness(W["high_failure_witness"],False),
 "finite_window_was_frozen":bool(W["finite_window_pre_registered"]),
 "gate_changes_control_and_state":any(int(r["gate_active_steps"])>0 for r in primary)
}
summary={"confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn},"sensitivity":sens,"specificity":spec,"false_negative_rate":fnr,
 "witnesses":{"low":W["low_failure_witness"],"interior":W["interior_success_witness"],"high":W["high_failure_witness"]},
 "paired_gate_minus_all_coupled":{"mean_task_success_change":statistics.mean(x["survival"] for x in paired),
   "median_recovery_time_change_s":statistics.median(x["recovery"] for x in paired),
   "median_energy_change":statistics.median(x["energy"] for x in paired),
   "median_communication_change":statistics.median(x["communication"] for x in paired)},
 "checks":checks,"uav_v3_qualified":all(checks.values())}
(OUT/"uav_v3_final_validation.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2)); raise SystemExit(0 if summary["uav_v3_qualified"] else 2)
