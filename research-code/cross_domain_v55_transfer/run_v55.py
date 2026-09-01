#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;O=H/"results";C=json.loads((H/"V55_FROZEN_CONTRACT.json").read_text())

def vehicle_rows():
    sys.path.insert(0,str(ROOT/"vehicle_v52_two_layer_platoon"));import run_v52 as v
    return [{"platform":"vehicle","case":f"{s}|{d}|{e}","policy":p,"success":r["success"],"energy":r["energy"],"error":r["observer_rmse"]} for s in v.C["heldout_seeds"] for d in v.C["message_delay_s"] for e in v.C["observer_error_bounds_mps"] for p in v.C["policies"] for r in [v.simulate(s,d,e,p)]]

def uav_rows():
    rr=list(csv.DictReader((ROOT/"uav_v54_sixdof/results/V54_HELDOUT.csv").open()))
    return [{"platform":"uav6dof","case":f"{r['seed']}|{r['rho']}","policy":r["policy"],"success":int(r["success"]),"energy":float(r["energy"]),"error":float(r["task_rmse_m"])} for r in rr]

def motor_rows():
    rr=[r for r in csv.DictReader((ROOT/"cross_domain_v42/results/v42_all_policy_runs.csv").open()) if r["domain"]=="motor" and r["split"]=="heldout"]
    return [{"platform":"motor","case":f"{r['n']}|{r['rho']}|{r['seed']}|{r.get('topology_family','')}","policy":r["policy"],"success":int(r["task_success"]),"energy":float(r["control_energy"]),"error":float(r["tail_error"])} for r in rr]

def main():
    O.mkdir(exist_ok=True);rows=vehicle_rows()+uav_rows()+motor_rows();groups=defaultdict(list)
    for r in rows:groups[(r["platform"],r["case"])].append(r)
    outrows=[]
    for (plat,case),q in groups.items():
        me=max(x["energy"] for x in q) or 1;mr=max(x["error"] for x in q) or 1
        for x in q:x["loss"]=(1-x["success"])+.05*x["energy"]/me+.05*x["error"]/mr
        oracle=min(x["loss"] for x in q);chosen=[x for x in q if x["policy"]==C["platform_policy"][plat]]
        if not chosen:continue
        x=chosen[0];outrows.append({"platform":plat,"case":case,"selected_policy":x["policy"],"selected_success":x["success"],"selected_loss":x["loss"],"oracle_loss":oracle,"regret":x["loss"]-oracle})
    summary={p:{"cases":len(q),"success_rate":float(np.mean([x["selected_success"] for x in q])),"median_regret":float(np.median([x["regret"] for x in q])),"p95_regret":float(np.quantile([x["regret"] for x in q],.95))} for p in C["platform_policy"] for q in [[x for x in outrows if x["platform"]==p]]}
    with (O/"V55_REGRET.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
    out={"contract_sha256":hashlib.sha256((H/"V55_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"summary":summary,"claim_boundary":C["claim_boundary"]};(O/"V55_REPORT.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
