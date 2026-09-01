#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/"UAV_V54_FROZEN_CONTRACT.json").read_text());R=json.loads((H/"results/V54_REPORT.json").read_text());rows=list(csv.DictReader((H/"results/V54_HELDOUT.csv").open()))
finite=all(all(np==np and abs(float(np))<1e12 for np in [r["task_rmse_m"],r["minimum_clearance_m"],r["energy"],r["observer_rmse_m"]]) for r in rows)
checks={"contract_frozen":C["frozen_before_execution"] is True,"contract_hash":R["contract_sha256"]==hashlib.sha256((H/"UAV_V54_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"expected_runs":len(rows)==len(C["heldout_seeds"])*len(C["rho_grid"])*len(C["policies"]),"finite":finite,"heterogeneous_mass":len(set(C["masses_kg"]))==C["agents"],"heterogeneous_drag":len(set(C["drag_per_s"]))==C["agents"],"heterogeneous_lag":len(set(C["thrust_lags_s"]))==C["agents"],"all_policies":set(r["policy"] for r in rows)==set(C["policies"]),"scope_not_official":"not official EPFL" in R["claim_boundary"]}
out={"all_pass":all(checks.values()),"checks":checks};(H/"results/V54_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0 if out["all_pass"] else 1)
