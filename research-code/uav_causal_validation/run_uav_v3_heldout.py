#!/usr/bin/env python3
"""Run untouched V3 held-out conditions using only the frozen contract."""
import csv, hashlib, json
from pathlib import Path
from uav_v3_engine import P, simulate

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
C=json.loads((OUT/"UAV_V3_FROZEN_CONTRACT.json").read_text())
base={k:C[k] for k in C if k!="contract_sha256"}
if hashlib.sha256(json.dumps(base,sort_keys=True,separators=(",",":")).encode()).hexdigest()!=C["contract_sha256"]:
    raise SystemExit("contract hash mismatch")
W=json.loads((OUT/"uav_v3_frozen_window.json").read_text())
wb={k:W[k] for k in W if k!="window_sha256"}
if hashlib.sha256(json.dumps(wb,sort_keys=True,separators=(",",":")).encode()).hexdigest()!=W["window_sha256"]:
    raise SystemExit("window hash mismatch")
if set(P["development_seeds"]) & set(P["heldout_seeds"]): raise SystemExit("seed leakage")

rows=[]
for env in P["heldout_environments"]:
  for seed in P["heldout_seeds"]:
    for rho in C["rho_grid"]:
      for policy in P["policies"]:
        r=simulate(policy,float(rho),env,int(seed),"all_heterogeneous")
        r["predicted_feasible"]=int(rho in W["feasible_rho"])
        r["contract_sha256"]=C["contract_sha256"]; r["window_sha256"]=W["window_sha256"]
        rows.append(r)
with (OUT/"uav_v3_heldout_runs.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps({"rows":len(rows),"contract_sha256":C["contract_sha256"],
                  "window_sha256":W["window_sha256"],"heldout_seeds":P["heldout_seeds"]},indent=2))
