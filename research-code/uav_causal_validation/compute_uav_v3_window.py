#!/usr/bin/env python3
"""Compute Psi and freeze low/interior/high witnesses from the frozen contract."""
import csv, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
C = json.loads((OUT / "UAV_V3_FROZEN_CONTRACT.json").read_text())
base = {k: C[k] for k in C if k != "contract_sha256"}
check = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if check != C["contract_sha256"]: raise SystemExit("frozen contract hash mismatch")
rows = []
for rho in C["rho_grid"]:
    branches = []
    for i, m in enumerate(C["modes"]):
        a = m["a_bar_lower"] + m["kappa_lower"] * rho
        r = max(m["D0_upper"] + rho*m["Db_upper"] - m["U_lower"], 0.) / a
        b = m["Vh_upper"] / rho
        branches.append(((r+b)/C["epsilon_peak"], r/C["epsilon_ultimate"], i, r, b))
    peak = max(x[0] for x in branches); ultimate = max(x[1] for x in branches); ps = max(peak, ultimate)
    rows.append({"rho": rho, "psi": ps, "peak_branch": peak, "ultimate_branch": ultimate,
                 "predicted_feasible": int(ps <= 1), "active_mode": max(branches)[2]})
with (OUT / "uav_v3_critical_function.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
feasible=[r for r in rows if r["predicted_feasible"]]; infeasible=[r for r in rows if not r["predicted_feasible"]]
low=[r for r in infeasible if feasible and r["rho"] < min(x["rho"] for x in feasible)]
high=[r for r in infeasible if feasible and r["rho"] > max(x["rho"] for x in feasible)]
window={"contract_sha256":C["contract_sha256"], "rows":rows,
        "feasible_rho":[r["rho"] for r in feasible],
        "low_failure_witness": low[-1]["rho"] if low else None,
        "interior_success_witness": min(feasible,key=lambda x:x["psi"])["rho"] if feasible else None,
        "high_failure_witness": high[0]["rho"] if high else None,
        "finite_window_pre_registered": bool(low and feasible and high)}
raw=json.dumps(window,sort_keys=True,separators=(",",":")).encode(); window["window_sha256"]=hashlib.sha256(raw).hexdigest()
(OUT / "uav_v3_frozen_window.json").write_text(json.dumps(window,indent=2)+"\n")
print(json.dumps(window,indent=2))
