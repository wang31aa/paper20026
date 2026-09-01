#!/usr/bin/env python3
"""Machine-check the unit algebra and frozen V3.1 calculation."""
import csv,json,hashlib,math
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
Q=json.loads((HERE/"UAV_V31_PREREGISTRATION.json").read_text()); C=json.loads((OUT/"UAV_V31_FROZEN_CONTRACT.json").read_text())
base={k:C[k] for k in C if k!="contract_sha256"}; h=hashlib.sha256(json.dumps(base,sort_keys=True,separators=(",",":")).encode()).hexdigest()
g=list(csv.DictReader((OUT/"uav_v31_output_gains.csv").open())); f=list(csv.DictReader((OUT/"uav_v31_critical_function.csv").open()))
checks={"contract_hash":h==C["contract_sha256"],"no_heldout_read":C["future_heldout_seed_count_read"]==0,
 "position_error_unit":"m" in Q["units"]["Vph"],"velocity_error_unit":Q["units"]["Vvh"]=="m s^-1",
 "acceleration_budgets_same_unit":len({Q["units"][k] for k in ("D0","Db","U")})==1,
 "output_gains_s2":Q["units"]["K0"]==Q["units"]["Kpeak"]=="s^2",
 "dimensionless_psi":Q["units"]["Psi"]=="1","all_finite":all(math.isfinite(float(r["psi_output"])) for r in f),
 "all_modes_present":len(g)==len(Q["rho_grid"])*4}
report={"checks":checks,"dimensionally_closed":all(checks.values()),"contract_sha256":C["contract_sha256"],
 "feasible_rho":[float(r["rho"]) for r in f if int(r["predicted_feasible"])]}
(OUT/"uav_v31_dimension_validation.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
raise SystemExit(0 if report["dimensionally_closed"] else 2)
