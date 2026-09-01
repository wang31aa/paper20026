#!/usr/bin/env python3
import csv,hashlib,json,math,re
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/"UAV_V57_FROZEN_CONTRACT.json").read_text());R=json.loads((H/"results/V57_REPORT.json").read_text());rows=list(csv.DictReader((H/"results/V57_HELDOUT.csv").open()))
svg=(H/"results/Fig_V57_task_filter.svg").read_text()
font_sizes=[float(x) for x in re.findall(r"font(?:-size)?:\s*([0-9.]+)px",svg)]
checks={"contract_hash":R["contract_sha256"]==hashlib.sha256((H/"UAV_V57_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"rows":len(rows)==len(C["heldout_seeds"])*len(C["rho_grid"])*len(C["policies"]),"complete_pairing":len({(x["seed"],x["rho"],x["policy"]) for x in rows})==len(rows),"finite":all(math.isfinite(float(x[k])) for x in rows for k in ["task_rmse_m","minimum_clearance_m","energy","filter_cost"]),"all_policies_reported":set(R["summary"])==set(C["policies"]),"figure_fonts":bool(font_sizes) and min(font_sizes)>=7.0,"scope":all(x in R["claim_boundary"] for x in ["computational","not source-controller replay","HIL"])}
out={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks};(H/"results/V57_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
