#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/"V55_FROZEN_CONTRACT.json").read_text());R=json.loads((H/"results/V55_REPORT.json").read_text());rows=list(csv.DictReader((H/"results/V55_REGRET.csv").open()))
checks={"contract_frozen":C["frozen_before_execution"] is True,"contract_hash":R["contract_sha256"]==hashlib.sha256((H/"V55_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"three_platforms":set(x["platform"] for x in rows)==set(C["platform_policy"]),"regret_nonnegative":all(float(x["regret"])>=-1e-12 for x in rows),"finite":all(abs(float(x["regret"]))<1e6 for x in rows),"scope_retrospective":"retrospective" in R["claim_boundary"],"scope_not_universal":"not a leave-physical-domain" in R["claim_boundary"]}
out={"all_pass":all(checks.values()),"checks":checks};(H/"results/V55_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0 if out["all_pass"] else 1)
