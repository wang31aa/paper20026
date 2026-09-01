#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
C=json.loads((H/"V53_FROZEN_CONTRACT.json").read_text());R=json.loads((H/"results/V53_REPORT.json").read_text())
rows=R["rows"]
checks={"contract_frozen":C["frozen_before_execution"] is True,
        "contract_hash":R["contract_sha256"]==hashlib.sha256((H/"V53_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),
        "all_resolutions":len(rows)==len(C["coarse_resolutions"]),
        "inner_exact_outer":all(x["inner_states"]<=x["coarse_exact_states"]<=x["outer_states"] for x in rows),
        "ambiguity_nonincreasing":all(rows[i+1]["ambiguity_fraction_of_safe"]<=rows[i]["ambiguity_fraction_of_safe"]+1e-12 for i in range(len(rows)-1)),
        "scope_finite": "finite three-vehicle" in R["claim_boundary"],
        "scope_not_continuous": "not an exact continuous" in R["claim_boundary"]}
out={"all_pass":all(checks.values()),"checks":checks};(H/"results/V53_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0 if out["all_pass"] else 1)
