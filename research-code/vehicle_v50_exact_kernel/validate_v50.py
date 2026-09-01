#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

H=Path(__file__).resolve().parent
C=json.loads((H/"V50_FROZEN_CONTRACT.json").read_text())
P=json.loads((H/"results/V50_FROZEN_POLICY.json").read_text())
R=json.loads((H/"results/V50_REPORT.json").read_text())
G=R["grid_certificates"]
checks={
 "contract_frozen":C["frozen_before_execution"] is True,
 "contract_hash":P["contract_sha256"]==hashlib.sha256((H/"V50_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),
 "prediction_hash":R["prediction_sha256"]==hashlib.sha256((H/"results/V50_FROZEN_POLICY.json").read_bytes()).hexdigest(),
 "seeds_disjoint":not set(C["development_seeds"])&set(C["heldout_seeds"]),
 "inner_subset_outer":all(x["inner_cells"]<=x["outer_cells"] for x in G),
 "geometric_bound_decreases":all(a["geometric_error_bound"]>b["geometric_error_bound"] for a,b in zip(G,G[1:])),
 "ambiguity_area_decreases":all(a["ambiguous_area_m_mps"]>b["ambiguous_area_m_mps"] for a,b in zip(G,G[1:])),
 "supervisor_all_safe":R["heldout"]["kernel_supervisor"]["success_rate"]==1.0,
 "claim_not_full_platform":"not asserted" in R["claim_boundary"]
}
out={"all_pass":all(checks.values()),"checks":checks}
(H/"results/V50_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2)); raise SystemExit(0 if out["all_pass"] else 1)
