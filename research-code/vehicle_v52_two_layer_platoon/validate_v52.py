#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/"V52_FROZEN_CONTRACT.json").read_text());R=json.loads((H/"results/V52_REPORT.json").read_text());S=R["summary"]
checks={"contract_frozen":C["frozen_before_execution"] is True,"contract_hash":R["contract_sha256"]==hashlib.sha256((H/"V52_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"expected_runs":R["runs"]==len(C["heldout_seeds"])*len(C["message_delay_s"])*len(C["observer_error_bounds_mps"])*len(C["policies"]),"heterogeneous_braking":len(set(C["braking_limits_mps2"]))==C["vehicles"],"heterogeneous_lag":len(set(C["actuator_lags_s"]))==C["vehicles"],"two_layer_all_safe":S["two_layer_lag_kernel"]["success_rate"]==1.0,"scope_is_sufficient":"sufficient" in R["claim_boundary"],"scope_not_exact_global":"not the exact global" in R["claim_boundary"]}
out={"all_pass":all(checks.values()),"checks":checks};(H/"results/V52_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0 if out["all_pass"] else 1)
