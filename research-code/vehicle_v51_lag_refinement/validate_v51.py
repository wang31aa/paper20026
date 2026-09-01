#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/"V51_FROZEN_CONTRACT.json").read_text());P=json.loads((H/"results/V51_FROZEN_POLICY.json").read_text());R=json.loads((H/"results/V51_REPORT.json").read_text())
ready=[x for x in R["grid"] if x["realized_braking"]==C["commanded_max_braking_mps2"]]
checks={"contract_frozen":C["frozen_before_execution"] is True,"contract_hash":P["contract_sha256"]==hashlib.sha256((H/"V51_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"prediction_hash":R["prediction_sha256"]==hashlib.sha256((H/"results/V51_FROZEN_POLICY.json").read_bytes()).hexdigest(),"seed_sets_disjoint":not set(C["development_seeds"])&set(C["heldout_seeds"]),"kernel_membership_equivalence":R["equivalence"]["mismatches"]==0,"lag_buffer_nonnegative":R["equivalence"]["minimum_beta"]>=-1e-9,"fully_realized_braking_recovers_reduced_kernel":all(abs(x["buffer"])<1e-9 for x in ready),"lag_kernel_all_safe":R["heldout"]["lag_kernel"]["success_rate"]==1.0,"scope_not_four_vehicle":"not a four-vehicle" in R["claim_boundary"]}
out={"all_pass":all(checks.values()),"checks":checks};(H/"results/V51_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0 if out["all_pass"] else 1)
