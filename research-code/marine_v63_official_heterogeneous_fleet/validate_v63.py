#!/usr/bin/env python3
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
C = json.loads((H / "V63_FROZEN_CONTRACT.json").read_text())
O = H / "results"
runs = [[float(v) for v in r] for r in csv.reader((O / "V63_RUNS.csv").open())]
ts = [[float(v) for v in r] for r in csv.reader((O / "V63_NODE_TIMESERIES.csv").open())]
expected_commit = "99bf0b30e9ae0dca3515d02e0bbd06c54cc0c2f7"

counts = Counter(int(r[0]) for r in ts)
run_ids = {int(r[0]) for r in runs}
rho_expected = {round(float(v), 12) for v in C["rho_grid"]}
rho_observed = {round(r[2], 12) for r in runs}
policy_ids = {int(r[3]) for r in runs}
node_payloads, node_currents = defaultdict(set), defaultdict(set)
for row in ts:
    node_payloads[int(row[3])].add(round(row[26], 12))
    node_currents[int(row[3])].add(round(row[27], 12))

max_applied = max(max(abs(r[16]), abs(r[17])) for r in ts)
max_requested = max(max(abs(r[18]), abs(r[19])) for r in ts)
checks = {
    "run_factorial": len(runs) == 18 and run_ids == set(range(1, 19)),
    "timeseries_complete": len(ts) == 18 * 301 * 5 and all(counts[i] == 301 * 5 for i in range(1, 19)),
    "full_schema": len(C["node_log_schema"]) == 28 and all(len(r) == 28 for r in ts),
    "finite": all(math.isfinite(v) for r in runs + ts for v in r),
    "heterogeneous_payloads_in_trace": sorted(next(iter(node_payloads[i])) for i in range(1, 6)) == [0, 10, 20, 30, 40],
    "heterogeneous_currents_in_trace": len({next(iter(node_currents[i])) for i in range(1, 6)}) == 5,
    "official_commit_exact": C["mss_commit"] == expected_commit,
    "rho_grid_exact": rho_observed == rho_expected,
    "policy_set_exact": policy_ids == {1, 2, 3},
    "paired_policies": all(sum(1 for r in runs if int(r[1]) == s and abs(r[2] - rho) < 1e-9) == 3 for s in C["seeds"] for rho in C["rho_grid"]),
    "applied_propeller_limit": max_applied <= 103.9309 + 1e-3,
    "requested_propeller_limit": max_requested <= 103.9309 + 1e-3,
    "run_trace_limit_agrees": abs(max(r[9] for r in runs) - max_applied) < 1e-9,
}
summary = {p: {"successes": sum(int(r[4]) for r in runs if int(r[3]) == i + 1), "runs": sum(1 for r in runs if int(r[3]) == i + 1)} for i, p in enumerate(C["policies"])}
report = {
    "status": "OFFICIAL_HETEROGENEOUS_FLEET_COMPUTATION_QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED",
    "checks": checks,
    "summary": summary,
    "max_applied_propeller_rad_s": max_applied,
    "max_requested_propeller_rad_s": max_requested,
    "runs_sha256": hashlib.sha256((O / "V63_RUNS.csv").read_bytes()).hexdigest(),
    "timeseries_sha256": hashlib.sha256((O / "V63_NODE_TIMESERIES.csv").read_bytes()).hexdigest(),
    "scope": C["claim_boundary"],
}
(O / "V63_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
