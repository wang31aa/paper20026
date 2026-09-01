#!/usr/bin/env python3
"""Execute the frozen V8 dense participation and mechanism matrix."""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys
from pathlib import Path
from uav_v3_engine import simulate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PROTOCOL = HERE / "UAV_V8_DENSE_PREREGISTRATION.json"
Q = json.loads(PROTOCOL.read_text())

def run_task(task: tuple[str, float, str, int, str]) -> dict:
    policy, rho, env, seed, arm = task
    return simulate(policy, rho, env, seed, arm, Q["observer_mode"],
                    Q["command_reserve_mps2"], Q["max_missed_updates"],
                    Q["base_update_interval_s"], Q["target_visibility"],
                    Q["target_manoeuvre_amplitude_m"], Q["target_observer_gain"])

dense_tasks = [(policy, rho, env, seed, "all_heterogeneous")
               for env in Q["environments"] for seed in Q["seeds"]
               for rho in Q["rho_grid"] for policy in Q["policies"]]
ablation_tasks = [("all_coupled_with_trim_sharing", rho, env, seed, arm)
                  for env in Q["environments"] for seed in Q["seeds"]
                  for rho in Q["mechanism_witness_rho"] for arm in Q["mechanism_arms"]]
all_tasks = [("dense", task) for task in dense_tasks] + [("ablation", task) for task in ablation_tasks]

def task_id(kind: str, task: tuple[str, float, str, int, str]) -> str:
    policy, rho, env, seed, arm = task
    return f"{kind}|{policy}|{rho:.12g}|{env}|{seed}|{arm}"

if len(sys.argv) == 4 and sys.argv[1] == "--worker":
    worker, workers = int(sys.argv[2]), int(sys.argv[3])
    path = OUT / f"uav_v8_worker_{worker}.jsonl"
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip(): completed.add(json.loads(line)["task_id"])
    with path.open("a") as handle:
        for index, (kind, task) in enumerate(all_tasks):
            identifier = task_id(kind, task)
            if index % workers != worker or identifier in completed: continue
            item = {"task_id":identifier, "kind":kind, "row":run_task(task)}
            handle.write(json.dumps(item, separators=(",", ":"))+"\n")
            handle.flush()
    raise SystemExit(0)

workers = 4
processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i), str(workers)])
             for i in range(workers)]
codes = [process.wait() for process in processes]
if any(codes):
    raise RuntimeError(f"V8 worker failure: {codes}")
payload = []
for i in range(workers):
    path = OUT / f"uav_v8_worker_{i}.jsonl"
    payload.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
expected = {task_id(kind, task) for kind, task in all_tasks}
observed = {item["task_id"] for item in payload}
if observed != expected or len(payload) != len(observed):
    raise RuntimeError(f"V8 checkpoint mismatch: expected={len(expected)} observed={len(observed)} rows={len(payload)}")
rows = [item["row"] for item in payload if item["kind"] == "dense"]
ablations = [item["row"] for item in payload if item["kind"] == "ablation"]

def write_csv(path: Path, values: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)

write_csv(OUT / "uav_v8_dense_runs.csv", rows)
write_csv(OUT / "uav_v8_mechanism_ablations.csv", ablations)

def rate(values: list[dict], policy: str, rho: float, arm: str | None = None) -> float:
    selected = [r["task_success"] for r in values
                if r["policy"] == policy and r["rho"] == rho
                and (arm is None or r["arm"] == arm)]
    return sum(selected) / len(selected)

all_policy, gate_policy, _ = Q["policies"]
curve = [{"rho": rho,
          "all_success_rate": rate(rows, all_policy, rho),
          "gate_success_rate": rate(rows, gate_policy, rho),
          "independent_success_rate": rate(rows, Q["policies"][2], rho)}
         for rho in Q["rho_grid"]]
low_ok = all(1-c["all_success_rate"] > .5 for c in curve if c["rho"] <= .55)
mid_ok = any(c["all_success_rate"] > .8 for c in curve if .95 <= c["rho"] <= 1.4)
high_ok = any(1-c["all_success_rate"] > .5 for c in curve if c["rho"] >= 2.2)
gate_ok = any(c["gate_success_rate"]-c["all_success_rate"] >= .5 for c in curve if c["rho"] >= 2.2)
homogeneous_ok = rate(ablations, all_policy, 2.5, "homogeneous") > .8
low_no_share_fail = 1-rate(ablations, all_policy, .5, "all_heterogeneous_without_feedforward_sharing") > .5
checks = {"complete_dense_matrix": len(rows)==len(Q["environments"])*len(Q["seeds"])*len(Q["rho_grid"])*len(Q["policies"]),
          "complete_ablation_matrix": len(ablations)==len(Q["environments"])*len(Q["seeds"])*len(Q["mechanism_witness_rho"])*len(Q["mechanism_arms"]),
          "low_information_region": low_ok, "intermediate_feasible_region": mid_ok,
          "high_mismatch_region": high_ok, "gate_recovery": gate_ok,
          "homogeneous_counterexample": homogeneous_ok,
          "low_information_ablation": low_no_share_fail}
result = {"schema_version":"8.0",
          "preregistration_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
          "status":"V8_SAME_SIMULATOR_CONFIRMATORY_EXTENSION_PASSED" if all(checks.values()) else "NOT_QUALIFIED",
          "n_dense_runs":len(rows), "n_ablation_runs":len(ablations),
          "curve":curve, "checks":checks, "scope":Q["claim_boundary"]}
(OUT / "uav_v8_validation.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
