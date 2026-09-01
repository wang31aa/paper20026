#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
CONTRACT_PATH = HERE / "V56_FROZEN_CONTRACT.json"
C = json.loads(CONTRACT_PATH.read_text())


def load_module(directory: str, module: str):
    sys.path.insert(0, str(ROOT / directory))
    try:
        return __import__(module)
    finally:
        sys.path.pop(0)


def uav_rows() -> list[dict]:
    model = load_module("uav_v54_sixdof", "run_v54")
    spec = C["new_streams"]["uav6dof"]
    rows = []
    for seed in spec["seeds"]:
        for rho in spec["rho"]:
            for policy in model.C["policies"]:
                r = model.simulate(seed, rho, policy)
                rows.append({
                    "platform": "uav6dof",
                    "case": f"{seed}|{rho}",
                    "policy": policy,
                    "task_success": r["success"],
                    "physical_margin": r["minimum_clearance_m"] - model.C["minimum_clearance_m"],
                    "terminal_error": r["task_rmse_m"],
                    "control_cost": r["energy"],
                    "communication": r["messages"],
                })
    return rows


def vehicle_rows() -> list[dict]:
    model = load_module("vehicle_v52_two_layer_platoon", "run_v52")
    spec = C["new_streams"]["vehicle"]
    rows = []
    for seed in spec["seeds"]:
        for delay in spec["delay_s"]:
            for error in spec["observer_error_mps"]:
                for policy in model.C["policies"]:
                    r = model.simulate(seed, delay, error, policy)
                    rows.append({
                        "platform": "vehicle",
                        "case": f"{seed}|{delay}|{error}",
                        "policy": policy,
                        "task_success": r["success"],
                        "physical_margin": r["minimum_margin_m"],
                        "terminal_error": r["observer_rmse"],
                        "control_cost": r["energy"],
                        "communication": r["messages"],
                    })
    return rows


def motor_rows() -> list[dict]:
    model = load_module("cross_domain_v21", "run_v21")
    spec = C["new_streams"]["motor"]
    rows = []
    for seed in spec["seeds"]:
        for size in spec["sizes"]:
            for topology in spec["topologies"]:
                for rho in spec["rho"]:
                    for policy in model.C["policies"]:
                        r = model.execute((size, topology, rho, seed, policy))
                        rows.append({
                            "platform": "motor",
                            "case": f"{seed}|{size}|{topology}|{rho}",
                            "policy": policy,
                            "task_success": r["task_success"],
                            "physical_margin": r["minimum_physical_margin"],
                            "terminal_error": r["tail_error"],
                            "control_cost": r["control_energy"],
                            "communication": r["communication_messages"],
                        })
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = uav_rows() + vehicle_rows() + motor_rows()
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["platform"], row["case"])].append(row)

    selected_rows = []
    for (platform, case), group in groups.items():
        max_cost = max(float(x["control_cost"]) for x in group) or 1.0
        max_error = max(float(x["terminal_error"]) for x in group) or 1.0
        for row in group:
            row["loss"] = ((1 - int(row["task_success"]))
                           + 0.05 * float(row["control_cost"]) / max_cost
                           + 0.05 * float(row["terminal_error"]) / max_error)
        oracle = min(float(x["loss"]) for x in group)
        chosen = [x for x in group if x["policy"] == C["selected_policy"][platform]]
        if len(chosen) != 1:
            raise RuntimeError(f"selected policy missing or duplicated: {platform} {case}")
        row = chosen[0]
        selected_rows.append({
            "platform": platform,
            "case": case,
            "selected_policy": row["policy"],
            "selected_success": row["task_success"],
            "selected_physical_margin": row["physical_margin"],
            "selected_loss": row["loss"],
            "oracle_loss": oracle,
            "regret": float(row["loss"]) - oracle,
        })

    with (OUT / "V56_ALL_POLICY_RUNS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "V56_SELECTED_POLICY_EVALUATION.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader(); writer.writerows(selected_rows)

    summary = {}
    for platform in C["platforms"]:
        q = [x for x in selected_rows if x["platform"] == platform]
        summary[platform] = {
            "conditions": len(q),
            "success_rate": float(np.mean([int(x["selected_success"]) for x in q])),
            "median_regret": float(np.median([float(x["regret"]) for x in q])),
            "maximum_regret": float(np.max([float(x["regret"]) for x in q])),
            "false_safe_count": int(sum(int(x["selected_success"]) == 0 for x in q)),
        }
    report = {
        "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "all_policy_rows": len(rows),
        "paired_conditions": len(selected_rows),
        "summary": summary,
        "claim_boundary": C["claim_boundary"],
    }
    (OUT / "V56_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
