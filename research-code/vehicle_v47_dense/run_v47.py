#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "vehicle_v46"))
from run_v46 import POLICIES, load_source, simulate  # noqa: E402

CONTRACT = HERE / "V47_STAGE1_CONTRACT.json"
OUT = HERE / "results"


def thresholds(source: dict[str, np.ndarray], fraction: float = 0.6) -> tuple[float, float, float]:
    split = int(fraction * len(source["time"]))
    cal = np.column_stack((source["acc_av"][:split], source["acc_sv1"][:split], source["acc_sv2"][:split]))
    disagreement = np.abs(cal - np.median(cal, axis=1, keepdims=True))
    high = float(np.quantile(disagreement, 0.9))
    low = float(np.quantile(disagreement, 0.6))
    gap = max(25.0, 0.5 * float(np.quantile(source["distance_av_headway"][:split], 0.1)))
    return high, low, gap


def run_rows(source, spec, seeds):
    high, low, gap = thresholds(source)
    return [simulate(source, stress, delay, rho, seed, policy, high, low, gap)
            for seed in seeds
            for stress in spec["stress"]
            for delay in spec["delay_s"]
            for rho in spec["participation_grid"]
            for policy in spec["policies"]]


def rate_curve(rows, policy):
    grid = sorted({float(r["rho"]) for r in rows})
    return [{"rho": rho,
             "success_rate": float(np.mean([r["task_success"] for r in rows
                                              if r["policy"] == policy and float(r["rho"]) == rho]))}
            for rho in grid]


def classify(curve, threshold):
    rates = np.array([r["success_rate"] for r in curve])
    feasible = rates >= threshold
    if not feasible.any():
        return "infeasible", []
    dif = np.diff(rates)
    if np.all(dif >= -1e-12):
        cls = "monotone_non_decreasing"
    elif np.all(dif <= 1e-12):
        cls = "monotone_non_increasing"
    else:
        idx = np.flatnonzero(feasible)
        cls = "finite_window" if np.array_equal(idx, np.arange(idx.min(), idx.max() + 1)) and idx.min() > 0 and idx.max() < len(rates) - 1 else "irregular"
    interval = [curve[int(np.flatnonzero(feasible)[0])]["rho"], curve[int(np.flatnonzero(feasible)[-1])]["rho"]]
    return cls, interval


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


def main():
    spec = json.loads(CONTRACT.read_text())
    if any(p not in POLICIES for p in spec["policies"]):
        raise ValueError("undeclared parent policy")
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_source()
    development = run_rows(source, spec, spec["development_seeds"])
    write_csv(OUT / "V47_DEVELOPMENT.csv", development)
    threshold = float(spec["response_rule"]["feasible_success_rate"])
    predictions = {"contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(), "policies": {}}
    for policy in spec["policies"]:
        curve = rate_curve(development, policy)
        cls, interval = classify(curve, threshold)
        predictions["policies"][policy] = {"class": cls, "feasible_interval": interval, "development_curve": curve}
    pred_path = OUT / "V47_FROZEN_PREDICTIONS.json"
    pred_path.write_text(json.dumps(predictions, indent=2) + "\n")
    pred_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()

    heldout = run_rows(source, spec, spec["heldout_seeds"])
    write_csv(OUT / "V47_HELDOUT.csv", heldout)
    evaluation = {"prediction_sha256": pred_sha, "policies": {}}
    for policy in spec["policies"]:
        curve = rate_curve(heldout, policy)
        observed_class, observed_interval = classify(curve, threshold)
        pred = predictions["policies"][policy]
        predicted_safe = {x["rho"] for x in pred["development_curve"] if x["success_rate"] >= threshold}
        y = [(float(r["rho"]), int(r["task_success"])) for r in heldout if r["policy"] == policy]
        tp = sum(outcome == 1 and rho in predicted_safe for rho, outcome in y)
        fp = sum(outcome == 0 and rho in predicted_safe for rho, outcome in y)
        tn = sum(outcome == 0 and rho not in predicted_safe for rho, outcome in y)
        fn = sum(outcome == 1 and rho not in predicted_safe for rho, outcome in y)
        evaluation["policies"][policy] = {
            "predicted_class": pred["class"], "observed_class": observed_class,
            "predicted_interval": pred["feasible_interval"], "observed_interval": observed_interval,
            "heldout_curve": curve, "false_safe_count": fp, "false_unsafe_count": fn,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "sensitivity": tp / (tp + fn) if tp + fn else None
        }
    evaluation["claim_boundary"] = spec["claim_boundary"]
    (OUT / "V47_HELDOUT_EVALUATION.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
