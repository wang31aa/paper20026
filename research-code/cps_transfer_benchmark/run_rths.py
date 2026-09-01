#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RIDGE = 1e-5
TRAIN_IDS = {11, 12, 13}
CAL_IDS = {14, 15}
TEST_IDS = set(range(16, 23))
SEED = 20260729


def file_id(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def design(disp: np.ndarray, force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.gradient(disp)
    x = np.column_stack([
        np.ones(len(disp) - 2), disp[1:-1], disp[:-2], velocity[1:-1],
        np.abs(disp[1:-1]), np.sign(velocity[1:-1]), force[:-2],
    ])
    return x, force[2:]


def ridge_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * RIDGE
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def robust_scale(y: np.ndarray) -> float:
    q25, q75 = np.quantile(y, [0.25, 0.75])
    return max(float((q75 - q25) / 1.349), 1e-12)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(int)
    total = int(y.sum())
    if total == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / total)


def load_records(raw: Path):
    records = []
    for path in sorted(raw.glob("*_RTHS_*.csv")):
        idx = file_id(path)
        frame = pd.read_csv(path)
        for actuator in (1, 2):
            dc, fc = f"Actuator{actuator}_Disp", f"Actuator{actuator}_Force"
            if dc in frame and fc in frame:
                records.append((idx, path.name, actuator,
                                frame[dc].to_numpy(float), frame[fc].to_numpy(float)))
    return records


def metrics(y: np.ndarray, pred: np.ndarray, disp: np.ndarray, scale: float):
    err = y - pred
    dx = np.diff(disp[2:], prepend=disp[2])
    true_energy = float(np.sum(y * dx))
    pred_energy = float(np.sum(pred * dx))
    return {
        "n": len(y), "nrmse": float(np.sqrt(np.mean(err**2)) / scale),
        "nmae": float(np.mean(np.abs(err)) / scale),
        "peak_abs_normalized": float(np.max(np.abs(err)) / scale),
        "hysteresis_energy_relative_error": float(abs(pred_energy - true_energy) /
                                                   max(abs(true_energy), 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.raw)
    assert {r[0] for r in records} == TRAIN_IDS | CAL_IDS | TEST_IDS

    train = [r for r in records if r[0] in TRAIN_IDS]
    xt, yt = zip(*(design(r[3], r[4]) for r in train))
    xtrain, ytrain = np.vstack(xt), np.concatenate(yt)
    beta = ridge_fit(xtrain, ytrain)
    scale = robust_scale(ytrain)

    cal_resid = []
    for r in records:
        if r[0] in CAL_IDS:
            x, y = design(r[3], r[4]); cal_resid.extend(np.abs(y - x @ beta) / scale)
    threshold = float(np.quantile(np.asarray(cal_resid), 0.95, method="higher"))

    rows, fault_rows = [], []
    rng = np.random.default_rng(SEED)
    for idx, name, actuator, disp, force in records:
        if idx not in TEST_IDS:
            continue
        x, y = design(disp, force)
        pred = x @ beta
        base = force[1:-1]
        row = {"run_id": idx, "file": name, "actuator": actuator, "split": "test"}
        row.update({f"model_{k}": v for k, v in metrics(y, pred, disp, scale).items()})
        row.update({f"persistence_{k}": v for k, v in metrics(y, base, disp, scale).items()})
        resid = np.abs(y - pred) / scale
        row["coverage_95"] = float(np.mean(resid <= threshold))
        rows.append(row)

        lo, hi = int(.30 * len(force)), int(.70 * len(force))
        def append_fault(corrupt, labels, fault, level, magnitude):
            xf, yf = design(disp, corrupt); score = np.abs(yf - xf @ beta) / scale
            fault_rows.append({"run_id": idx, "actuator": actuator, "fault": fault,
                               "level": level, "magnitude_scale": magnitude,
                               "positive_prevalence": float(np.mean(labels)),
                               "auprc": average_precision(labels, score),
                               "false_alarm_rate": float(np.mean(score[~labels] > threshold)),
                               "detected_fraction": float(np.mean(score[labels] > threshold))})
        for level, magnitude in enumerate([0.5, 1.0, 2.0, 4.0], 1):
            corrupt = force.copy(); labels = np.zeros(len(force) - 2, dtype=bool)
            corrupt[lo:hi] += magnitude * scale; labels[max(0, lo-2):max(0, hi-2)] = True
            append_fault(corrupt, labels, "force_bias", level, magnitude)
        for level, gain in enumerate([0.01, 0.03, 0.06, 0.10], 1):
            corrupt = force.copy(); labels = np.zeros(len(force) - 2, dtype=bool)
            corrupt[lo:hi] *= 1.0 + gain; labels[max(0, lo-2):max(0, hi-2)] = True
            append_fault(corrupt, labels, "force_gain", level, gain)
        for level, width in enumerate([1, 3, 6, 12], 1):
            corrupt = force.copy(); labels_source = np.zeros(len(force), dtype=bool)
            for start in range(lo, hi, 200):
                stop = min(start + width, hi)
                corrupt[start:stop] = corrupt[max(start - 1, 0)]
                labels_source[start:stop] = True
            append_fault(corrupt, labels_source[2:], "force_dropout", level, width)
        for level, sigma in enumerate([0.25, 0.5, 1.0, 2.0], 1):
            corrupt = force.copy(); labels = np.zeros(len(force) - 2, dtype=bool)
            corrupt[lo:hi] += rng.normal(0, sigma * scale, hi-lo)
            labels[max(0, lo-2):max(0, hi-2)] = True
            append_fault(corrupt, labels, "force_noise", level, sigma)

    run_df, fault_df = pd.DataFrame(rows), pd.DataFrame(fault_rows)
    run_df.to_csv(args.out / "rths_run_metrics.csv", index=False)
    fault_df.to_csv(args.out / "rths_fault_metrics.csv", index=False)
    improvement = run_df.persistence_nrmse - run_df.model_nrmse
    run_improvement = run_df.assign(improvement=improvement).groupby("run_id").improvement.median()
    boot, boot_cluster = [], []
    boot_rng = np.random.default_rng(SEED + 1)
    for _ in range(10000):
        boot.append(float(np.median(boot_rng.choice(improvement, len(improvement), replace=True))))
        boot_cluster.append(float(np.median(boot_rng.choice(run_improvement, len(run_improvement), replace=True))))
    summary = {
        "source": {"doi": "10.5281/zenodo.17296336", "archive_md5": "da06b1056528e8429e7f64e3019cb17b",
                   "license": "CC-BY-SA-4.0"},
        "split": {"train": sorted(TRAIN_IDS), "calibration": sorted(CAL_IDS), "test": sorted(TEST_IDS)},
        "model": {"type": "offline contemporaneous displacement-conditioned lagged ridge", "ridge": RIDGE, "coefficients": beta.tolist(),
                  "training_force_robust_scale": scale, "calibration_threshold_95": threshold},
        "primary": {"independent_records": len(run_df),
                    "median_model_nrmse": float(run_df.model_nrmse.median()),
                    "median_persistence_nrmse": float(run_df.persistence_nrmse.median()),
                    "median_paired_nrmse_reduction": float(np.median(improvement)),
                    "bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    "post_review_run_cluster_sensitivity_95_ci":
                    [float(np.quantile(boot_cluster, .025)), float(np.quantile(boot_cluster, .975))]},
        "fault_replay": {"rows": len(fault_df), "median_auprc_by_fault_level":
                         fault_df.groupby(["fault", "level"]).auprc.median().unstack().to_dict()},
        "claim_boundary": "offline residual transfer on RTHS measurements; not O1-O3/C1 deployment",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {}
    for p in sorted(args.out.glob("*")):
        if p.name != "SHA256SUMS.json":
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary["primary"], indent=2))


if __name__ == "__main__":
    main()
