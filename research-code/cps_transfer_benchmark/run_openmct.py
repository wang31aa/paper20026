#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RIDGE = 1e-5
SEED = 20260731


def read_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    needed = {"REF", "MEAS", "DT_ms", "PWM"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} lacks {sorted(needed - set(df.columns))}")
    return df


def design(df: pd.DataFrame):
    y = df.MEAS.to_numpy(float)
    ref = df.REF.to_numpy(float)
    pwm = df.PWM.to_numpy(float)
    dt = df.DT_ms.to_numpy(float)
    x = np.column_stack([y[1:-1], y[:-2], pwm[1:-1], ref[2:], ref[1:-1], dt[1:-1]])
    return x, y[2:], ref[2:], pwm[2:], dt[2:]


def fit_ridge(x: np.ndarray, y: np.ndarray):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    za = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(za.shape[1]) * RIDGE
    penalty[0, 0] = 0
    beta = np.linalg.solve(za.T @ za + penalty, za.T @ y)
    return mean, scale, beta


def predict(x, mean, scale, beta):
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ beta


def robust_scale(y):
    q = np.quantile(y, [.25, .75])
    return max(float((q[1] - q[0]) / 1.349), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--archive-sha256", default="0f1781b7443dc5f6cfaab8e8cb473ca84832f8d8e4ad8e0b368da09d1a1a06df")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    train_paths = sorted((args.raw / "03_system_identification").glob("* ms sampling time/raw_data.txt"))
    tests = []
    for p in sorted((args.raw / "04_continuous_PID_validation").glob("*_ms/raw_data_*.txt")):
        tests.append(("continuous_PI", p.parent.name.replace("_ms", ""), p))
    for p in sorted((args.raw / "05_discrete_controller_validation").glob("Case_*/raw_data.txt")):
        tests.append(("discrete", p.parent.name.split("_", 2)[1].upper(), p))
    if len(train_paths) != 3 or len(tests) != 7:
        raise ValueError(f"expected 3 training and 7 test logs, got {len(train_paths)}, {len(tests)}")

    train_designs = [design(read_log(p)) for p in train_paths]
    xt = np.vstack([z[0] for z in train_designs]); yt = np.concatenate([z[1] for z in train_designs])
    mean, xscale, beta = fit_ridge(xt, yt); yscale = robust_scale(yt)

    rows = []; representative = None
    for family, condition, path in tests:
        x, y, ref, pwm, dt = design(read_log(path))
        model = predict(x, mean, xscale, beta); persistence = x[:, 0]
        ref_scale = max(float(np.max(np.abs(ref))), 1.0)
        tracking = y - ref
        tail = slice(int(.8 * len(y)), None)
        rows.append({
            "family": family, "condition": condition, "file": str(path.relative_to(args.raw)), "n": len(y),
            "model_nrmse": float(np.sqrt(np.mean((y-model)**2)) / yscale),
            "persistence_nrmse": float(np.sqrt(np.mean((y-persistence)**2)) / yscale),
            "tracking_nrmse": float(np.sqrt(np.mean(tracking**2)) / ref_scale),
            "tail20_tracking_nrmse": float(np.sqrt(np.mean(tracking[tail]**2)) / ref_scale),
            "pwm_rms": float(np.sqrt(np.mean(pwm**2))),
            "dt_median_ms": float(np.median(dt)), "dt_iqr_ms": float(np.subtract(*np.quantile(dt, [.75,.25]))),
        })
        if family == "continuous_PI" and condition == "10":
            representative = pd.DataFrame({"time_s": np.cumsum(dt) / 1000.0,
                                           "reference_rpm": ref, "measured_rpm": y,
                                           "model_prediction_rpm": model,
                                           "persistence_prediction_rpm": persistence,
                                           "pwm": pwm})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "openmct_run_metrics.csv", index=False)
    if representative is None:
        raise AssertionError("missing frozen 10-ms representative record")
    representative.iloc[::5].to_csv(args.out / "openmct_representative_10ms.csv", index=False)
    improvement = table.persistence_nrmse - table.model_nrmse
    rng = np.random.default_rng(SEED)
    boot = [float(np.median(rng.choice(improvement, len(improvement), replace=True))) for _ in range(10000)]
    summary = {
        "source": {"doi": "10.17632/5xvg43r9r8.1", "archive_sha256": args.archive_sha256,
                   "license": "CC-BY-4.0"},
        "split": {"train": [str(p.relative_to(args.raw)) for p in train_paths],
                  "test": table.file.tolist()},
        "model": {"type": "causal ARX ridge", "ridge": RIDGE, "training_rows": len(yt),
                  "training_speed_robust_scale": yscale, "feature_mean": mean.tolist(),
                  "feature_scale": xscale.tolist(), "coefficients": beta.tolist()},
        "primary": {"test_records": len(table), "median_model_nrmse": float(table.model_nrmse.median()),
                    "median_persistence_nrmse": float(table.persistence_nrmse.median()),
                    "median_paired_nrmse_reduction": float(np.median(improvement)),
                    "record_bootstrap_95_ci": [float(np.quantile(boot,.025)), float(np.quantile(boot,.975))]},
        "claim_boundary": "offline physical motor measurement-side error benchmark; not O1-O3/C1 deployment",
    }
    (args.out / "openmct_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [args.out / "openmct_run_metrics.csv", args.out / "openmct_representative_10ms.csv",
               args.out / "openmct_summary.json"]}
    (args.out / "openmct_SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary["primary"], indent=2))


if __name__ == "__main__":
    main()
