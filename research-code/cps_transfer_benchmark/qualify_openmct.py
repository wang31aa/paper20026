#!/usr/bin/env python3
"""Strict qualification gate for the public OpenMCT V1 model."""
from __future__ import annotations

import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from run_openmct import RIDGE, design, fit_ridge, predict, read_log, robust_scale

EXPECTED_ARCHIVE_SHA256 = "0f1781b7443dc5f6cfaab8e8cb473ca84832f8d8e4ad8e0b368da09d1a1a06df"
FEATURES = ("speed_k", "speed_km1", "pwm_k", "reference_kp1", "reference_k", "dt_k")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_fit(x, y, mask):
    mean, scale, beta = fit_ridge(x[:, mask], y)
    return mask, mean, scale, beta


def model_predict(x, fitted):
    mask, mean, scale, beta = fitted
    return predict(x[:, mask], mean, scale, beta)


def free_run(df: pd.DataFrame, fitted):
    """Roll the lifted ARX state without measured speed after k=1."""
    y, ref = df.MEAS.to_numpy(float), df.REF.to_numpy(float)
    pwm, dt = df.PWM.to_numpy(float), df.DT_ms.to_numpy(float)
    simulated = np.empty(len(y) - 2)
    ykm1, yk = float(y[0]), float(y[1])
    for out_idx, k in enumerate(range(1, len(y) - 1)):
        row = np.array([[yk, ykm1, pwm[k], ref[k + 1], ref[k], dt[k]]])
        nxt = float(model_predict(row, fitted)[0])
        simulated[out_idx] = nxt
        ykm1, yk = yk, nxt
    return y[2:], simulated


def source_paths(raw: Path):
    train = sorted((raw / "03_system_identification").glob("* ms sampling time/raw_data.txt"))
    tests = sorted((raw / "04_continuous_PID_validation").glob("*_ms/raw_data_*.txt"))
    tests += sorted((raw / "05_discrete_controller_validation").glob("Case_*/raw_data.txt"))
    if len(train) != 3 or len(tests) != 7:
        raise ValueError(f"expected 3 training and 7 test records, got {len(train)} and {len(tests)}")
    return train, tests


def bind_extracted_files_to_archive(archive: Path, raw: Path, selected):
    """Require each analysed extract to be byte-identical to one zip member."""
    bindings = {}
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        for path in selected:
            rel = path.relative_to(raw).as_posix()
            matches = [name for name in members if name == rel or name.endswith("/" + rel)]
            if len(matches) != 1:
                raise ValueError(f"expected one archive member ending in {rel}, found {len(matches)}")
            member = matches[0]
            archive_hash = hashlib.sha256(zf.read(member)).hexdigest()
            extracted_hash = sha256(path)
            if archive_hash != extracted_hash:
                raise ValueError(f"extracted file differs from archive member: {rel}")
            bindings[rel] = {"archive_member": member, "sha256": extracted_hash}
    return bindings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    observed = sha256(args.archive)
    if observed != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {observed}")
    train_paths, test_paths = source_paths(args.raw)
    archive_bindings = bind_extracted_files_to_archive(
        args.archive, args.raw, train_paths + test_paths)
    input_hashes = {rel: item["sha256"] for rel, item in archive_bindings.items()}
    train = [design(read_log(p)) for p in train_paths]
    xt = np.vstack([z[0] for z in train]); yt = np.concatenate([z[1] for z in train])
    yscale = robust_scale(yt)
    masks = {"full": np.ones(6, bool), "no_pwm": np.array([1,1,0,1,1,1], bool),
             "no_dt": np.array([1,1,1,1,1,0], bool)}
    fitted = {name: model_fit(xt, yt, mask) for name, mask in masks.items()}
    for idx, path in enumerate(train_paths):
        fitted[f"single_interval_{path.parent.name}"] = model_fit(train[idx][0], train[idx][1], masks["full"])
    rows = []
    for path in test_paths:
        df = read_log(path); x, y, *_ = design(df); persistence = x[:, 0]
        row = {"file": str(path.relative_to(args.raw)), "n": len(y),
               "persistence_nrmse": float(np.sqrt(np.mean((y-persistence)**2))/yscale)}
        for name, fit in fitted.items():
            one = model_predict(x, fit); truth, rolled = free_run(df, fit)
            row[f"{name}_one_step_nrmse"] = float(np.sqrt(np.mean((y-one)**2))/yscale)
            row[f"{name}_free_run_nrmse"] = float(np.sqrt(np.mean((truth-rolled)**2))/yscale)
            row[f"{name}_free_run_finite"] = bool(np.isfinite(rolled).all())
        rows.append(row)
    table = pd.DataFrame(rows); table.to_csv(args.out / "openmct_qualification_metrics.csv", index=False)
    result = {
      "evidence_label": "public-data-constrained model qualification; not deployment",
      "archive": {"sha256": observed, "expected_sha256": EXPECTED_ARCHIVE_SHA256},
      "analysed_input_sha256": input_hashes,
      "archive_to_extract_bindings": archive_bindings,
      "causality": {"features": list(FEATURES), "target": "speed_kp1",
                    "future_measured_speed_in_features": False,
                    "target_row_dt_in_features": False,
                    "reference_kp1_timing_verified": False,
                    "reference_timing_note": "source acquisition ordering must prove the command precedes the measured response"},
      "split": {"train": [str(p.relative_to(args.raw)) for p in train_paths],
                "test": [str(p.relative_to(args.raw)) for p in test_paths]},
      "gate": {"all_full_one_step_better_than_persistence": bool((table.full_one_step_nrmse < table.persistence_nrmse).all()),
               "all_full_free_run_finite": bool(table.full_free_run_finite.all()),
               "closed_loop_network_authorized": False,
               "reason": "numeric free-run and operating-domain thresholds are not yet frozen"},
      "ridge": RIDGE}
    (args.out / "openmct_qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__": main()
