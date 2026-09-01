#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
run = pd.read_csv(OUT / "rths_run_metrics.csv")
fault = pd.read_csv(OUT / "rths_fault_metrics.csv")
summary = json.loads((OUT / "summary.json").read_text())
hashes = json.loads((OUT / "SHA256SUMS.json").read_text())
assert len(run) == 14 and set(run.run_id) == set(range(16, 23))
assert len(fault) == 224 and set(fault.level) == {1, 2, 3, 4}
assert set(fault.fault) == {"force_bias", "force_gain", "force_dropout", "force_noise"}
assert ((fault.positive_prevalence > 0) & (fault.positive_prevalence < 1)).all()
assert np.isfinite(run.select_dtypes("number")).all().all()
assert np.isfinite(fault.select_dtypes("number")).all().all()
assert abs(run.model_nrmse.median() - summary["primary"]["median_model_nrmse"]) < 1e-12
assert abs(run.persistence_nrmse.median() - summary["primary"]["median_persistence_nrmse"]) < 1e-12
for key, value in fault.groupby(["fault", "level"]).auprc.median().items():
    assert abs(value - summary["fault_replay"]["median_auprc_by_fault_level"][str(key[1])][key[0]]) < 1e-12
improvement = run.persistence_nrmse - run.model_nrmse
run_improvement = run.assign(improvement=improvement).groupby("run_id").improvement.median()
rng = np.random.default_rng(20260730)
boot, cluster = [], []
for _ in range(10000):
    boot.append(float(np.median(rng.choice(improvement, len(improvement), replace=True))))
    cluster.append(float(np.median(rng.choice(run_improvement, len(run_improvement), replace=True))))
assert np.allclose(np.quantile(boot, [.025, .975]), summary["primary"]["bootstrap_95_ci"])
assert np.allclose(np.quantile(cluster, [.025, .975]), summary["primary"]["post_review_run_cluster_sensitivity_95_ci"])
for name, expected in hashes.items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
print("PASS: RTHS grouped split, 14 held-out actuator records, 224 synthetic-fault replays and hashes validated")

water = pd.read_csv(OUT / "water_hil_session_metrics.csv")
water_summary = json.loads((OUT / "water_hil_summary.json").read_text())
water_hashes = json.loads((OUT / "water_hil_SHA256SUMS.json").read_text())
assert 3 <= len(water) <= 4
assert set(water.condition).issubset({"level_sensor_spoof", "flow_sensor_spoof", "fill_valve_spoof", "display_spoof"})
assert np.isfinite(water.select_dtypes("number")).all().all()
assert water_summary["normal"]["threshold_exceedance_rate"] <= 0.011
assert water_summary["split"]["test"] == water.file.tolist()
for row, frozen in zip(water.to_dict("records"), water_summary["sessions"]):
    for key in ["n", "median_residual", "median_ratio_to_normal",
                "threshold_exceedance_rate", "common_language_probability"]:
        assert np.isclose(row[key], frozen[key]), (row["file"], key)
for name, expected in water_hashes.items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
print(f"PASS: water-HIL grouped train/calibration and {len(water)} weakly-labelled spoof sessions validated")

motor = pd.read_csv(OUT / "openmct_run_metrics.csv")
motor_rep = pd.read_csv(OUT / "openmct_representative_10ms.csv")
motor_summary = json.loads((OUT / "openmct_summary.json").read_text())
motor_hashes = json.loads((OUT / "openmct_SHA256SUMS.json").read_text())
assert len(motor) == 7 and set(motor.family) == {"continuous_PI", "discrete"}
assert len(motor_rep) > 400 and np.isfinite(motor_rep.select_dtypes("number")).all().all()
assert motor_summary["source"]["archive_sha256"] == "0f1781b7443dc5f6cfaab8e8cb473ca84832f8d8e4ad8e0b368da09d1a1a06df"
improvement = motor.persistence_nrmse - motor.model_nrmse
assert np.isclose(improvement.median(), motor_summary["primary"]["median_paired_nrmse_reduction"])
rng = np.random.default_rng(20260731)
boot = [float(np.median(rng.choice(improvement, len(improvement), replace=True))) for _ in range(10000)]
assert np.allclose(np.quantile(boot, [.025, .975]), motor_summary["primary"]["record_bootstrap_95_ci"])
for name, expected in motor_hashes.items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
print("PASS: OpenMCT 3-run training, 7 held-out physical motor records and hashes validated")
