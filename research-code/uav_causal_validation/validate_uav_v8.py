#!/usr/bin/env python3
"""Fail-closed independent validation of the frozen UAV V8 matrix."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
Q = json.loads((HERE / "UAV_V8_DENSE_PREREGISTRATION.json").read_text())
R = json.loads((OUT / "uav_v8_validation.json").read_text())
dense = list(csv.DictReader((OUT / "uav_v8_dense_runs.csv").open()))
abl = list(csv.DictReader((OUT / "uav_v8_mechanism_ablations.csv").open()))

def key(r: dict) -> tuple:
    return (r["policy"], float(r["rho"]), r["environment"], int(r["seed"]), r["arm"])

expected_dense = {(p,float(r),e,int(s),"all_heterogeneous") for e in Q["environments"]
                  for s in Q["seeds"] for r in Q["rho_grid"] for p in Q["policies"]}
expected_abl = {("all_coupled_with_trim_sharing",float(r),e,int(s),a)
                for e in Q["environments"] for s in Q["seeds"]
                for r in Q["mechanism_witness_rho"] for a in Q["mechanism_arms"]}
assert len(dense)==len(expected_dense) and {key(r) for r in dense}==expected_dense
assert len(abl)==len(expected_abl) and {key(r) for r in abl}==expected_abl
assert len({key(r) for r in dense})==len(dense) and len({key(r) for r in abl})==len(abl)

numeric = ["rho","task_success","physical_task_success","tube_success","completion_fraction",
           "minimum_pair_clearance_m","maximum_corridor_excursion_m","peak_tracking_error_m",
           "tail_tracking_error_m","control_energy","communication_messages"]
for row in dense + abl:
    for field in numeric: assert math.isfinite(float(row[field]))
    physical = float(row["minimum_pair_clearance_m"]) >= .34 and float(row["maximum_corridor_excursion_m"]) <= 1.60 and float(row["completion_fraction"]) >= .8
    tube = float(row["tail_tracking_error_m"]) <= .14
    assert int(row["physical_task_success"]) == int(physical)
    assert int(row["tube_success"]) == int(tube)
    assert int(row["task_success"]) == int(physical and tube)

def rate(rows: list[dict], policy: str, rho: float, arm: str | None=None) -> float:
    z=[int(r["task_success"]) for r in rows if r["policy"]==policy and float(r["rho"])==rho and (arm is None or r["arm"]==arm)]
    assert z
    return sum(z)/len(z)

allp, gate, independent = Q["policies"]
curve={rho:(rate(dense,allp,rho),rate(dense,gate,rho),rate(dense,independent,rho)) for rho in map(float,Q["rho_grid"])}
checks={
 "low_information_region":all(1-v[0]>.5 for rho,v in curve.items() if rho<=.55),
 "intermediate_feasible_region":any(v[0]>.8 for rho,v in curve.items() if .95<=rho<=1.4),
 "high_mismatch_region":any(1-v[0]>.5 for rho,v in curve.items() if rho>=2.2),
 "gate_recovery":any(v[1]-v[0]>=.5 for rho,v in curve.items() if rho>=2.2),
 "homogeneous_counterexample":rate(abl,allp,2.5,"homogeneous")>.8,
 "low_information_ablation":1-rate(abl,allp,.5,"all_heterogeneous_without_feedforward_sharing")>.5}
assert all(checks.values())
assert R["preregistration_sha256"]==hashlib.sha256((HERE/"UAV_V8_DENSE_PREREGISTRATION.json").read_bytes()).hexdigest()
assert R["n_dense_runs"]==len(dense) and R["n_ablation_runs"]==len(abl)
assert R["status"]=="V8_SAME_SIMULATOR_CONFIRMATORY_EXTENSION_PASSED"
print(f"PASS UAV V8: {len(dense)} dense + {len(abl)} ablation runs; all frozen checks independently recomputed")
