#!/usr/bin/env python3
"""Create machine-readable V8 summaries without refitting any threshold."""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
dense=list(csv.DictReader((OUT/"uav_v8_dense_runs.csv").open()))
abl=list(csv.DictReader((OUT/"uav_v8_mechanism_ablations.csv").open()))

def wilson(k:int,n:int,z:float=1.959963984540054)->tuple[float,float]:
    p=k/n; den=1+z*z/n; mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

curves=[]
for rho in sorted({float(r["rho"]) for r in dense}):
    for policy in sorted({r["policy"] for r in dense}):
        z=[r for r in dense if float(r["rho"])==rho and r["policy"]==policy]
        k=sum(int(r["task_success"]) for r in z); lo,hi=wilson(k,len(z))
        curves.append({"rho":rho,"policy":policy,"n":len(z),"successes":k,"success_rate":k/len(z),
                       "wilson95_low":lo,"wilson95_high":hi,
                       "median_tail_error_m":statistics.median(float(r["tail_tracking_error_m"]) for r in z),
                       "median_control_energy":statistics.median(float(r["control_energy"]) for r in z),
                       "median_communication_messages":statistics.median(float(r["communication_messages"]) for r in z),
                       "median_gate_active_steps":statistics.median(float(r["gate_active_steps"]) for r in z)})
with (OUT/"uav_v8_curve_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)

mechanism=[]
for rho in sorted({float(r["rho"]) for r in abl}):
    for arm in sorted({r["arm"] for r in abl}):
        z=[r for r in abl if float(r["rho"])==rho and r["arm"]==arm]
        mechanism.append({"rho":rho,"arm":arm,"n":len(z),"success_rate":sum(int(r["task_success"]) for r in z)/len(z),
                          "median_tail_error_m":statistics.median(float(r["tail_tracking_error_m"]) for r in z)})
with (OUT/"uav_v8_mechanism_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(mechanism[0]));w.writeheader();w.writerows(mechanism)

summary={"interpretation":{
 "low_end":"All four arms fail at rho=0.5, including the homogeneous arm; the low-end loss is therefore attributable to target-information age rather than heterogeneity.",
 "high_end":"At rho=2.5 the homogeneous and no-shared-feedforward arms succeed, while trim-only and all-heterogeneous arms fail; the high-end loss requires transmitted persistent trim mismatch in this simulator.",
 "recovery":"The gate restores high-participation cooperative tracking, but independent constrained tracking also succeeds; V8 does not establish uniqueness or superiority of gating.",
 "communication":"Message counts are unchanged by design because the target-rooted information backbone remains active while physical influence weights change."},
 "scope":"same-simulator confirmatory resolution; not an independent domain, official NMPC replay, HIL or hardware validation"}
(OUT/"uav_v8_interpretation.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
