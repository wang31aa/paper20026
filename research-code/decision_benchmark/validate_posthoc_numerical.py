#!/usr/bin/env python3
"""Independent structural/arithmetic validator for the post-hoc dt audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import run_decision_benchmark as pb

ROOT=Path(__file__).resolve().parent; RAW=ROOT/"posthoc_numerical_raw"
DTS=(.0025,.00125,.000625,.0003125)
def close(a,b,tol=1e-12): assert abs(a-b)<=tol,(a,b)

def main():
 runs=list(csv.DictReader((RAW/"runs.csv").open()))
 comp=list(csv.DictReader((RAW/"adjacent_comparisons.csv").open()))
 summary=json.loads((RAW/"summary.json").read_text())
 archived=list(csv.DictReader((ROOT/"phase_b_raw"/"decision_runs.csv").open()))
 assert len(runs)==1920 and len(comp)==1440
 assert summary["n_runs"]==1920 and summary["n_comparisons"]==1440
 assert summary["phase_b_gate_preserved"]=="FAIL"
 assert tuple(summary["dt_levels"])==DTS
 cases={(s,k) for s in range(1000,1030) for k in pb.SCENARIOS}
 assert {(int(r["seed"]),r["scenario"]) for r in runs}==cases
 assert {r["strategy"] for r in runs}==set(pb.STRATEGIES)
 assert {float(r["dt"]) for r in runs}==set(DTS)
 assert len({(r["seed"],r["scenario"],r["strategy"],r["dt"]) for r in runs})==1920
 labels={}
 for r in archived:
  key=(int(r["seed"]),r["scenario"]); val=int(r["protection_required"])
  assert key not in labels or labels[key]==val; labels[key]=val
 assert set(labels)==cases
 for r in runs:
  key=(int(r["seed"]),r["scenario"]); par=pb.case_parameters(*key,"B")
  assert int(r["protection_required"])==labels[key]
  close(float(r["amplitude"]),par["amp"]); close(float(r["bscale"]),par["bscale"])
  assert int(r["disturbance_node"])==par["node"]
  assert 0<=float(r["max_abs_frequency"])<5 and float(r["unsafe_time"])>=0
 idx={(int(r["seed"]),r["scenario"],r["strategy"],float(r["dt"])):r for r in runs}
 recomputed=[]
 for coarse,fine in zip(DTS[:-1],DTS[1:]):
  level=[r for r in comp if float(r["coarse_dt"])==coarse]; assert len(level)==480
  for r in level:
   assert float(r["fine_dt"])==fine
   key=(int(r["seed"]),r["scenario"],r["strategy"])
   a,b=idx[key+(coarse,)],idx[key+(fine,)]
   close(float(r["coarse_peak"]),float(a["max_abs_frequency"]))
   close(float(r["fine_peak"]),float(b["max_abs_frequency"]))
   close(float(r["peak_abs_difference"]),abs(float(a["max_abs_frequency"])-float(b["max_abs_frequency"])))
   assert int(r["coarse_action"])==int(a["action"]); assert int(r["fine_action"])==int(b["action"])
   assert int(r["action_agreement"])==(int(a["action"])==int(b["action"]))
  d=np.array([float(r["peak_abs_difference"]) for r in level])
  timing=np.array([float(r["action_time_abs_difference"]) for r in level if r["action_time_abs_difference"]])
  recomputed.append(dict(max=float(d.max()),p99=float(np.quantile(d,.99)),median=float(np.quantile(d,.5)),
   agree=float(np.mean([int(r["action_agreement"]) for r in level])),
   disagree=sum(1-int(r["action_agreement"]) for r in level),both=len(timing),
   max_t=float(timing.max()),p99_t=float(np.quantile(timing,.99))))
 for got,w in zip(summary["adjacent_levels"],recomputed):
  close(got["max_peak_abs_difference"],w["max"]); close(got["p99_peak_abs_difference"],w["p99"])
  close(got["median_peak_abs_difference"],w["median"]); close(got["action_agreement_rate"],w["agree"])
  assert got["action_disagreements"]==w["disagree"] and got["both_act_n"]==w["both"]
  close(got["max_action_time_abs_difference"],w["max_t"]); close(got["p99_action_time_abs_difference"],w["p99_t"])
 prev,fine=recomputed[-2:]
 gates={"finest_max_peak_le_0.005":fine["max"]<=.005,
  "finest_p99_peak_le_0.001":fine["p99"]<=.001,
  "finest_action_agreement_ge_0.99":fine["agree"]>=.99,
  "max_peak_nonincreasing_at_finest":fine["max"]<=prev["max"],
  "p99_peak_nonincreasing_at_finest":fine["p99"]<=prev["p99"]}
 assert summary["frozen_gates"]==gates
 assert summary["numerical_resolution_closed"]==all(gates.values()) is False
 print("PASS: 1,920 complete runs, 1,440 comparisons, archived labels/parameters, recomputed summaries and frozen FAIL decision")
if __name__=="__main__": main()
