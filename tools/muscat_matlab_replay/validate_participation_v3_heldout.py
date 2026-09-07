#!/usr/bin/env python3
"""Evaluate the V2 response prediction on frozen, unopened V3 conditions."""
from __future__ import annotations
import csv, json, math, pathlib, statistics, sys

RHO=(0.10,0.20,0.40,0.80,1.60,2.40); CONDITIONS=range(1,9)
POLICIES=("permanent","two_layer"); THRESHOLD=2.5e-4

def read_metrics(path:pathlib.Path, condition:int, rho:float)->dict:
    with path.open(newline="") as handle: rows=[[float(x) for x in row] for row in csv.reader(handle)]
    if len(rows)!=360 or any(len(row)!=17 for row in rows): raise AssertionError(f"{path.name}: shape")
    if not all(math.isfinite(x) for row in rows for x in row): raise AssertionError(f"{path.name}: nonfinite")
    if any(int(row[0])!=condition or abs(row[1]-rho)>1e-12 for row in rows): raise AssertionError(f"{path.name}: identifiers")
    if not any(row[16]>0 for row in rows): raise AssertionError(f"{path.name}: zero applied torque")
    task=[math.sqrt(sum((row[8+j]-row[5+j])**2 for j in range(3))) for row in rows]
    observer=[math.sqrt(sum((row[11+j]-row[5+j])**2 for j in range(3))) for row in rows]
    tail=task[int(.8*len(task)):]
    observer_tail=observer[int(.8*len(observer)):]
    return {"tail_max":max(tail),"success":max(tail)<=THRESHOLD,
            "observer_tail_max":max(observer_tail),
            "edge_time":sum(row[15] for row in rows if int(row[4])==1),
            "energy":sum(row[16]**2 for row in rows),"visible_node":int(rows[0][2])}

def main(directory:str)->None:
    base=pathlib.Path(directory); raw={p:{} for p in POLICIES}
    for c in CONDITIONS:
      for i,rho in enumerate(RHO,1):
       for p in POLICIES:
        raw[p][(c,rho)]=read_metrics(base/f"muscat_heldout_c{c:02d}_r{i:02d}_{p}.csv",c,rho)
    summary={}
    for p in POLICIES:
      summary[p]=[]
      for rho in RHO:
        q=[raw[p][(c,rho)] for c in CONDITIONS]
        summary[p].append({"rho":rho,"successes":sum(x["success"] for x in q),
          "median_tail_max_rad_s":statistics.median(x["tail_max"] for x in q),
          "median_edge_time_s":statistics.median(x["edge_time"] for x in q),
          "median_energy_Nm2_s":statistics.median(x["energy"] for x in q)})
    # Post-hoc mechanism audit. These strata diagnose a failed frozen prediction;
    # they are not promoted to preregistered endpoints.
    diagnostic={"by_visible_node":{},"condition_transition":[]}
    for visible in (1,2):
      diagnostic["by_visible_node"][str(visible)]=[]
      selected=[c for c in CONDITIONS if raw["permanent"][(c,RHO[0])]["visible_node"]==visible]
      for rho in RHO:
        q=[raw["permanent"][(c,rho)] for c in selected]
        diagnostic["by_visible_node"][str(visible)].append({
          "rho":rho,"conditions":len(q),"successes":sum(x["success"] for x in q),
          "median_task_tail_max_rad_s":statistics.median(x["tail_max"] for x in q),
          "median_observer_tail_max_rad_s":statistics.median(x["observer_tail_max"] for x in q)})
    for c in CONDITIONS:
      q=[raw["permanent"][(c,rho)] for rho in RHO]
      first=next((rho for rho,x in zip(RHO,q) if x["success"]),None)
      diagnostic["condition_transition"].append({"condition":c,"visible_node":q[0]["visible_node"],
        "first_feasible_rho":first,"success_sequence":[bool(x["success"]) for x in q]})
    y=[x["median_tail_max_rad_s"] for x in summary["permanent"]]
    successes=[x["successes"] for x in summary["permanent"]]
    checks={"all_96_trajectories":len(list(base.glob('muscat_heldout_*.csv')))==96,
      "low_rho_information_failure":successes[0]<=2,
      "transition_by_rho_0_8":successes[3]>=7,
      "high_rho_feasible":successes[-1]>=7,
      "no_high_rho_decline":successes[-1]>=successes[-2],
      "median_tail_nonincreasing":all(y[i+1]<=y[i]+2e-8 for i in range(len(y)-1))}
    payload={"schema":"MUSCAT_PARTICIPATION_V3_HELDOUT","v2_model_refit":False,
      "prediction":"low-information transition by rho=0.8; monotone-or-saturating thereafter",
      "threshold_rad_s":THRESHOLD,"summary":summary,"checks":checks,
      "posthoc_mechanism_diagnostic":diagnostic,
      "heldout_prediction_qualified":all(checks.values()),
      "boundary":"source-class designed-heterogeneity OOD computation; not independent physical identification, HIL, flight, or cross-domain universality"}
    print('PARTICIPATION_V3_HELDOUT='+json.dumps(payload,sort_keys=True))

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: validate_participation_v3_heldout.py ARTIFACT_DIRECTORY')
    main(sys.argv[1])
