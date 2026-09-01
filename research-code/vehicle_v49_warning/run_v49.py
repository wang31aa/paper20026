#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"vehicle_v48_supervisor")); sys.path.insert(0,str(ROOT/"vehicle_v46"))
from run_v48 import calibration, run_supervisor  # noqa: E402
from run_v46 import load_source  # noqa: E402

SPEC=json.loads((HERE/"V49_STAGE1_CONTRACT.json").read_text()); OUT=HERE/"results"


def runs(source,seeds,threshold):
    high,low,gap=calibration(source); out=[]
    for seed in seeds:
      for stress in SPEC["stress"]:
       for delay in SPEC["delay_s"]:
        for rho in SPEC["participation"]:
         out.append(run_supervisor(source,stress,delay,rho,seed,high,low,gap,
                                   warning_mode="physical",warning_threshold=threshold))
    return out


def metrics(rows):
    fail=np.array([not bool(r["task_success"]) for r in rows]); warn=np.array([r["first_warning_s"]<20 for r in rows])
    tp=int(np.sum(fail&warn)); fn=int(np.sum(fail&~warn)); fp=int(np.sum(~fail&warn)); tn=int(np.sum(~fail&~warn))
    sensitivity=tp/(tp+fn) if tp+fn else 1.0; specificity=tn/(tn+fp) if tn+fp else 1.0
    leads=[r["warning_lead_s"] for r in rows if not r["task_success"] and r["warning_lead_s"]>0]
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"sensitivity":sensitivity,"specificity":specificity,
            "youden":sensitivity+specificity-1,"median_lead_s":float(np.median(leads)) if leads else 0.0}


def main():
    source=load_source(); OUT.mkdir(parents=True,exist_ok=True); candidates=[]
    for threshold in SPEC["candidate_thresholds_m"]:
        m=metrics(runs(source,SPEC["development_seeds"],threshold)); m["threshold_m"]=threshold; candidates.append(m)
    eligible=[x for x in candidates if x["sensitivity"]>=.8]
    chosen=sorted(eligible or candidates,key=lambda x:(-x["youden"],x["threshold_m"]))[0]
    frozen={"contract_sha256":hashlib.sha256((HERE/"V49_STAGE1_CONTRACT.json").read_bytes()).hexdigest(),
            "candidate_metrics":candidates,"chosen_threshold_m":chosen["threshold_m"],"selection_rule":SPEC["selection_rule"]}
    pred=OUT/"V49_FROZEN_WARNING.json"; pred.write_text(json.dumps(frozen,indent=2)+"\n")
    heldout=runs(source,SPEC["heldout_seeds"],chosen["threshold_m"]); evaluation=metrics(heldout)
    report={"prediction_sha256":hashlib.sha256(pred.read_bytes()).hexdigest(),"chosen_threshold_m":chosen["threshold_m"],
            "development":chosen,"heldout":evaluation,"development_runs":len(SPEC["development_seeds"])*27,
            "heldout_runs":len(heldout),"claim_boundary":SPEC["claim_boundary"]}
    (OUT/"V49_HELDOUT_WARNING.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
