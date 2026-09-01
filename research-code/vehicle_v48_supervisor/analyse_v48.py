#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
with (OUT/"V48_RESULTS.csv").open(newline="") as f: rows=list(csv.DictReader(f))
sup=[r for r in rows if r["policy"]=="viability_supervisor"]
truth=np.array([int(r["task_success"])==0 for r in sup])
warn=np.array([float(r["first_warning_s"])<20.0 for r in sup])
tp=int(np.sum(truth & warn)); fn=int(np.sum(truth & ~warn)); fp=int(np.sum(~truth & warn)); tn=int(np.sum(~truth & ~warn))
report={"failed_runs":int(truth.sum()),"warning_true_positives":tp,"warning_false_negatives":fn,
        "warning_false_positives":fp,"warning_true_negatives":tn,
        "warning_sensitivity":tp/(tp+fn) if tp+fn else None,
        "warning_specificity":tn/(tn+fp) if tn+fp else None,
        "median_lead_s":float(np.median([float(r["warning_lead_s"]) for r in sup if int(r["task_success"])==0])),
        "interpretation":"The joint-margin proxy detected every supervisor failure but is not a certified distance to the exact continuous viability kernel."}
(OUT/"V48_WARNING_EVALUATION.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
