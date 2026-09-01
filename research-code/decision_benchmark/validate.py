#!/usr/bin/env python3
import csv,json,sys
from statistics import mean
from pathlib import Path
paths=[Path(x) for x in sys.argv[1:]] or [Path("raw")]
failures=[]
def validate_one(p):
 global rows,s
 rows=list(csv.DictReader((p/"decision_runs.csv").open()))
 s=json.loads((p/"summary.json").read_text())
 assert len(rows)==4*30*4, len(rows)
 assert set(r["strategy"] for r in rows)=={"oracle","full","no_model","uncertified"}
 assert s["label"].startswith("synthetic high-fidelity")
 for r in rows:
  assert 0<=float(r["max_abs_frequency"])<5
  assert float(r["unsafe_time"])>=0
  assert int(r["false_alarm"])<=int(r["action"])
 audit=list(csv.DictReader((p/"dt_audit.csv").open()))
 assert len(audit)==64
 maxdiff=max(float(r["abs_difference"]) for r in audit)
 agree=sum(int(r["action_agreement"]) for r in audit)/len(audit)
 if not maxdiff<0.02: failures.append(f"{p}: dt peak gate FAIL ({maxdiff:.8g} >= 0.02 Hz)")
 if not agree>=.95: failures.append(f"{p}: dt action gate FAIL ({agree:.3f} < 0.95)")
 for strategy in ("oracle","full","no_model","uncertified"):
  rr=[r for r in rows if r["strategy"]==strategy]; got=s["summary"][strategy]
  checks={"false_alarm_rate":mean(int(r["false_alarm"]) for r in rr),
          "missed_violation_rate":mean(int(r["missed_violation"]) for r in rr),
          "unsafe_case_rate":mean(float(r["unsafe_time"])>0 for r in rr),
          "mean_unsafe_time":mean(float(r["unsafe_time"]) for r in rr),
          "mean_utility":mean(float(r["utility"]) for r in rr),
          "action_rate":mean(int(r["action"]) for r in rr)}
  for k,v in checks.items(): assert abs(got[k]-v)<1e-12,(strategy,k,got[k],v)
 assert len({(r["seed"],r["scenario"]) for r in rows})==120
 return rows
allrows=[validate_one(p) for p in paths]
if len(paths)==2:
 assert not ({r["seed"] for r in allrows[0]} & {r["seed"] for r in allrows[1]})
 assert {int(r["seed"]) for r in allrows[1]}==set(range(1000,1030))
print(f"PASS: {len(paths)} phase(s), schema, paired cases, recomputed summaries, invariants, held-out seeds")
if failures:
 for x in failures: print(x)
 print("OVERALL: FAIL (pre-registered numerical gate)")
 sys.exit(1)
print("PASS: all pre-registered numerical gates")
