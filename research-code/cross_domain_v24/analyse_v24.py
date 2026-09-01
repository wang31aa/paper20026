#!/usr/bin/env python3
import csv,json
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'results/v24_runs.csv').open()))
held=[r for r in rows if r['split']=='heldout']
curve=defaultdict(list); policies=defaultdict(list)
for r in held:
    if r['topology']=='certified_switching' and r['policy']=='all_coupled':
        curve[(r['domain'],float(r['rho']))].append(int(r['task_success']))
    if r['topology']=='certified_switching': policies[(r['domain'],r['policy'])].append(int(r['task_success']))
out={
 'heldout_rows':len(held),
 'all_coupled_certified_switching_curve':{d:{str(rho):sum(v)/len(v) for (dd,rho),v in sorted(curve.items()) if dd==d} for d in sorted({k[0] for k in curve})},
 'certified_switching_policy_success':{d:{p:sum(v)/len(v) for (dd,p),v in sorted(policies.items()) if dd==d} for d in sorted({k[0] for k in policies})},
 'interpretation':'Held-out software-in-the-loop outcomes under a directed switching family with G=I certified mode by mode; not an exact modal reduction, HIL or physical evidence.'}
(P/'results/v24_analysis.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
