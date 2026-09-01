#!/usr/bin/env python3
from pathlib import Path
import csv,json,statistics

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
rows=list(csv.DictReader(open(OUT/'v20_runs.csv')))
held={str(x) for x in json.load(open(HERE/'V20_FROZEN_CONTRACT.json'))['heldout_seeds']}
rows=[r for r in rows if r['seed'] in held]
summary={}; windows={}
for d in sorted({r['domain'] for r in rows}):
  dr=[r for r in rows if r['domain']==d]
  summary[d]={}
  for p in sorted({r['policy'] for r in dr}):
    x=[r for r in dr if r['policy']==p]
    summary[d][p]={'n':len(x),'success_rate':sum(int(r['task_success']) for r in x)/len(x),'median_margin':statistics.median(float(r['minimum_physical_margin']) for r in x),'median_energy':statistics.median(float(r['control_energy']) for r in x),'median_messages':statistics.median(float(r['communication_messages']) for r in x)}
  curve=[]
  for rho in sorted({float(r['rho']) for r in dr}):
    x=[r for r in dr if r['policy']=='all_coupled' and float(r['rho'])==rho]
    curve.append({'rho':rho,'n':len(x),'success_rate':sum(int(r['task_success']) for r in x)/len(x),'median_margin':statistics.median(float(r['minimum_physical_margin']) for r in x)})
  best=max(range(len(curve)),key=lambda i:(curve[i]['success_rate'],curve[i]['median_margin']))
  windows[d]={'curve':curve,'best_rho':curve[best]['rho'],'finite_window_witness':best not in (0,len(curve)-1) and curve[best]['success_rate']>curve[0]['success_rate'] and curve[best]['success_rate']>curve[-1]['success_rate']}
payload={'scope':'heldout only','rows':len(rows),'summary':summary,'participation_curves':windows}
(OUT/'v20_analysis.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
