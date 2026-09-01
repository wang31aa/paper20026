#!/usr/bin/env python3
from pathlib import Path
import csv,json,statistics
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
rows=list(csv.DictReader(open(OUT/'v21_runs.csv'))); held=[r for r in rows if r['split']=='heldout']
summary={}
for p in sorted({r['policy'] for r in held}):
    x=[r for r in held if r['policy']==p]
    summary[p]={'n':len(x),'success_rate':sum(int(r['task_success']) for r in x)/len(x),
      'median_margin':statistics.median(float(r['minimum_physical_margin']) for r in x),
      'median_energy':statistics.median(float(r['control_energy']) for r in x)}
curve=[]
for rho in sorted({float(r['rho']) for r in held}):
    x=[r for r in held if r['policy']=='all_coupled_pi' and float(r['rho'])==rho]
    curve.append({'rho':rho,'n':len(x),'success_rate':sum(int(r['task_success']) for r in x)/len(x),
      'median_margin':statistics.median(float(r['minimum_physical_margin']) for r in x)})
best=max(range(len(curve)),key=lambda i:(curve[i]['success_rate'],curve[i]['median_margin']))
payload={'scope':'heldout only','rows':len(held),'summary':summary,'participation_curve':curve,'best_rho':curve[best]['rho'],
 'finite_window_witness':best not in (0,len(curve)-1) and curve[best]['success_rate']>curve[0]['success_rate'] and curve[best]['success_rate']>curve[-1]['success_rate']}
(OUT/'v21_analysis.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
