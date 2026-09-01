#!/usr/bin/env python3
"""Paired endpoint analysis for the frozen computational pressure matrix."""
import csv
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
s=list(csv.DictReader((OUT/'pressure_summary.csv').open()))
by={(r['domain'],r['condition'],r['policy']):r for r in s}
policies=sorted({r['policy'] for r in s if r['policy']!='all_coupled'})
rows=[]
for d,c,_ in sorted({(r['domain'],r['condition'],r['policy']) for r in s if r['policy']=='all_coupled'}):
 b=by[d,c,'all_coupled']
 for p in policies:
  q=by[d,c,p]
  be=float(b['control_energy']); qe=float(q['control_energy'])
  bm=float(b['communication_messages']); qm=float(q['communication_messages'])
  rows.append({
   'domain':d,'condition':c,'policy':p,
   'baseline_failure':b['failure'],'policy_failure':q['failure'],
   'failure_prevented':int(int(b['failure'])==1 and int(q['failure'])==0),
   'failure_induced':int(int(b['failure'])==0 and int(q['failure'])==1),
   'minimum_margin_change':float(q['minimum_margin'])-float(b['minimum_margin']),
   'tail_margin_change':float(q['tail_margin'])-float(b['tail_margin']),
   'control_energy_ratio':qe/be if be else float('nan'),
   'communication_ratio':qm/bm if bm else float('nan')})
with (OUT/'paired_policy_endpoints.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

agg=[]
for d in sorted({r['domain'] for r in rows}):
 for p in policies:
  a=[r for r in rows if r['domain']==d and r['policy']==p]; n=len(a)
  agg.append({'domain':d,'policy':p,'n_paired_conditions':n,
   'failure_prevented_n':sum(r['failure_prevented'] for r in a),
   'failure_induced_n':sum(r['failure_induced'] for r in a),
   'margin_improved_fraction':sum(r['minimum_margin_change']>0 for r in a)/n,
   'tail_improved_fraction':sum(r['tail_margin_change']>0 for r in a)/n,
   'median_control_energy_ratio':sorted(r['control_energy_ratio'] for r in a)[n//2],
   'median_communication_ratio':sorted(r['communication_ratio'] for r in a)[n//2]})
with (OUT/'paired_policy_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=agg[0]);w.writeheader();w.writerows(agg)
print(f'WROTE {len(rows)} paired endpoints and {len(agg)} domain-policy summaries')
