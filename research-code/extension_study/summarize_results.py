#!/usr/bin/env python3
"""Descriptive summaries only; never filters or changes raw runs."""
import csv, math, statistics
from pathlib import Path

ROOT=Path(__file__).parent/'results'
rows=list(csv.DictReader((ROOT/'raw_runs.csv').open()))
fields=['model','n','scenario','runs','failures','median_final_tracking','max_final_tracking',
        'median_final_observer','max_final_observer','median_tail20_tracking',
        'median_final_tracking_per_sqrt_n','median_tail20_tracking_per_sqrt_n']
summary=[]
for model in ('chua','lorenz'):
 for n in (8,16,32):
  for scenario in ('clean','noise','delay','dropout'):
   z=[r for r in rows if r['model']==model and int(r['n'])==n and r['scenario']==scenario]
   finite=lambda key:[float(r[key]) for r in z if math.isfinite(float(r[key]))]
   ft,fo,tt=finite('final_tracking'),finite('final_observer'),finite('tail20_max_tracking')
   fn,tn=finite('final_tracking_per_sqrt_n'),finite('tail20_max_tracking_per_sqrt_n')
   summary.append(dict(model=model,n=n,scenario=scenario,runs=len(z),failures=sum(bool(r['failure_reason']) for r in z),
    median_final_tracking=statistics.median(ft),max_final_tracking=max(ft),median_final_observer=statistics.median(fo),
    max_final_observer=max(fo),median_tail20_tracking=statistics.median(tt),
    median_final_tracking_per_sqrt_n=statistics.median(fn),median_tail20_tracking_per_sqrt_n=statistics.median(tn)))
with (ROOT/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)

audit=list(csv.DictReader((ROOT/'dt_audit.csv').open()))
conv=[]
for model in ('chua','lorenz'):
 for scenario in ('clean','noise','delay','dropout'):
  ds=[]; unavailable=0
  for n in (8,16,32):
   z={float(r['dt']):r for r in audit if r['model']==model and int(r['n'])==n and r['scenario']==scenario}
   a,b=float(z[.001]['final_tracking']),float(z[.002]['final_tracking'])
   failed=bool(z[.001]['failure_reason']) or bool(z[.002]['failure_reason'])
   if not failed and math.isfinite(a) and math.isfinite(b): ds.append(abs(a-b)/max(abs(a),1e-12))
   else: unavailable+=1
  conv.append(dict(model=model,scenario=scenario,status=('available' if ds else 'unavailable'),
   median_relative_endpoint_difference_dt002_vs_dt001=(statistics.median(ds) if ds else ''),
   available_sizes=len(ds),unavailable_sizes=unavailable))
with (ROOT/'dt_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=conv[0]);w.writeheader();w.writerows(conv)

failure=[]
for model in ('chua','lorenz'):
 for n in (8,16,32):
  for scenario in ('clean','noise','delay','dropout'):
   z=sorted((r for r in audit if r['model']==model and int(r['n'])==n and r['scenario']==scenario),key=lambda r:float(r['dt']),reverse=True)
   times=[float(r['failure_time']) for r in z]
   failure.append(dict(model=model,n=n,scenario=scenario,failed_dt004=bool(z[0]['failure_reason']),
    failed_dt002=bool(z[1]['failure_reason']),failed_dt001=bool(z[2]['failure_reason']),
    failure_time_dt004=times[0],failure_time_dt002=times[1],failure_time_dt001=times[2],
    failure_time_range=max(times)-min(times)))
with (ROOT/'dt_failure_sensitivity.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=failure[0]);w.writeheader();w.writerows(failure)
