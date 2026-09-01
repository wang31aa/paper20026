#!/usr/bin/env python3
import csv, json, math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results'
rows=list(csv.DictReader((RESULTS/'v9_runs.csv').open()))
groups=defaultdict(list)
for r in rows: groups[(r['domain'],r['policy'],float(r['rho']))].append(r)
curve=[]
for (d,p,rho),g in sorted(groups.items()):
    n=len(g); k=sum(int(x['success']) for x in g); ph=k/n; z=1.96
    den=1+z*z/n; centre=(ph+z*z/(2*n))/den; half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    curve.append(dict(domain=d,policy=p,rho=rho,n=n,success_fraction=ph,wilson_low=centre-half,wilson_high=centre+half,
                      median_tail_error=sorted(float(x['tail_error']) for x in g)[n//2],
                      median_energy=sorted(float(x['control_energy']) for x in g)[n//2]))
with (RESULTS/'v9_domain_curves.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=curve[0]);w.writeheader();w.writerows(curve)
detail=defaultdict(list)
for r in rows: detail[(r['domain'],r['policy'],int(r['n']),r['topology'],float(r['rho']))].append(int(r['success']))
scales=[]
for key,g in sorted(detail.items()):
    d,p,n,t,rho=key; scales.append(dict(domain=d,policy=p,n=n,topology=t,rho=rho,success_fraction=sum(g)/len(g),runs=len(g)))
with (RESULTS/'v9_scale_topology_curves.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=scales[0]);w.writeheader();w.writerows(scales)
classification={}
for d in sorted({r['domain'] for r in rows}):
    vals=[x for x in curve if x['domain']==d and x['policy']=='all_coupled']
    good=[x['rho'] for x in vals if x['success_fraction']>=.8]
    classification[d]={'all_coupled_good_rho':good,'finite_window_observed':bool(good and min(good)>min(x['rho'] for x in vals) and max(good)<max(x['rho'] for x in vals)),'low_success':vals[0]['success_fraction'],'high_success':vals[-1]['success_fraction']}
summary={'runs':len(rows),'classification':classification,'interpretation':'Frozen author-defined cross-domain computation; each domain is analysed separately and no raw critical constant is pooled.'}
(RESULTS/'v9_analysis.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
