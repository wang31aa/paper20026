#!/usr/bin/env python3
import csv,json
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent;rows=list(csv.DictReader((R/'results/v91_runs.csv').open()));g=defaultdict(list)
for x in rows:g[(x['domain'],x['policy'],int(x['n']),x['topology'],float(x['rho']))].append(int(x['success']))
out=[]
for (d,p,n,t,rho),v in sorted(g.items()):out.append(dict(domain=d,policy=p,n=n,topology=t,rho=rho,runs=len(v),success_fraction=sum(v)/len(v)))
with (R/'results/v91_scale_topology.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
by_size=defaultdict(list)
for x in rows:by_size[(x['domain'],x['policy'],int(x['n']))].append(int(x['success']))
summary=[dict(domain=d,policy=p,n=n,success_fraction=sum(v)/len(v),evaluations=len(v)) for (d,p,n),v in sorted(by_size.items())]
with (R/'results/v91_size_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
report={'runs':len(rows),'finding':'Feasibility is strongly scale- and topology-dependent. Aggregate cross-size success does not establish a universal window; N=5 robot strata show gate recovery at high participation, whereas larger information diameters frequently miss the fixed deadline. Vehicle success remains sparse and baseline-dependent.'}
(R/'results/v91_analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
