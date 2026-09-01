#!/usr/bin/env python3
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
from run_benchmark import topologies, graph_constants, certificate, LEVELS, A0, B0, R

root=Path(__file__).parent; out=root/'results'; checks={}
rows=list(csv.DictReader((out/'raw_runs.csv').open()))
checks['registered_run_count']=len(rows)==4*len(LEVELS)*10
checks['all_finite']=all(x['finite']=='True' for x in rows)
checks['all_graphs_positive']=all(graph_constants(W)[2]>0 and graph_constants(W)[3]>0 for W in topologies().values())
checks['all_dissipation_margins_positive']=all(certificate(W,l)['d']>0 for W in topologies().values() for l in LEVELS)
checks['observer_gain_condition']=all(35*graph_constants(W)[2]>2*(A0+B0) for W in topologies().values())
checks['target_absorbing_radius_formula']=abs(R-B0/(A0-B0))<1e-14
checks['target_stayed_in_invariant_ball']=max(float(x['target_radius_max']) for x in rows)<=R+2e-10
checks['finite_time_comparison_bound_covered']=max(float(x['max_certificate_ratio']) for x in rows)<=1+2e-8
checks['node_bounds_covered']=max(float(x['max_node_ratio']) for x in rows)<=1+2e-8
raw=np.load(out/'representative_timeseries.npz')
checks['all_120_unrounded_timeseries_present']=len(raw.files)==4*len(LEVELS)*10
checks['timeseries_columns_complete']=all(raw[k].ndim==2 and raw[k].shape[1]==6 for k in raw.files)
dt=list(csv.DictReader((out/'dt_audit.csv').open())); checks['dt_all_finite']=all(x['finite']=='True' for x in dt)
groups={}
for x in dt: groups.setdefault((x['topology'],x['seed']),{})[float(x['dt'])]=float(x['final_error'])
rels=[]
for g in groups.values(): rels.append(abs(g[.002]-g[.001])/max(abs(g[.001]),1e-12))
checks['dt_median_relative_difference_below_1pct']=float(np.median(rels))<.01
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='validation.json'}
report={'passed':all(checks.values()),'checks':checks,'diagnostics':{'max_ratio':max(float(x['max_certificate_ratio']) for x in rows),'median_dt_relative_difference':float(np.median(rels))},'sha256':hashes}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); sys.exit(0 if report['passed'] else 1)
