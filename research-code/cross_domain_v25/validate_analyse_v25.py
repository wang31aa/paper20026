#!/usr/bin/env python3
import csv,json
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parent;C=json.loads((P/'V25_FAIR_BASELINE_CONTRACT.json').read_text());rows=list(csv.DictReader((P/'results/v25_runs.csv').open()))
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*(len(C['development_seeds'])+len(C['heldout_seeds']))*len(C['policies']);assert len(rows)==expected
groups=defaultdict(list)
for r in rows:groups[r['paired_replay_id']].append(r)
assert len(groups)==expected//len(C['policies']) and all({x['policy'] for x in g}==set(C['policies']) for g in groups.values())
for g in groups.values():
    d=g[0]['domain'];field='parameter_ratio_spread' if d=='motor' else 'parameter_spread_model_units';assert len({x[field] for x in g})==1
assert all(float(r['minimum_parameter_ratio_spread' if r['domain']=='motor' else 'minimum_parameter_spread_model_units'])>0 for r in rows)
switch=[r for r in rows if r['topology']=='certified_switching'];assert min(float(r['common_metric_mu']) for r in switch)>0 and min(int(r['distinct_adjacencies']) for r in switch)>=2
held=[r for r in rows if r['split']=='heldout'];rates=defaultdict(list)
for r in held:rates[(r['domain'],r['policy'])].append(int(r['task_success']))
summary={d:{p:sum(v)/len(v) for (dd,p),v in sorted(rates.items()) if dd==d} for d in C['domains']}
out={'status':'PASS','rows':len(rows),'heldout_rows':len(held),'minimum_common_metric_mu':min(float(r['common_metric_mu']) for r in switch),'policy_success_rates':summary,'universal_two_layer_dominance':all(summary[d]['two_layer_gate']>max(v for p,v in summary[d].items() if p!='two_layer_gate') for d in C['domains']),'evidence_class':C['evidence_class']}
(P/'results/V25_VALIDATION_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
