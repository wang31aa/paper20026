#!/usr/bin/env python3
from pathlib import Path
import csv,json
from collections import defaultdict

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
rows=list(csv.DictReader((OUT/'v17_runs.csv').open()))
groups=defaultdict(dict)
for r in rows: groups[(r['domain'],r['n'],r['topology'],r['rho'],r['seed'])][r['policy']]=r
domains=sorted({r['domain'] for r in rows}); summary={}
for d in domains:
    rr=[r for r in rows if r['domain']==d]
    pairs=[v for k,v in groups.items() if k[0]==d and len(v)==5]
    changed=sum(abs(float(x['all_coupled']['minimum_physical_margin'])-float(x['two_layer_gate']['minimum_physical_margin']))>1e-12 for x in pairs)
    induced=sum(int(x['all_coupled']['task_success'])==1 and int(x['two_layer_gate']['task_success'])==0 for x in pairs)
    prevented=sum(int(x['all_coupled']['task_success'])==0 and int(x['two_layer_gate']['task_success'])==1 for x in pairs)
    summary[d]={'rows':len(rr),'complete_policy_pairs':len(pairs),
                'minimum_parameter_spread':min(float(x['minimum_parameter_spread']) for x in rr),
                'two_layer_changed_future_margin':changed,'prevented_failures':prevented,
                'induced_failures':induced,
                'success_rate_by_policy':{p:sum(int(x['task_success']) for x in rr if x['policy']==p)/sum(x['policy']==p for x in rr) for p in sorted({x['policy'] for x in rr})}}
checks={'all_domains_present':len(summary)==8,
        'all_parameter_spreads_positive':all(x['minimum_parameter_spread']>0 for x in summary.values()),
        'all_pairs_complete':all(x['complete_policy_pairs']==135 for x in summary.values()),
        'all_two_layer_interventions_change_future_margin':all(x['two_layer_changed_future_margin']>0 for x in summary.values())}
payload={'schema':'V17-ANALYSIS-1','checks':checks,'domains':summary,
         'interpretation':'Computational cross-domain mechanism test; not source reproduction, HIL or physical evidence.'}
(OUT/'v17_analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
assert all(checks.values())
print(json.dumps(payload,indent=2))
