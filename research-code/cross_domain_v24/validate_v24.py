#!/usr/bin/env python3
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'results/v24_runs.csv').open())); C=json.loads((P/'V24_COMMON_METRIC_SWITCHING_CONTRACT.json').read_text())
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*(len(C['development_seeds'])+len(C['heldout_seeds']))*len(C['policies'])
assert len(rows)==expected
assert len({(r['paired_replay_id'],r['policy']) for r in rows})==expected
by_condition={}
for r in rows:
    key=r['paired_replay_id']; domain=r['domain']
    field='parameter_ratio_spread' if domain=='motor' else 'parameter_spread_model_units'
    value=r[field]
    by_condition.setdefault(key,set()).add(value)
assert max(map(len,by_condition.values()))==1
for r in rows:
    field='minimum_parameter_ratio_spread' if r['domain']=='motor' else 'minimum_parameter_spread_model_units'
    assert float(r[field])>0
assert max(float(r['observer_identity_error']) for r in rows)<1e-10
assert max(float(r['observer_transition_radius']) for r in rows)<1
switch=[r for r in rows if r['topology']=='certified_switching']
assert switch and min(int(r['distinct_adjacencies']) for r in switch)>=C['common_metric_contract']['minimum_distinct_adjacencies']
assert min(float(r['common_metric_mu']) for r in switch)>C['common_metric_contract']['strict_minimum_mu']
assert min(float(r['directed_asymmetry']) for r in switch)>1e-12
fixed=[r for r in rows if r['topology']!='certified_switching']
assert max(int(r['actual_switch_count']) for r in fixed)==0
out={'status':'PASS','rows':len(rows),'heldout_rows':sum(r['split']=='heldout' for r in rows),
 'max_observer_identity_error':max(float(r['observer_identity_error']) for r in rows),
 'max_transition_radius':max(float(r['observer_transition_radius']) for r in rows),
 'minimum_common_metric_mu':min(float(r['common_metric_mu']) for r in switch),
 'minimum_directed_asymmetry':min(float(r['directed_asymmetry']) for r in switch),
 'minimum_switching_adjacencies':min(int(r['distinct_adjacencies']) for r in switch),
 'paired_parameter_ledgers_consistent':True,
 'all_parameter_spreads_positive':True,
 'evidence_class':C['evidence_class']}
(P/'results/V24_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
