#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
HERE=Path(__file__).resolve().parent; C=json.loads((HERE/'V19_FROZEN_CONTRACT.json').read_text())
rows=list(csv.DictReader((HERE/'results/v19_heldout_runs.csv').open())); report={'schema':'V19-RESULT-1','rows':len(rows),'domains':{}}
for d in C['domains']:
    rr=[r for r in rows if r['domain']==d]; tp=tn=fp=fn=0
    for r in rr:
        y=int(r['task_success']);p=int(r['predicted_success']);tp+=y and p;tn+=(not y) and (not p);fp+=(not y) and p;fn+=y and (not p)
    sens=tp/(tp+fn) if tp+fn else None; spec=tn/(tn+fp) if tn+fp else None
    passed=sens is not None and spec is not None and sens>=.8 and spec>=.8
    report['domains'][d]={'n':len(rr),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'sensitivity':sens,'specificity':spec,'promotion_pass':passed}
report['qualified_domain_count']=sum(x['promotion_pass'] for x in report['domains'].values());report['all_eight_qualified']=report['qualified_domain_count']==8
report['interpretation']=C['claim_boundary'];(HERE/'results/V19_QUALIFICATION_REGISTRY.json').write_text(json.dumps(report,indent=2)+'\n')
assert len(rows)==len(C['domains'])*len(C['heldout_sizes'])*len(C['heldout_topologies'])*len(C['heldout_rho'])*len(C['heldout_seeds'])*len(C['policies'])
assert len({(r['paired_replay_id'],r['policy']) for r in rows})==len(rows)
assert all(float(r['minimum_parameter_spread'])>0 and math.isfinite(float(r['minimum_physical_margin'])) for r in rows)
print(json.dumps(report,indent=2))
