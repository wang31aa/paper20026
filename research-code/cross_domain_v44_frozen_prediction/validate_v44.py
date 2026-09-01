#!/usr/bin/env python3
import csv, hashlib, json, sys
from pathlib import Path
H=Path(__file__).resolve().parent; C=json.loads((H/'V44_STAGE1_CONTRACT.json').read_text())
P=json.loads((H/'V44_STAGE2_FROZEN_PREDICTIONS.json').read_text());R=json.loads((H/'results/V44_FROZEN_PREDICTION_REPORT.json').read_text())
rows=list(csv.DictReader((H/'results/v44_all_runs.csv').open()))
dev=[r for r in rows if r['split']=='development']; held=[r for r in rows if r['split']=='heldout']
checks={
 'stage1_hash_matches':P['stage1_sha256']==hashlib.sha256((H/'V44_STAGE1_CONTRACT.json').read_bytes()).hexdigest(),
 'stage2_hash_matches':R['stage2_sha256']==hashlib.sha256((H/'V44_STAGE2_FROZEN_PREDICTIONS.json').read_bytes()).hexdigest(),
 'seed_sets_disjoint':not ({r['seed'] for r in dev}&{r['seed'] for r in held}),
 'expected_development_rows':len(dev)==8*2*11*4,
 'expected_heldout_rows':len(held)==8*2*11*8,
 'all_domains_present':len({r['domain'] for r in held})==8,
 'binary_endpoints':all(r['task_success'] in ('0','1') for r in rows),
 'finite_margins':all(r['minimum_physical_margin'] not in ('','nan','inf','-inf') for r in rows),
 'claim_boundary_not_hil':'not leave-domain-out' in C['claim_boundary'] and 'not' in C['claim_boundary'].lower()
}
out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
     'promotion':{'within_domain_prospective_prediction':all(checks.values()),'leave_domain_out_universality':False,'physical_or_hil':False}}
print(json.dumps(out,indent=2));sys.exit(0 if out['status']=='PASS' else 1)
