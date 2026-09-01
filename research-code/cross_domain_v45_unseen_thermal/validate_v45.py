#!/usr/bin/env python3
import csv,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/'V45_FROZEN_CONTRACT.json').read_text());R=json.loads((H/'results/V45_REPORT.json').read_text())
rows=list(csv.DictReader((H/'results/v45_runs.csv').open()));pred=list(csv.DictReader((H/'results/v45_predictions.csv').open()))
checks={'contract_hash':R['contract_sha256']==hashlib.sha256((H/'V45_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
 'expected_rows':len(rows)==len(C['rho_grid'])*(len(C['random_seeds'])+1),'all_rhos':len(pred)==len(C['rho_grid']),
 'adversarial_agreement':R['adversarial_agreement'] is True,'no_false_safe':R['false_safe']==0,
 'new_domain_absent_from_prior':C['domain_absent_from_v41_v44'] is True,
 'not_physical_claim':'not source-identified' in C['claim_boundary']}
out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
 'promotion':{'ninth_theorem_class_domain':True,'leave_physical_domain':False,'hil':False}}
print(json.dumps(out,indent=2));sys.exit(0 if out['status']=='PASS' else 1)
