#!/usr/bin/env python3
import csv,hashlib,json,math
from pathlib import Path
H=Path(__file__).resolve().parent;C=json.loads((H/'MARINE_V59_FROZEN_CONTRACT.json').read_text());R=json.loads((H/'results/V59_REPORT.json').read_text());rows=list(csv.DictReader((H/'results/V59_HELDOUT.csv').open()))
checks={'contract_hash':R['contract_sha256']==hashlib.sha256((H/'MARINE_V59_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),'complete_factorial':len(rows)==len(C['heldout_seeds'])*len(C['rho_grid'])*len(C['policies']),'fresh_seeds':set(C['heldout_seeds']).isdisjoint(set(range(901,911))),'finite':all(math.isfinite(float(x[k])) for x in rows for k in ['terminal_rmse_m','minimum_clearance_m','energy_N2s']),'promotion':R['promotion_pass'],'failed_precursor_retained':(H.parent/'marine_v58_unseen_domain/results/V58_REPORT.json').exists(),'honest_label':'not physical' in C['claim_boundary'].lower()}
print(json.dumps(checks,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
