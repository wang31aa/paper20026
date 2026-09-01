#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
HERE=Path(__file__).resolve().parent; C=json.load(open(HERE/'V20_FROZEN_CONTRACT.json')); rows=list(csv.DictReader(open(HERE/'results/v20_runs.csv')))
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*(len(C['development_seeds'])+len(C['heldout_seeds']))*len(C['policies'])
assert len(rows)==expected
keys={(r['domain'],r['n'],r['topology'],r['rho'],r['seed'],r['policy']) for r in rows};assert len(keys)==expected
for r in rows:
 assert float(r['minimum_parameter_spread'])>0
 for k in ('minimum_physical_margin','tail_error','control_energy','communication_messages'): assert math.isfinite(float(r[k]))
 assert r['evidence_class']=='prospective_heterogeneous_computational_stress_test'
print(f'PASS: {expected} complete finite V20 trajectories with positive heterogeneous spread')
