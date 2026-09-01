#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
HERE=Path(__file__).resolve().parent; C=json.load(open(HERE/'V21_MOTOR_FROZEN_CONTRACT.json'))
rows=list(csv.DictReader(open(HERE/'results/v21_runs.csv')))
expected=len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*(len(C['development_seeds'])+len(C['heldout_seeds']))*len(C['policies'])
assert len(rows)==expected
keys={(r['paired_replay_id'],r['policy']) for r in rows}; assert len(keys)==expected
for r in rows:
    assert float(r['minimum_parameter_spread'])>0
    for k in ('minimum_physical_margin','tail_error','control_energy'): assert math.isfinite(float(r[k]))
print(f'PASS: {expected} complete V21 motor trajectories; evidence remains computational')
