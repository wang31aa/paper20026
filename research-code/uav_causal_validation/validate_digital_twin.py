#!/usr/bin/env python3
import csv,json,math,sys
from pathlib import Path
h=Path(__file__).resolve().parent
s=json.loads((h/'results/frozen_uav_digital_twin_summary.json').read_text())
rows=list(csv.DictReader((h/'results/frozen_uav_digital_twin_runs.csv').open()))
errors=[]; expected=4*8*7*2*4
if len(rows)!=expected or s['rows']!=expected: errors.append('row count mismatch')
keys=[(r['split'],r['environment'],r['seed'],r['rho'],r['plant'],r['policy']) for r in rows]
if len(set(keys))!=len(keys): errors.append('duplicate factorial key')
for r in rows:
 for k in ('completion_fraction','minimum_pair_clearance_m','minimum_obstacle_clearance_m','control_energy','communication_edges'):
  if not math.isfinite(float(r[k])): errors.append(f'nonfinite {k}')
g=[r for r in rows if r['policy']=='participation_gated_mpc_cbf']
if not any(int(r['gate_changed_edges_before_control']) for r in g): errors.append('gate never changed graph')
if not any(int(r['gate_changed_control_and_state']) for r in g): errors.append('gate never changed control/state')
supported=bool(s['frozen_predicted_feasible_rho']) and s['heldout_sensitivity'] is not None
out={'validator':'FAIL' if errors else 'PASS','execution_qualified':not errors,'scientific_prediction_supported':supported,'result_class':'complete_adverse_paired_computational_test','errors':errors}
print(json.dumps(out,indent=2)); sys.exit(bool(errors))
