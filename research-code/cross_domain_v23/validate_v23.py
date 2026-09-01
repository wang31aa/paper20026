#!/usr/bin/env python3
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'results/v23_runs.csv').open()));C=json.loads((P/'V23_CORRECTED_OBSERVER_SWITCHING_CONTRACT.json').read_text())
expected=len(C['domains'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*(len(C['development_seeds'])+len(C['heldout_seeds']))*len(C['policies'])
assert len(rows)==expected
assert len({(r['paired_replay_id'],r['policy']) for r in rows})==expected
assert max(float(r['observer_identity_error']) for r in rows)<1e-10
assert max(float(r['observer_transition_radius']) for r in rows)<1
switch=[r for r in rows if r['topology']=='switching']
assert switch and min(int(r['distinct_adjacencies']) for r in switch)>=2
fixed=[r for r in rows if r['topology']!='switching']
assert max(int(r['actual_switch_count']) for r in fixed)==0
assert all(r['evidence_class']==C['evidence_class'] for r in rows)
out={'status':'PASS','rows':len(rows),'heldout_rows':sum(r['split']=='heldout' for r in rows),'max_observer_identity_error':max(float(r['observer_identity_error']) for r in rows),'max_transition_radius':max(float(r['observer_transition_radius']) for r in rows),'minimum_switching_adjacencies':min(int(r['distinct_adjacencies']) for r in switch),'evidence_class':C['evidence_class']}
(P/'results/V23_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
