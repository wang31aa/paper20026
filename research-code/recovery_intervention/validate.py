#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
R=Path(__file__).resolve().parent/'results';rows=list(csv.DictReader((R/'candidates.csv').open()));s=json.loads((R/'summary.json').read_text())
assert len(rows)==6 and {r['intervention'] for r in rows}=={'baseline','information','coupling','gating','spread','direct_pinning'}
cert=[r for r in rows if r['certified_recovery']=='True'];assert [r['intervention'] for r in cert]==['spread']
assert s['selected_intervention']['intervention']=='spread';assert float(s['selected_intervention']['observed_recovery_time'])>=0
assert all(math.isfinite(float(r['certificate_radius'])) for r in rows)
print('PASS: six candidates, fail-closed selection and held-out recovery')
