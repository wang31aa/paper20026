#!/usr/bin/env python3
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;C=json.loads((P/'VHIL_V3_FROZEN_CONTRACT.json').read_text());rows=list(csv.DictReader((P/'results/vhil_v3_cycles.csv').open()));q=json.loads((P/'results/VHIL_V3_QUALIFICATION.json').read_text())
assert len(rows)==len(C['domains'])*len(C['policies'])*C['cycles'];assert q['status']=='PASS' and not q['hardware_hil'] and not q['entity_platform'];assert q['minimum_parameter_spread']>0 and q['minimum_adjacency_modes']>=3;assert all(int(r['target_visible_nodes'])==1 for r in rows);assert all(r['observer_json']!=r['state_json'] for r in rows);print(json.dumps(q,indent=2))
