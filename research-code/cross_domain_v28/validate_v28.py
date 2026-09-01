#!/usr/bin/env python3
import csv,json,math
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent;C=json.loads((R/'V28_DYNAMIC_SWITCHING_CONTRACT.json').read_text());rows=list(csv.DictReader((R/'results/v28_runs.csv').open()))
expected=len(C['domains'])*len(C['sizes'])*len(C['rho_grid'])*len(C['seeds'])*len(C['policies']);assert len(rows)==expected
blocks=defaultdict(list)
for x in rows:
 assert all(math.isfinite(float(x[k])) for k in ('common_metric_mu','minimum_physical_margin','tail_error','control_energy'))
 assert int(x['distinct_adjacencies'])>=3 and int(x['actual_switch_count'])>=3 and float(x['common_metric_mu'])>0
 blocks[(x['domain'],x['n'],x['rho'],x['seed'])].append(x)
assert all(len(v)==6 and len({x['parameter_sha256'] for x in v})==1 for v in blocks.values())
out={'status':'PASS','trajectory_rows':len(rows),'paired_blocks':len(blocks),'minimum_distinct_adjacencies':min(int(x['distinct_adjacencies']) for x in rows),'minimum_actual_switches':min(int(x['actual_switch_count']) for x in rows),'minimum_common_metric_mu':min(float(x['common_metric_mu']) for x in rows),'claim_boundary':C['claim_boundary']}
(R/'results/V28_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
