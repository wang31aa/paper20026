#!/usr/bin/env python3
"""Test whether the frozen low-dimensional feature fibres have unique labels."""
import csv,json
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parent
rows=list(csv.DictReader(open(R/'results/v13_confirmation_runs.csv')))
rows=[x for x in rows if x['policy']=='all_coupled']
# Gamma, visible fraction and heterogeneous fraction are included at full CSV
# precision.  Domain is deliberately omitted because the candidate law claims
# transfer across domains.
keys=['n','topology','rho','seed','gamma','visible_fraction','heterogeneous_fraction','fault']
groups=defaultdict(list)
for x in rows:groups[tuple(x[k] for k in keys)].append(x)
mixed=[]
for k,g in groups.items():
 y={int(x['task_success']) for x in g}
 if len(y)>1:
  mixed.append({'feature_key':'|'.join(k),'successful_domains':';'.join(sorted(x['domain'] for x in g if int(x['task_success'])==1)),'failed_domains':';'.join(sorted(x['domain'] for x in g if int(x['task_success'])==0))})
with open(R/'results/v13_feature_fibre_conflicts.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=mixed[0].keys());w.writeheader();w.writerows(mixed)
out={'groups':len(groups),'complete_five_domain_groups':sum(len(g)==5 for g in groups.values()),'mixed_outcome_groups':len(mixed),'mixed_fraction':len(mixed)/len(groups),'result':'LOW_DIMENSIONAL_FEATURE_MAP_NOT_IDENTIFYING' if mixed else 'NO_CONFLICT_OBSERVED','logical_scope':'One mixed fibre falsifies deterministic universal classification by these features; it does not falsify richer system-specific functions.'}
(R/'results/v13_feature_identifiability.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
