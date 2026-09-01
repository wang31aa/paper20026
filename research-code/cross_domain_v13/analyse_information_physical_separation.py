#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'results/v14_baseline_runs.csv');keys=['domain','n','topology','rho','seed']
tab={p:g.set_index(keys) for p,g in d.groupby('policy')};rows=[]
for domain in sorted(d.domain.unique()):
 idx=tab['all_coupled'].loc[domain].index
 a=tab['all_coupled'].loc[domain].loc[idx];i=tab['independent_tracking'].loc[domain].loc[idx];g=tab['two_layer_gate'].loc[domain].loc[idx]
 rows.append({'domain':domain,'conditions':len(idx),'all_success_rate':float(a.task_success.mean()),'independent_success_rate':float(i.task_success.mean()),'gate_success_rate':float(g.task_success.mean()),'independent_beats_gate_success':int(((i.task_success>g.task_success)).sum()),'gate_beats_independent_success':int(((g.task_success>i.task_success)).sum()),'independent_higher_margin':int((i.minimum_physical_margin>g.minimum_physical_margin).sum()),'gate_higher_margin':int((g.minimum_physical_margin>i.minimum_physical_margin).sum()),'median_independent_minus_gate_margin':float(np.median(i.minimum_physical_margin-g.minimum_physical_margin))})
out={'comparison':'same observer and innovations; independent_tracking removes physical graph only','domains':rows,'interpretation':'Tests the information/physical separation mechanism. It does not estimate kappa_e or d_e and is not a proof that the edge-test premises hold in each nonlinear model.'};(R/'results/information_physical_separation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
