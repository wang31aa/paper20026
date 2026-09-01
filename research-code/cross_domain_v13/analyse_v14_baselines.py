#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'results/v14_baseline_runs.csv');keys=['domain','n','topology','rho','seed'];base=d[d.policy.eq('all_coupled')].set_index(keys);out=[]
for (domain,policy),g in d[~d.policy.eq('all_coupled')].groupby(['domain','policy']):
 x=base.loc[domain];z=g.set_index(keys[1:]).loc[x.index]
 out.append({'domain':domain,'policy':policy,'success_rate':float(z.task_success.mean()),'prevented_failures':int(((x.task_success==0)&(z.task_success==1)).sum()),'induced_failures':int(((x.task_success==1)&(z.task_success==0)).sum()),'median_margin_change':float(np.median(z.minimum_physical_margin-x.minimum_physical_margin)),'median_energy_change':float(np.median(z.control_energy-x.control_energy)),'median_message_change':float(np.median(z.communication_messages-x.communication_messages))})
(R/'results/v14_baseline_analysis.json').write_text(json.dumps({'rows':len(d),'condition_pairs':len(base),'comparisons':out,'scope':'computational comparators; no MPC/CBF/SOTA claim'},indent=2)+'\n');print(json.dumps(out,indent=2))
