#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'results/v11_1_confirmatory_runs.csv');base=d[d.policy.eq('all_coupled')]
rows=[]
for domain,g in base.groupby('domain'):
 y=g.task_success.to_numpy(int);p=g.predicted_success.to_numpy(int);tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());sens=tp/max(tp+fn,1);spec=tn/max(tn+fp,1)
 rows.append({'domain':domain,'n':len(g),'sensitivity':sens,'specificity':spec,'false_negative_rate':1-sens,'balanced_accuracy':(sens+spec)/2,'brier_score':float(np.mean((p-y)**2))})
eff=[]
keys=['domain','n','topology','rho','seed'];a=d[d.policy.eq('all_coupled')].set_index(keys);b=d[d.policy.eq('two_layer_gate')].set_index(keys)
for domain in d.domain.unique():
 x=a.loc[domain];z=b.loc[domain];eff.append({'domain':domain,'prevented_failures':int(((x.task_success==0)&(z.task_success==1)).sum()),'induced_failures':int(((x.task_success==1)&(z.task_success==0)).sum()),'median_margin_change':float(np.median(z.minimum_physical_margin-x.minimum_physical_margin)),'median_energy_change':float(np.median(z.control_energy-x.control_energy)),'median_message_change':float(np.median(z.communication_messages-x.communication_messages))})
passed=sum(r['sensitivity']>=.8 and r['specificity']>=.8 and r['false_negative_rate']<=.2 for r in rows)
out={'status':'PASS' if passed>=2 else 'FROZEN_CONFIRMATORY_FAILURE','passing_domains':passed,'prediction':rows,'paired_intervention':eff,'interpretation':'Same critical-function structure is promoted only if at least two domains pass; otherwise V11.1 is retained as adverse evidence.'}
(R/'results/v11_1_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
