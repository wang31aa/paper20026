#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'results/v12_1_fault_confirmatory_runs.csv');base=d[d.policy.eq('all_coupled')];pred=[]
for domain,g in base.groupby('domain'):
 y=g.task_success.to_numpy(int);p=g.predicted_success.to_numpy(int);tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());se=tp/max(tp+fn,1);sp=tn/max(tn+fp,1);pred.append({'domain':domain,'n':len(g),'sensitivity':se,'specificity':sp,'false_negative_rate':1-se,'balanced_accuracy':(se+sp)/2,'tp':tp,'fn':fn,'tn':tn,'fp':fp})
keys=['domain','n','topology','rho','seed'];a=d[d.policy.eq('all_coupled')].set_index(keys);b=d[d.policy.eq('two_layer_gate')].set_index(keys);effects=[]
for domain in d.domain.unique():
 x=a.loc[domain];z=b.loc[domain];effects.append({'domain':domain,'prevented_failures':int(((x.task_success==0)&(z.task_success==1)).sum()),'induced_failures':int(((x.task_success==1)&(z.task_success==0)).sum()),'median_margin_change':float(np.median(z.minimum_physical_margin-x.minimum_physical_margin)),'median_energy_change':float(np.median(z.control_energy-x.control_energy))})
passing=sum(x['sensitivity']>=.8 and x['specificity']>=.8 and x['false_negative_rate']<=.2 for x in pred);out={'status':'PASS' if passing>=2 else 'FROZEN_CONFIRMATORY_FAILURE','passing_domains':passing,'prediction':pred,'paired_intervention':effects,'interpretation':'Fault-rotated confirmation of the already frozen V12 function; no parameter revision.'};(R/'results/v12_1_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
