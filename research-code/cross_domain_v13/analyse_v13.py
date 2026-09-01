#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'results/v13_confirmation_runs.csv');P={x['domain']:x for x in json.loads((R/'V13_FROZEN_PARAMETERS.json').read_text())['parameters']}
base=d[d.policy.eq('all_coupled')].copy();base['psi']=[P[x.domain]['A_information']/(x.rho*x.gamma)+P[x.domain]['B_mismatch']*x.rho*x.heterogeneous_fraction+P[x.domain]['C_constraint']*x.rho**2*x.heterogeneous_fraction+P[x.domain]['D_scale']*np.log(x.n/5) for _,x in base.iterrows()];base['predicted_success']=(base.psi<=1).astype(int)
pred=[]
for domain,g in base.groupby('domain'):
 y=g.task_success.to_numpy(int);p=g.predicted_success.to_numpy(int);tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());se=tp/max(tp+fn,1);sp=tn/max(tn+fp,1);pred.append({'domain':domain,'n':len(g),'sensitivity':se,'specificity':sp,'false_negative_rate':1-se,'balanced_accuracy':(se+sp)/2,'tp':tp,'fn':fn,'tn':tn,'fp':fp})
keys=['domain','n','topology','rho','seed'];a=d[d.policy.eq('all_coupled')].set_index(keys);b=d[d.policy.eq('two_layer_gate')].set_index(keys);effects=[]
for domain in d.domain.unique():
 x=a.loc[domain];z=b.loc[domain];effects.append({'domain':domain,'prevented_failures':int(((x.task_success==0)&(z.task_success==1)).sum()),'induced_failures':int(((x.task_success==1)&(z.task_success==0)).sum()),'median_margin_change':float(np.median(z.minimum_physical_margin-x.minimum_physical_margin)),'median_energy_change':float(np.median(z.control_energy-x.control_energy)),'median_message_change':float(np.median(z.communication_messages-x.communication_messages))})
passing=sum(x['sensitivity']>=.8 and x['specificity']>=.8 and x['false_negative_rate']<=.2 for x in pred);need=json.loads((R/'V13_FROZEN_CONTRACT.json').read_text())['prediction_acceptance']['minimum_passing_domains']
out={'status':'PASS' if passing>=need else 'FROZEN_CONFIRMATORY_FAILURE','passing_domains':passing,'required_domains':need,'prediction':pred,'paired_intervention':effects,'interpretation':'Independent new-seed computation under frozen domain-specific coefficients; failed domains are retained.'};(R/'results/v13_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
