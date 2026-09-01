#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent;root=R.parent
old=json.loads((root/'cross_domain_v11/results/v12_1_analysis.json').read_text());new=json.loads((R/'results/v13_analysis.json').read_text())
pred=old['prediction']+new['prediction'];eff=old['paired_intervention']+new['paired_intervention'];order=['uav6dof','vehicle','motor','robot','microgrid','circuit','water','structure'];labels=['UAV','Vehicle','Motor','Robot','Microgrid','Circuit','Water','Structure'];P={x['domain']:x for x in pred};E={x['domain']:x for x in eff}
with (R/'results/eight_domain_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['domain','balanced_accuracy','sensitivity','specificity','prevented_failures','induced_failures','evidence_generation'])
 for d in order:w.writerow([d,P[d]['balanced_accuracy'],P[d]['sensitivity'],P[d]['specificity'],E[d]['prevented_failures'],E[d]['induced_failures'],'V12.1' if d in ('uav6dof','vehicle','motor') else 'V13'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.7})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),constrained_layout=True);x=np.arange(8);ba=np.array([P[d]['balanced_accuracy'] for d in order])
ax[0].bar(x,ba,color=['#0072B2']*3+['#999999']*5,width=.72);ax[0].axhline(.8,color='#D55E00',ls='--',lw=1);ax[0].set(ylim=(0,1),ylabel='Held-out balanced accuracy',xticks=x,xticklabels=labels);ax[0].tick_params(axis='x',rotation=35);ax[0].text(-.12,1.04,'a',transform=ax[0].transAxes,fontweight='bold',fontsize=10);ax[0].text(.02,.83,'pre-registered threshold',color='#D55E00',transform=ax[0].transAxes)
prev=np.array([E[d]['prevented_failures'] for d in order]);ind=np.array([E[d]['induced_failures'] for d in order]);ax[1].bar(x,prev,color='#009E73',width=.72,label='Prevented');ax[1].bar(x,-ind,color='#CC79A7',width=.72,label='Induced');ax[1].axhline(0,color='black',lw=.7);ax[1].set(ylabel='Paired failure count',xticks=x,xticklabels=labels);ax[1].tick_params(axis='x',rotation=35);ax[1].legend(frameon=False,ncol=2,loc='upper right');ax[1].text(-.12,1.04,'b',transform=ax[1].transAxes,fontweight='bold',fontsize=10)
for ext in ('pdf','svg','png'):fig.savefig(R/f'results/Fig_V13_eight_domain.{ext}',dpi=300)
print(R/'results/Fig_V13_eight_domain.pdf')
