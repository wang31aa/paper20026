#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg'); mpl.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; A=json.loads((R/'results/v12_analysis.json').read_text())
domains=['motor','uav6dof','vehicle']; labels=['Motor','UAV','Vehicle']
P={x['domain']:x for x in A['prediction']}; E={x['domain']:x for x in A['paired_intervention']}
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True); x=range(3); w=.34
ax[0].bar([i-w/2 for i in x],[P[d]['sensitivity'] for d in domains],w,label='Sensitivity',color='#337ab7')
ax[0].bar([i+w/2 for i in x],[P[d]['specificity'] for d in domains],w,label='Specificity',color='#e6a000')
ax[0].axhline(.8,color='.35',ls='--',lw=1); ax[0].set(xticks=list(x),xticklabels=labels,ylim=(0,1),ylabel='Frozen held-out classification',title='a  Parameterized critical function'); ax[0].legend(frameon=False)
ax[1].bar(x,[E[d]['prevented_failures'] for d in domains],label='Failures prevented',color='#00a17a')
ax[1].bar(x,[E[d]['induced_failures'] for d in domains],label='Failures induced',color='#d95f02')
ax[1].set(xticks=list(x),xticklabels=labels,ylabel='Paired conditions',title='b  Two-layer intervention'); ax[1].legend(frameon=False)
for ext in ('pdf','svg','png'): fig.savefig(R/f'results/Fig_V12_confirmatory.{ext}',dpi=300)
plt.close(fig)
