#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;A=json.loads((R/'results/v11_1_analysis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8),constrained_layout=True)
domains=[x['domain'] for x in A['prediction']];x=range(3)
ax[0].bar([i-.18 for i in x],[q['sensitivity'] for q in A['prediction']],.36,label='Sensitivity',color='#2878B5')
ax[0].bar([i+.18 for i in x],[q['specificity'] for q in A['prediction']],.36,label='Specificity',color='#E69F00');ax[0].axhline(.8,ls='--',lw=.8,color='.35');ax[0].set_xticks(list(x),domains);ax[0].set_ylim(0,1);ax[0].set_ylabel('Held-out classification');ax[0].set_title('a  Frozen critical-function test',loc='left',weight='bold');ax[0].legend(frameon=False,fontsize=7)
e={q['domain']:q for q in A['paired_intervention']};prevent=[e[d]['prevented_failures'] for d in domains];induce=[e[d]['induced_failures'] for d in domains]
ax[1].bar([i-.18 for i in x],prevent,.36,label='Failures prevented',color='#009E73');ax[1].bar([i+.18 for i in x],induce,.36,label='Failures induced',color='#D55E00');ax[1].set_xticks(list(x),domains);ax[1].set_ylabel('Paired conditions');ax[1].set_title('b  Two-layer intervention',loc='left',weight='bold');ax[1].legend(frameon=False,fontsize=7)
for ext in ('pdf','svg','png'):fig.savefig(R/'results'/f'Fig_V11_confirmatory.{ext}',dpi=300,bbox_inches='tight')
