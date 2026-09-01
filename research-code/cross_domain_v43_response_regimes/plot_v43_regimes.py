#!/usr/bin/env python3
"""Plot the complete V41 held-out all-coupled response classification."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
curve=list(csv.DictReader((ROOT/'results/v43_curve_summary.csv').open()))
summary={r['domain']:r for r in csv.DictReader((ROOT/'results/v43_regime_summary.csv').open())}
domains=['uav6dof','vehicle','motor','robot','microgrid','circuit','water','structure']
labels={'uav6dof':'UAV','vehicle':'Vehicle','motor':'Motor','robot':'Robot',
        'microgrid':'Microgrid','circuit':'Circuit','water':'Water','structure':'Structure'}
colors={'monotone_improving':'#0072B2','finite_window':'#D55E00',
        'monotone_worsening':'#CC79A7','plateau':'#009E73',
        'irregular_or_unresolved':'#666666'}
fig,axs=plt.subplots(2,4,figsize=(7.2,3.7),sharex=True,sharey=True)
for ax,d in zip(axs.flat,domains):
    rr=sorted([r for r in curve if r['domain']==d],key=lambda x:float(x['rho']))
    x=np.array([float(r['rho']) for r in rr]); y=np.array([float(r['success_rate']) for r in rr])
    lo=np.array([float(r['wilson_low']) for r in rr]); hi=np.array([float(r['wilson_high']) for r in rr])
    cls=summary[d]['response_class']; c=colors[cls]
    ax.fill_between(x,lo,hi,color=c,alpha=.15,lw=0)
    ax.plot(x,y,'o-',color=c,ms=3,lw=1.4)
    ax.set_title(f"{labels[d]}\n{cls.replace('_',' ')}",fontsize=8)
    ax.grid(alpha=.16,lw=.5); ax.spines[['top','right']].set_visible(False)
for ax in axs[-1]: ax.set_xlabel('Participation, $\\rho$',fontsize=8)
for ax in axs[:,0]: ax.set_ylabel('Held-out success',fontsize=8)
for ax in axs.flat: ax.tick_params(labelsize=7)
fig.suptitle('One frozen experiment contains several participation regimes',fontsize=10,fontweight='bold')
fig.tight_layout(rect=(0,0,1,.94),w_pad=.8,h_pad=.8)
fig.savefig(ROOT/'results/Fig_V43_response_regimes.pdf')
fig.savefig(ROOT/'results/Fig_V43_response_regimes.png',dpi=300)
