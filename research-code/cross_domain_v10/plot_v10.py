#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

R=Path(__file__).resolve().parent
d=pd.read_csv(R/'results/v10_runs.csv')
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
colors={'all_coupled':'#C44E52','two_layer_gate':'#2878B5','independent_tracking':'#55A868'}
labels={'all_coupled':'Permanent coupling','two_layer_gate':'Two-layer gate','independent_tracking':'Independent tracking'}
for ax,(domain,title) in zip(axs[0],[('robot','Robot formation'),('motor','Motor-speed network')]):
    z=d[d.domain==domain].groupby(['policy','eta']).success.mean().reset_index()
    for p,g in z.groupby('policy'):
        ax.plot(g.eta,g.success,marker='o',lw=1.5,ms=4,color=colors[p],label=labels[p])
    ax.axhline(.8,color='.55',ls='--',lw=.8);ax.set_xscale('log');ax.set_ylim(-.04,1.04)
    ax.set(title=title,xlabel=r'Topology-normalized participation, $\eta=\rho\gamma$',ylabel='Task success fraction')
axs[0,1].legend(frameon=False,fontsize=7,loc='lower left')
for ax,(domain,title) in zip(axs[1],[('robot','Scale dependence: robot'),('motor','Scale dependence: motor')]):
    z=d[(d.domain==domain)&(d.policy=='two_layer_gate')].groupby(['n','eta']).success.mean().reset_index()
    for n,g in z.groupby('n'):ax.plot(g.eta,g.success,marker='o',lw=1.2,ms=3,label=f'N={n}')
    ax.set_xscale('log');ax.set_ylim(-.04,1.04);ax.set(title=title,xlabel=r'$\eta$',ylabel='Two-layer-gate success fraction')
    ax.legend(frameon=False,fontsize=7,ncol=2)
fig.suptitle('Topology normalization reveals conditional, not universal, participation windows',fontsize=10)
fig.savefig(R/'results/Fig_V10_topology_normalized.pdf',bbox_inches='tight')
fig.savefig(R/'results/Fig_V10_topology_normalized.svg',bbox_inches='tight')
print('figure PASS')
