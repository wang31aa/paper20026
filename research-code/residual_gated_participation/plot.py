#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
with (ROOT/'results/trajectories.csv').open() as f: tr=list(csv.DictReader(f))
with (ROOT/'results/summary.csv').open() as f: sm=list(csv.DictReader(f))
blue, red, grey='#2F6F9F','#C75446','#555555'
fig, ax=plt.subplots(1,3,figsize=(7.2,2.25),constrained_layout=True)
for policy,c,label in [('all_coupled',red,'All nodes coupled'),('gated',blue,'Residual-gated')]:
    q=[r for r in tr if r['policy']==policy and r['seed']=='0']
    ax[0].plot([float(r['time']) for r in q],[float(r['good_node_max_error']) for r in q],color=c,lw=1.4,label=label)
ax[0].axhline(.65,color=grey,ls='--',lw=.9); ax[0].set(xlabel='Time',ylabel='Unaffected-node max. error',title='a  Error propagation')
ax[0].legend(frameon=False,fontsize=6.5)
q=[r for r in tr if r['policy']=='gated' and r['seed']=='0']
ax[1].step([float(r['time']) for r in q],[int(r['trusted_count']) for r in q],where='post',color=blue,lw=1.4)
ax[1].set(xlabel='Time',ylabel='Participating nodes',ylim=(7.5,12.5),title='b  Dynamic participation')
for j,(policy,c,label) in enumerate([('all_coupled',red,'All'),('gated',blue,'Gated')]):
    vals=[float(r['survival_time']) for r in sm if r['policy']==policy]
    ax[2].scatter(np.full(len(vals),j)+np.linspace(-.08,.08,len(vals)),vals,s=13,color=c,alpha=.8)
    ax[2].plot([j-.18,j+.18],[np.median(vals)]*2,color='black',lw=1.3)
ax[2].set(xticks=[0,1],xticklabels=['All','Gated'],ylabel='Task-survival time',ylim=(0,31),title='c  Twelve deterministic seeds')
for a in ax:
    a.spines[['top','right']].set_visible(False); a.tick_params(labelsize=7); a.title.set_fontsize(8); a.xaxis.label.set_size(7.5); a.yaxis.label.set_size(7.5)
out=ROOT.parent/'figures'
fig.savefig(out/'Fig17_residual_gated_participation.pdf')
fig.savefig(out/'Fig17_residual_gated_participation.svg')
