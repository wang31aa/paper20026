#!/usr/bin/env python3
"""Nature-style discovery figure: theorem, frozen prediction and intervention."""
from __future__ import annotations
import csv, json, plistlib, subprocess
from collections import defaultdict
from pathlib import Path
# Avoid a slow macOS-wide font inventory in restricted reproducibility runners.
_check_output = subprocess.check_output
def _bounded_check_output(cmd, *args, **kwargs):
    if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == 'system_profiler':
        return plistlib.dumps([{'_items': []}])
    return _check_output(cmd, *args, **kwargs)
subprocess.check_output = _bounded_check_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
report=json.loads((ROOT/'cross_domain_v44_frozen_prediction/results/V44_FROZEN_PREDICTION_REPORT.json').read_text())
rows=list(csv.DictReader((ROOT/'cross_domain_v44_frozen_prediction/results/v44_all_runs.csv').open()))
v41=list(csv.DictReader((ROOT/'cross_domain_v41/results/v41_heldout_summary.csv').open()))
rhos=[.3,.5,.7,.9,1.1,1.3,1.6,1.9,2.2,2.5,2.8]
domains=['uav6dof','vehicle','motor','robot','microgrid','circuit','water','structure']
labels=['UAV','Vehicle','Motor','Robot','Microgrid','Circuit','Water','Structure']
colors={'uav6dof':'#0072B2','vehicle':'#D55E00','motor':'#009E73','robot':'#56B4E9',
        'microgrid':'#E69F00','circuit':'#CC79A7','water':'#777777','structure':'#6A3D9A'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':9,
 'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(11.4,6.15),constrained_layout=True)
gs=fig.add_gridspec(2,2,width_ratios=(.92,1.35),height_ratios=(1,1))

# a: theorem-first mechanism, deliberately sparse text
ax=fig.add_subplot(gs[0,0]); ax.set_axis_off(); ax.set_xlim(0,1);ax.set_ylim(0,1)
x=np.linspace(.08,.92,240); info=.22+.34/(x+.12); exposure=.20+.68*x*x; risk=np.maximum(info,exposure)
ax.plot(x,info,color='#0072B2',lw=1.8,label='Information loss')
ax.plot(x,exposure,color='#D55E00',lw=1.8,label='Mismatch exposure')
ax.plot(x,risk,color='#222222',lw=2.4,label='Active task risk')
ax.axhline(.78,color='#555555',ls='--',lw=1)
safe=risk<=.78; ax.fill_between(x,.05,.78,where=safe,color='#009E73',alpha=.16)
ax.text(.50,.10,'task-feasible participation set',ha='center',color='#007A5E',weight='bold')
ax.text(.13,.90,'weak information',color='#0072B2');ax.text(.68,.90,'excess exposure',color='#D55E00')
ax.text(.94,.79,'threshold',va='bottom',ha='right',color='#555555')
ax.legend(frameon=False,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.02))
ax.set_title('a  Conditional participation window',loc='left',weight='bold')

# b: all eight frozen development versus heldout curves
ax=fig.add_subplot(gs[0,1]); g=defaultdict(list)
for r in rows:g[(r['split'],r['domain'],float(r['rho']))].append(int(r['task_success']))
for d,lbl in zip(domains,labels):
    dev=[np.mean(g[('development',d,r)]) for r in rhos]
    held=[np.mean(g[('heldout',d,r)]) for r in rhos]
    ax.plot(rhos,dev,color=colors[d],ls='--',lw=1,alpha=.65)
    ax.plot(rhos,held,color=colors[d],marker='o',ms=2.7,lw=1.45,label=lbl)
ax.set(xlabel='Participation strength, ρ',ylabel='Task success',ylim=(-.04,1.04),xlim=(.25,2.85))
ax.grid(alpha=.16);ax.legend(frameon=False,ncol=4,loc='upper center')
ax.text(.32,.04,'dashed: development   solid: frozen holdout',fontsize=7,color='#555555')
ax.set_title('b  Frozen prediction in new random streams',loc='left',weight='bold')

# c: prediction audit, not a celebratory accuracy bar
ax=fig.add_subplot(gs[1,0]); detail={x['domain']:x for x in report['detail']}
match=[1 if detail[d]['class_match'] else 0 for d in domains]
fp=[detail[d]['false_safe_grid_points'] for d in domains]
y=np.arange(8);ax.barh(y,match,color=[colors[d] if m else '#D55E00' for d,m in zip(domains,match)],height=.5)
bad=[i for i,v in enumerate(fp) if v>0]
ax.scatter([1.06]*len(bad),bad,marker='x',s=35,color='#111111',label='False-safe prediction',zorder=4)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(-.03,1.12);ax.set_xticks([0,1],['mismatch','match'])
ax.grid(axis='x',alpha=.15)
ax.text(.02,1.7,'Vehicle: plateau → finite window',color='#D55E00',fontsize=7)
ax.text(.02,3.15,'Robot: one false-safe grid point',color='#333333',fontsize=7)
ax.set_title('c  Seven of eight response classes transfer',loc='left',weight='bold')

# d: strong-baseline landscape from V41; avoid claiming universal dominance
ax=fig.add_subplot(gs[1,1]); policies=['all_coupled','global_gain_reduction','independent_tracking',
 'connectivity_gate','physical_filter','two_layer_gate','supervisor']
plabel=['Coupled','Low gain','Independent','Connectivity','Physical filter','Two-layer','Supervisor']
M=np.zeros((8,7))
for i,d in enumerate(domains):
 for j,p in enumerate(policies):
  rr=[r for r in v41 if r['domain']==d and r['policy']==p]; M[i,j]=float(rr[0]['success_rate'])
im=ax.imshow(M,aspect='auto',vmin=0,vmax=1,cmap='viridis')
ax.set_xticks(range(7),plabel,rotation=28,ha='right');ax.set_yticks(range(8),labels)
for i in range(8):
 for j in range(7):
  ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if M[i,j]<.48 else '#111111')
cb=fig.colorbar(im,ax=ax,shrink=.72,pad=.02);cb.set_label('Held-out task success')
ax.set_title('d  No intervention dominates across domains',loc='left',weight='bold')

for ext in ('pdf','svg','png'):
 fig.savefig(OUT/f'Fig1_v44_theory_prediction_intervention.{ext}',dpi=400 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)

with (OUT/'Fig1_v44_source_data.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['panel','domain','split_or_policy','rho','value','source'])
 for d in domains:
  for split in ('development','heldout'):
   for rho in rhos:w.writerow(['b',d,split,rho,np.mean(g[(split,d,rho)]),'cross_domain_v44_frozen_prediction/results/v44_all_runs.csv'])
 for d in domains:
  w.writerow(['c',d,'class_match','',int(detail[d]['class_match']),'V44_FROZEN_PREDICTION_REPORT.json'])
  w.writerow(['c',d,'false_safe_grid_points','',detail[d]['false_safe_grid_points'],'V44_FROZEN_PREDICTION_REPORT.json'])
 for i,d in enumerate(domains):
  for j,p in enumerate(policies):w.writerow(['d',d,p,'',M[i,j],'cross_domain_v41/results/v41_heldout_summary.csv'])
