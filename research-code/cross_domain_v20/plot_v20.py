#!/usr/bin/env python3
from pathlib import Path
import csv,json
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
A=json.load(open(OUT/'v20_analysis.json')); R=json.load(open(HERE.parent/'parameter_qualification/PARAMETER_FAMILY_VALIDATION.json'))
colors={'uav6dof':'#0072B2','vehicle':'#D55E00','motor':'#009E73'}
labels={'uav6dof':'UAV','vehicle':'Vehicle','motor':'Motor'}
plt.rcParams.update({'font.family':'sans-serif','font.size':7.5,'axes.titlesize':8.5,'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.15,4.55),constrained_layout=True)

ax=axs[0,0]; stages=['Public\nevidence','Complete\nbaseline','Correlated\nfamily','Joint\nconstraints','Held-out\ntest']; xs=np.arange(len(stages))
ax.plot(xs,np.zeros_like(xs),color='#6B7280',lw=1.4,zorder=1)
ax.scatter(xs,np.zeros_like(xs),s=170,c=['#B0BEC5','#56B4E9','#E69F00','#009E73','#0072B2'],edgecolor='white',lw=.8,zorder=2)
for x,t in zip(xs,stages): ax.text(x,-.18,t,ha='center',va='top')
ax.set_ylim(-.68,.45);ax.set_xlim(-.45,4.45);ax.axis('off');ax.set_title('a  Parameter evidence is qualified in stages',loc='left',fontweight='bold')

ax=axs[0,1]
for d in labels:
 c=A['participation_curves'][d]['curve'];x=[z['rho'] for z in c];y=[z['success_rate'] for z in c]
 ax.plot(x,y,'o-',label=labels[d],color=colors[d],lw=1.7,ms=4)
ax.set(xlabel='Participation, $\\rho$',ylabel='Held-out task success',ylim=(-.03,.72));ax.grid(alpha=.18);ax.legend(frameon=False,ncol=3,loc='upper right');ax.set_title('b  Frozen participation curves',loc='left',fontweight='bold')

ax=axs[1,0]; domains=list(labels); policies=['all_coupled','global_gain_reduction','two_layer_gate']; names=['All coupled','Global reduction','Two-layer gate'];x=np.arange(3);w=.23
for j,p in enumerate(policies):
 ax.bar(x+(j-1)*w,[A['summary'][d][p]['success_rate'] for d in domains],w,label=names[j],color=['#B0BEC5','#E69F00','#0072B2'][j])
ax.set_xticks(x,labels=[labels[d] for d in domains]);ax.set(ylabel='Held-out task success',ylim=(0,.55));ax.legend(frameon=False,fontsize=6.5);ax.set_title('c  No intervention dominates every domain',loc='left',fontweight='bold')

ax=axs[1,1]; params=[]; lo=[];hi=[]; groups=[]
for d in ['uav','vehicle','motor']:
 for k,v in R['summaries'][d].items():
  params.append(k.replace('_ratio','').replace('_',' '));lo.append(v['min']);hi.append(v['max']);groups.append(d)
y=np.arange(len(params));
for i,(a,b,d) in enumerate(zip(lo,hi,groups)): ax.plot([a,b],[i,i],lw=3,color=colors['uav6dof' if d=='uav' else d]);ax.plot([a,b],[i,i],'|',color='#222',ms=5)
ax.axvline(1,color='#777',ls='--',lw=.8);ax.set_yticks(y,params,fontsize=5.8);ax.invert_yaxis();ax.set(xlabel='Ratio to complete baseline');ax.grid(axis='x',alpha=.15);ax.set_title('d  Correlated physical parameter families',loc='left',fontweight='bold')

for ext in ('pdf','svg','png'): fig.savefig(OUT/f'Fig_V20_physical_parameter_test.{ext}',dpi=400,bbox_inches='tight')

with (OUT/'Fig_V20_source_data.csv').open('w',newline='') as f:
 wri=csv.writer(f);wri.writerow(['panel','domain','item','x','value'])
 for d in labels:
  for z in A['participation_curves'][d]['curve']:wri.writerow(['b',d,'success_rate',z['rho'],z['success_rate']])
  for p in policies:wri.writerow(['c',d,p,'',A['summary'][d][p]['success_rate']])
 for p,a,b,d in zip(params,lo,hi,groups):wri.writerow(['d',d,p,a,b])
print(OUT/'Fig_V20_physical_parameter_test.pdf')
