#!/usr/bin/env python3
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
rows=list(csv.DictReader((HERE/'results/v23_runs.csv').open()))
held=[r for r in rows if r['split']=='heldout']; rhos=sorted({float(r['rho']) for r in held})
domains=['uav6dof','vehicle','motor']; labels={'uav6dof':'UAV','vehicle':'Vehicle','motor':'Motor'}
policies=['all_coupled','global_gain_reduction','connectivity_gate','two_layer_gate']
colors={'uav6dof':'#0072B2','vehicle':'#D55E00','motor':'#009E73'}
summary={}
for d in domains:
    q=[r for r in held if r['domain']==d]
    summary[d]={'n':len(q),'success_by_rho':{},'success_by_policy':{}}
    for rho in rhos:
        z=[r for r in q if r['policy']=='all_coupled' and float(r['rho'])==rho]
        summary[d]['success_by_rho'][str(rho)]=sum(int(r['task_success']) for r in z)/len(z)
    for p in policies:
        z=[r for r in q if r['policy']==p]
        summary[d]['success_by_policy'][p]=sum(int(r['task_success']) for r in z)/len(z)
(HERE/'results/v23_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.4,'axes.linewidth':.7,'svg.fonttype':'none'})
fig,ax=plt.subplots(2,2,figsize=(7.2,5.1),constrained_layout=True)

# a: mechanism and qualification flow
a=ax[0,0];a.axis('off')
boxes=[(.02,.62,.25,.25,'Information\nbackbone','#DCEAF7'),(.375,.62,.25,.25,'Physical\ninfluence','#F8E1D5'),(.73,.62,.25,.25,'Task-feasible\nintervention','#DCEFE5')]
for x,y,w,h,t,c in boxes:a.add_patch(plt.Rectangle((x,y),w,h,fc=c,ec='#333333',lw=.8));a.text(x+w/2,y+h/2,t,ha='center',va='center',weight='bold',fontsize=7)
for start,end in ((.275,.365),(.63,.72)):a.annotate('',(end,.745),(start,.745),arrowprops=dict(arrowstyle='->',lw=.9))
a.text(.02,.42,'Corrected observer',weight='bold');a.text(.02,.31,r'$L=\mathrm{diag}(A\mathbf{1})-A,\quad H\mathbf{1}=V\mathbf{1}$')
a.text(.02,.16,'Frozen test: fixed parameters + matched random streams + actual graph changes',fontsize=7)
a.text(.0,.98,'a',weight='bold',fontsize=9,va='top')

# b observer/switch qualification
b=ax[0,1];
tops=['chain','random_directed','switching'];err=[];rad=[];distinct=[]
for t in tops:
 q=[r for r in held if r['topology']==t];err.append(max(float(r['observer_identity_error']) for r in q));rad.append(max(float(r['observer_transition_radius']) for r in q));distinct.append(max(int(r['distinct_adjacencies']) for r in q))
x=np.arange(3);b.bar(x,rad,color='#7F8C8D',width=.58);b.axhline(1,color='#B2182B',ls='--',lw=.9);b.set_ylim(.96,1.001);b.set_ylabel('max transition spectral radius');b.set_xticks(x,['chain','random directed','switching'],rotation=15,ha='right');
for i,n in enumerate(distinct):b.text(i,rad[i]-.003,f'{n} graph'+('s' if n>1 else ''),ha='center',va='top',color='white',fontsize=7,weight='bold')
b.text(.02,.90,r'max $|H\mathbf{1}-V\mathbf{1}|=2.2\times10^{-16}$',transform=b.transAxes,va='bottom',fontsize=7);b.text(-.18,1.02,'b',transform=b.transAxes,weight='bold',fontsize=9)

# c held-out response curves
c=ax[1,0]
for d in domains:c.plot(rhos,[summary[d]['success_by_rho'][str(r)] for r in rhos],'-o',ms=3.5,lw=1.4,label=labels[d],color=colors[d])
c.set_xlabel('participation, '+r'$\rho$');c.set_ylabel('held-out task success');c.set_ylim(-.03,1.04);c.legend(frameon=False,ncol=3,fontsize=7,loc='upper left');c.text(-.18,1.03,'c',transform=c.transAxes,weight='bold',fontsize=9)

# d policy performance, no dominance implication
dax=ax[1,1];w=.2;xx=np.arange(3)
for j,p in enumerate(policies):dax.bar(xx+(j-1.5)*w,[summary[d]['success_by_policy'][p] for d in domains],w,label=p.replace('_',' '))
dax.set_xticks(xx,[labels[d] for d in domains]);dax.set_ylabel('held-out task success');dax.set_ylim(0,1);dax.legend(frameon=False,fontsize=7,ncol=2,loc='upper left');dax.text(-.18,1.03,'d',transform=dax.transAxes,weight='bold',fontsize=9)
for ext in ('pdf','svg','png'):fig.savefig(ROOT/'figures'/f'Fig1_v23_corrected_closure.{ext}',dpi=300)
source=[]
for d in domains:
 for rho in rhos:source.append({'panel':'c','domain':d,'rho':rho,'policy':'all_coupled','value':summary[d]['success_by_rho'][str(rho)]})
 for p in policies:source.append({'panel':'d','domain':d,'rho':'','policy':p,'value':summary[d]['success_by_policy'][p]})
with (ROOT/'figures/Fig1_v23_corrected_closure_source_data.csv').open('w',newline='') as f:wri=csv.DictWriter(f,source[0]);wri.writeheader();wri.writerows(source)
print(json.dumps(summary,indent=2))
