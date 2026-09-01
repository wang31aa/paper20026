#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
v24=json.loads((ROOT/'cross_domain_v24/results/v24_analysis.json').read_text())
q24=json.loads((ROOT/'cross_domain_v24/results/V24_VALIDATION.json').read_text())
v25=json.loads((ROOT/'cross_domain_v25/results/V25_VALIDATION_ANALYSIS.json').read_text())
v24rows=list(csv.DictReader((ROOT/'cross_domain_v24/results/v24_runs.csv').open()))
v25rows=list(csv.DictReader((ROOT/'cross_domain_v25/results/v25_runs.csv').open()))

def wilson(successes,n,z=1.959963984540054):
    if n == 0:
        return float('nan'),float('nan')
    p=successes/n; den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return centre-half,centre+half

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42})
colors={'uav6dof':'#0072B2','vehicle':'#D55E00','motor':'#009E73'}
labels={'uav6dof':'UAV','vehicle':'Vehicle','motor':'Motor'}
fig,axs=plt.subplots(2,2,figsize=(10.8,6.4),constrained_layout=True)

# a: theorem architecture
ax=axs[0,0]; ax.set_axis_off(); ax.set_xlim(0,1);ax.set_ylim(0,1)
boxes=[(.04,.64,.27,.22,'Target information\nreachable backbone','#DCEAF7'),(.365,.64,.27,.22,'Physical influence\nselected edges','#FBE5D5'),(.69,.64,.27,.22,'Task feasibility\nrobust kernel','#DFF0E6')]
for x,y,w,h,t,c in boxes:
 ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=c,edgecolor='#333333',lw=.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',weight='bold')
for x in (.31,.635):ax.annotate('',xy=(x+.05,.75),xytext=(x,.75),arrowprops=dict(arrowstyle='->',lw=1.2,color='#444'))
ax.text(.5,.43,'Information enrichment enlarges available actions',ha='center',color='#0072B2')
ax.text(.5,.30,'Physical coupling changes the plant and has no universal sign',ha='center',color='#D55E00')
ax.text(.5,.12,r'$x_0\in\mathfrak{U}(\Xi)$  iff a robust task-preserving policy exists',ha='center',fontsize=9)
ax.set_title('a  Two-layer capability principle',loc='left',weight='bold')

# b: curves under certified switching
ax=axs[0,1]
for d,curve in v24['all_coupled_certified_switching_curve'].items():
 x=np.array(sorted(float(k) for k in curve)); y=[]; lo=[]; hi=[]
 for rho in x:
  rr=[r for r in v24rows if r['domain']==d and r['policy']=='all_coupled' and r['topology']=='certified_switching' and r['split']=='heldout' and abs(float(r['rho'])-rho)<1e-12]
  s=sum(int(r['task_success']) for r in rr); a,b=wilson(s,len(rr)); y.append(s/len(rr));lo.append(a);hi.append(b)
 y=np.asarray(y);lo=np.asarray(lo);hi=np.asarray(hi)
 ax.fill_between(x,lo,hi,color=colors[d],alpha=.12,lw=0)
 ax.plot(x,y,'o-',lw=1.6,ms=4,label=f"{labels[d]} (n={len(rr)} per point)",color=colors[d])
ax.set(xlabel='Participation strength, ρ',ylabel='Held-out task success',ylim=(-.04,1.04));ax.grid(alpha=.2)
ax.legend(frameon=False,ncol=1,loc='best')
ax.set_title('b  One theorem permits different critical shapes',loc='left',weight='bold')

# c: common metric evidence on one physical scale
ax=axs[1,0]; keys=[];vals=[]
for d in ('uav6dof','vehicle','motor'):
 for n in (5,20):
  rr=[float(r['common_metric_mu']) for r in v24rows if r['domain']==d and int(r['n'])==n and r['topology']=='certified_switching']
  keys.append(f'{labels[d]}\nN={n}');vals.append(min(rr))
bars=ax.bar(np.arange(len(vals)),vals,color=[colors[d] for d in ('uav6dof','uav6dof','vehicle','vehicle','motor','motor')],width=.68)
ax.axhline(0,color='#333',lw=.8);ax.set_xticks(np.arange(len(vals)),keys);ax.set_ylabel(r'Minimum $\lambda_{\min}(H_k+H_k^\top)$');ax.grid(axis='y',alpha=.2)
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.0015,f'{v:.3f}',ha='center',va='bottom',fontsize=7)
ax.set_title('c  Directed switching shares one metric',loc='left',weight='bold')

# d: fair policies
ax=axs[1,1]; policies=['all_coupled','global_gain_reduction','independent_tracking','connectivity_gate','physical_filter','two_layer_gate'];short=['All','Low gain','Independent','Conn. gate','Physical','Two-layer'];xx=np.arange(len(policies));width=.24
for j,d in enumerate(('uav6dof','vehicle','motor')):
 y=[];elow=[];ehigh=[]
 for p in policies:
  rr=[r for r in v25rows if r['domain']==d and r['policy']==p and r['split']=='heldout']
  s=sum(int(r['task_success']) for r in rr);a,b=wilson(s,len(rr));v=s/len(rr);y.append(v);elow.append(v-a);ehigh.append(b-v)
 ax.bar(xx+(j-1)*width,y,width,label=labels[d],color=colors[d],yerr=np.array([elow,ehigh]),capsize=2,error_kw={'lw':.7})
ax.set_xticks(xx,short,rotation=25,ha='right');ax.set_ylabel('Held-out task success');ax.set_ylim(0,1.04);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,ncol=3,loc='upper center')
ax.set_title('d  No controller dominates across domains',loc='left',weight='bold')

for ext in ('pdf','svg','png'):
 fig.savefig(OUT/f'Fig1_v25_two_layer_discovery.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)

with (OUT/'Fig1_v25_source_data.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['panel','domain_or_metric','x_or_policy','value','wilson_lower','wilson_upper','n'])
 for d,curve in v24['all_coupled_certified_switching_curve'].items():
  for rho in sorted(float(k) for k in curve):
   rr=[r for r in v24rows if r['domain']==d and r['policy']=='all_coupled' and r['topology']=='certified_switching' and r['split']=='heldout' and abs(float(r['rho'])-rho)<1e-12]
   s=sum(int(r['task_success']) for r in rr);a,b=wilson(s,len(rr));w.writerow(['b',d,rho,s/len(rr),a,b,len(rr)])
 for name,v in zip(keys,vals):w.writerow(['c',name,'common_metric_margin',v,'','',len([r for r in v24rows if r['topology']=='certified_switching'])])
 for d in ('uav6dof','vehicle','motor'):
  for p in policies:
   rr=[r for r in v25rows if r['domain']==d and r['policy']==p and r['split']=='heldout']
   s=sum(int(r['task_success']) for r in rr);a,b=wilson(s,len(rr));w.writerow(['d',d,p,s/len(rr),a,b,len(rr)])
