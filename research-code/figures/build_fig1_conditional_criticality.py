#!/usr/bin/env python3
"""Build the theory-led conditional-criticality discovery figure."""
from pathlib import Path
import csv,json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
v20=json.load(open(ROOT/'cross_domain_v20/results/v20_analysis.json'))
v21=json.load(open(ROOT/'cross_domain_v21/results/v21_analysis.json'))
blue='#0072B2'; orange='#D55E00'; green='#009E73'; grey='#7A8793'; pale='#F4F6F7'; red='#B23A48'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                     'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7.6,6.35)); gs=fig.add_gridspec(2,2,left=.07,right=.98,bottom=.08,top=.96,wspace=.36,hspace=.43)

# a: theorem-led architecture
ax=fig.add_subplot(gs[0,0]); ax.set_axis_off(); ax.set_title('a  Two network resources',loc='left',fontweight='bold',fontsize=8.5)
boxes=[(.03,.62,.42,.19,'Information ancestry','visibility · delay\nupdate rate',blue),
       (.55,.62,.42,.19,'Physical influence','heterogeneity · authority\ntask set',orange),
       (.29,.26,.42,.19,'Capability functional','winning set or\nrestricted-class $\\Psi_s(\\rho)$',green)]
for x,y,w,h,title,sub,c in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.025',facecolor=pale,edgecolor=c,linewidth=1.3))
    ax.text(x+w/2,y+h*.65,title,ha='center',va='center',fontweight='bold',color=c,fontsize=7.7)
    ax.text(x+w/2,y+h*.27,sub,ha='center',va='center',fontsize=7.0,linespacing=1.10)
for x0,x1 in ((.24,.42),(.76,.58)):
    ax.add_patch(FancyArrowPatch((x0,.60),(x1,.47),arrowstyle='-|>',mutation_scale=9,color=grey,linewidth=1.1))
ax.text(.50,.08,'Prediction: window, monotone response, or empty set',ha='center',fontsize=7.0,fontweight='bold')
ax.set_xlim(0,1); ax.set_ylim(0,1)

# b: held-out curves
ax=fig.add_subplot(gs[0,1]); ax.set_title('b  Window and monotone classes',loc='left',fontweight='bold',fontsize=8.5)
for domain,label,color in [('uav6dof','UAV V20',blue),('vehicle','Vehicle V20',orange)]:
    curve=v20['participation_curves'][domain]['curve']; ax.plot([x['rho'] for x in curve],[x['success_rate'] for x in curve],'-o',ms=3.8,lw=1.5,label=label,color=color)
curve=v21['participation_curve']; ax.plot([x['rho'] for x in curve],[x['success_rate'] for x in curve],'-o',ms=3.8,lw=1.5,label='Motor V21',color=green)
ax.set(xlabel='Participation, $\\rho$',ylabel='Held-out task success',ylim=(-.04,1.05)); ax.grid(alpha=.22); ax.legend(frameon=False,loc='upper left')
ax.annotate('finite windows',xy=(.95,.51),xytext=(.55,.72),arrowprops=dict(arrowstyle='->',color=grey),fontsize=7)
ax.annotate('monotone',xy=(2.35,1),xytext=(1.75,.77),arrowprops=dict(arrowstyle='->',color=grey),fontsize=7)

# c: motor contract repair
ax=fig.add_subplot(gs[1,0]); ax.set_title('c  Frozen motor-controller repair',loc='left',fontweight='bold',fontsize=8.5)
rhos=[x['rho'] for x in v20['participation_curves']['motor']['curve']]
old=[x['success_rate'] for x in v20['participation_curves']['motor']['curve']]
new=[x['success_rate'] for x in v21['participation_curve']]
ax.plot(rhos,old,'--o',ms=3.8,color=red,label='V20 sampled controller')
ax.plot(rhos,new,'-o',ms=3.8,color=green,label='V21 time-scale-aware PI')
ax.fill_between(rhos,old,new,color=green,alpha=.10)
ax.set(xlabel='Participation, $\\rho$',ylabel='Held-out task success',ylim=(-.04,1.05)); ax.grid(alpha=.22); ax.legend(frameon=False,loc='upper left')
ax.text(.53,.12,'control-contract failure',transform=ax.transAxes,color=red,fontsize=7)
ax.text(.53,.72,'no high-participation loss',transform=ax.transAxes,color=green,fontsize=7)

# d: policy ordering
ax=fig.add_subplot(gs[1,1]); ax.set_title('d  No intervention dominates',loc='left',fontweight='bold',fontsize=8.5)
domains=['UAV\nV20','Vehicle\nV20','Motor\nV21']; x=range(3); w=.24
allv=[v20['summary']['uav6dof']['all_coupled']['success_rate'],v20['summary']['vehicle']['all_coupled']['success_rate'],v21['summary']['all_coupled_pi']['success_rate']]
gain=[v20['summary']['uav6dof']['global_gain_reduction']['success_rate'],v20['summary']['vehicle']['global_gain_reduction']['success_rate'],v21['summary']['global_gain_reduction']['success_rate']]
gate=[v20['summary']['uav6dof']['two_layer_gate']['success_rate'],v20['summary']['vehicle']['two_layer_gate']['success_rate'],v21['summary']['two_layer_pi']['success_rate']]
ax.bar([i-w for i in x],allv,w,label='Permanent coupling',color='#B7C2C8'); ax.bar(x,gain,w,label='Global reduction',color='#E69F00'); ax.bar([i+w for i in x],gate,w,label='Two-layer gate',color=blue)
ax.set_xticks(list(x),domains); ax.set_ylabel('Held-out task success'); ax.set_ylim(0,1.04); ax.grid(axis='y',alpha=.22); ax.legend(frameon=False,loc='upper left')

fig.savefig(OUT/'Fig1_conditional_criticality.pdf',bbox_inches='tight')
fig.savefig(OUT/'Fig1_conditional_criticality.svg',bbox_inches='tight')
fig.savefig(OUT/'Fig1_conditional_criticality.png',dpi=300,bbox_inches='tight')
rows=[]
for d in ('uav6dof','vehicle','motor'):
    curve=v21['participation_curve'] if d=='motor' else v20['participation_curves'][d]['curve']
    for q in curve: rows.append({'panel':'b','domain':d,'rho':q['rho'],'value':q['success_rate'],'metric':'heldout_task_success'})
for i,d in enumerate(('uav6dof','vehicle','motor')):
    for p,v in [('all_coupled',allv[i]),('global_gain_reduction',gain[i]),('two_layer_gate',gate[i])]: rows.append({'panel':'d','domain':d,'rho':'','value':v,'metric':p})
with (OUT/'Fig1_conditional_criticality_source_data.csv').open('w',newline='') as f:
    wri=csv.DictWriter(f,fieldnames=['panel','domain','rho','value','metric']); wri.writeheader(); wri.writerows(rows)
print('PASS: editable theory-led discovery figure and source data written')
