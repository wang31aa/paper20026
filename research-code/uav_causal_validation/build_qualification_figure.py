#!/usr/bin/env python3
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
reg=json.loads((HERE/'qualification_registry.json').read_text())
summary=json.loads((HERE/'results/official_record_summary.json').read_text())
dyn=json.loads((HERE/'results/double_integrator_test.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(7.2,2.35),constrained_layout=True)

ax=axs[0]; ax.axis('off'); levels=[('L0','archive',True),('L1','source replay',False),('L2','paired gate',False),('L3','held-out',False)]
for i,(key,label,ok) in enumerate(levels):
 y=.82-i*.22; fc='#009E73' if ok else '#E6E6E6'
 ax.add_patch(FancyBboxPatch((.08,y-.07),.72,.13,boxstyle='round,pad=.02',facecolor=fc,edgecolor='#333'))
 ax.text(.44,y,f'{key}  {label}',ha='center',va='center',color='white' if ok else '#333',weight='bold')
 if i<3: ax.annotate('',(.44,y-.15),(.44,y-.08),arrowprops={'arrowstyle':'-|>','color':'#666'})
ax.text(.02,.98,'a',weight='bold',fontsize=10,va='top');ax.set_title('Fail-closed evidence ladder',weight='bold',fontsize=9)

ax=axs[1]; vals=[summary['mpc_records'],summary['pf_records'],summary['collision_events_from_archive_threshold']]
bars=ax.bar(['NMPC','PF','events'],vals,color=['#0072B2','#E69F00','#D55E00'])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+2,str(v),ha='center',fontsize=8)
ax.set_ylim(0,105);ax.set_ylabel('complete official records');ax.set_title('All official workspaces retained',weight='bold',fontsize=9);ax.text(-.18,1.02,'b',transform=ax.transAxes,weight='bold',fontsize=10)

ax=axs[2]; metrics=[dyn['velocity_one_step_rmse'],dyn['position_explicit_rmse']]
ax.bar(['velocity','position'],metrics,color=['#009E73','#CC79A7']);ax.set_yscale('log');ax.set_ylabel('one-step RMSE');ax.set_title('Dynamics component only',weight='bold',fontsize=9)
ax.axhline(.1,color='#333',ls='--',lw=1,label='trajectory tolerance')
ax.legend(frameon=False,fontsize=7);ax.text(-.18,1.02,'c',transform=ax.transAxes,weight='bold',fontsize=10)
ax.text(.5,.10,'optimizer not replayed',transform=ax.transAxes,ha='center',color='#A33',weight='bold',fontsize=7.5)

out=ROOT/'figures/Fig23_uav_qualification'
fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(out.with_suffix('.svg'),bbox_inches='tight');fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight')
print(out)
