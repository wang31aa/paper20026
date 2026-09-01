#!/usr/bin/env python3
"""Editable Nature-style summary of the prospective V40 supervisor test."""
from pathlib import Path
import csv, collections, os
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
VERSION=os.environ.get('SUPERVISOR_VERSION','v40')
rows=list(csv.DictReader((ROOT/f'cross_domain_{VERSION}/results/{VERSION}_all_policy_runs.csv').open()))
held=[r for r in rows if r['split']=='heldout']; sel=[r for r in held if r['selected_by_supervisor']=='1']
domains=['uav6dof','vehicle','motor','robot','microgrid','circuit','water','structure']
labels=['UAV','Vehicle','Motor','Robot','Grid','Circuit','Water','Structure']
colors={'supervisor':'#0072B2','independent_tracking':'#009E73','two_layer_gate':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.15,5.3)); gs=fig.add_gridspec(2,2,wspace=.32,hspace=.48)

ax=fig.add_subplot(gs[0,0]); ax.axis('off')
boxes=[(.00,.59,.27,.27,'System\ndescriptor','dynamics · graph\nresources · task'),(.365,.59,.27,.27,'Capability\noperator','robust predecessor\n+ greatest fixed point'),(.73,.59,.27,.27,'System\nboundary','viability kernel\n+ capture basin'),(.365,.14,.27,.25,'Supervisor','select a certified\naction, or abstain')]
for x,y,w,h,t,b in boxes:
 ax.add_patch(plt.Rectangle((x,y),w,h,fc='#F2F5F7',ec='#4A6572',lw=.8));ax.text(x+w/2,y+h*.68,t,ha='center',va='center',weight='bold',fontsize=7.0);ax.text(x+w/2,y+h*.28,b,ha='center',va='center',fontsize=7.0,linespacing=.95)
for a,b in [((.27,.725),(.365,.725)),((.635,.725),(.73,.725)),((.50,.59),(.50,.39))]: ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',lw=.9,color='#4A6572'))
ax.text(.50,.015,'Universal operator structure\nDomain-specific numerical boundary',ha='center',weight='bold',fontsize=7.0,color='#333333');ax.set_title('a  Capability law is an operator, not one scalar',loc='left',weight='bold')

ax=fig.add_subplot(gs[0,1]); x=np.arange(len(domains)); w=.24
for j,p in enumerate(['supervisor','independent_tracking','two_layer_gate']):
 vals=[]
 for d in domains:
  q=[r for r in (sel if p=='supervisor' else held) if r['domain']==d and (p=='supervisor' or r['policy']==p)]
  vals.append(np.mean([int(r['task_success']) for r in q]))
 ax.bar(x+(j-1)*w,vals,w,label={'supervisor':'Frozen supervisor','independent_tracking':'Independent','two_layer_gate':'Two-layer gate'}[p],color=colors[p])
ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylim(0,1.05);ax.set_ylabel('Held-out task success');ax.legend(frameon=False,ncol=1,fontsize=7,loc='upper left');ax.set_title('b  No fixed gate dominates across domains',loc='left',weight='bold')

ax=fig.add_subplot(gs[1,0]); cnt=collections.Counter(r['frozen_supervisor_policy'] for r in sel); pols=['independent_tracking','connectivity_gate','two_layer_gate','all_coupled','physical_filter','global_gain_reduction']; labs=['Independent','Connectivity gate','Two-layer gate','All coupled','Physical filter','Reduced gain']; vals=[cnt[p] for p in pols]
ax.barh(np.arange(len(pols)),vals,color=['#009E73','#56B4E9','#D55E00','#777777','#CC79A7','#E69F00']);ax.set_yticks(np.arange(len(pols)),labs);ax.invert_yaxis();ax.set_xlabel('Held-out conditions selected');ax.set_title('c  Frozen actions remain system-dependent',loc='left',weight='bold')

ax=fig.add_subplot(gs[1,1]); policies=['all_coupled','global_gain_reduction','independent_tracking','connectivity_gate','physical_filter','two_layer_gate']; rates=[]
for p in policies:
 q=[r for r in held if r['policy']==p];rates.append(np.mean([int(r['task_success']) for r in q]))
sr=np.mean([int(r['task_success']) for r in sel]); ax.bar(np.arange(7),rates+[sr],color=['#777777','#E69F00','#009E73','#56B4E9','#CC79A7','#D55E00','#0072B2']);ax.set_xticks(np.arange(7),['All','Reduced','Independent','Connectivity','Physical','Two-layer','Supervisor'],rotation=35,ha='right');ax.set_ylabel('Success across 560 tests');ax.set_ylim(0,.65);ax.axhline(sr,color='#0072B2',lw=.8,ls=':');ax.set_title('d  Prospective eight-domain evaluation',loc='left',weight='bold')
fig.suptitle('A viability supervisor replaces the claim of a universally beneficial gate',x=.055,y=.995,ha='left',fontsize=10,weight='bold')
out=ROOT/'figures';fig.savefig(out/f'Fig1_supervisor_{VERSION}.pdf',bbox_inches='tight');fig.savefig(out/f'Fig1_supervisor_{VERSION}.svg',bbox_inches='tight');fig.savefig(out/f'Fig1_supervisor_{VERSION}.png',dpi=300,bbox_inches='tight');plt.close(fig)
