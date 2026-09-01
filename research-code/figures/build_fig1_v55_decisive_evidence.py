#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
blue,orange,green,purple,grey='#0072B2','#D55E00','#009E73','#CC79A7','#6b7280'
plt.rcParams.update({'font.size':7.2,'axes.titlesize':8.5,'axes.labelsize':7.5,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none'})

def wilson(k,n,z=1.96):
 p=np.asarray(k,float)/np.asarray(n,float);n=np.asarray(n,float);den=1+z*z/n
 cen=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return cen-half,cen+half

fig,axs=plt.subplots(2,3,figsize=(7.2,4.8));ax=axs.ravel()

a=ax[0];a.set_title('a  Two network requirements',loc='left',fontweight='bold');a.axis('off');a.set(xlim=(0,1),ylim=(0,1))
pos=[(.12,.67),(.36,.67),(.60,.67),(.84,.67)]
for i,(xx,yy) in enumerate(pos):
 a.scatter(xx,yy,s=120,color='white',edgecolor=grey,lw=1.0,zorder=3);a.text(xx,yy,str(i),ha='center',va='center',fontsize=7,zorder=4)
a.scatter(.05,.90,s=90,marker='*',color=blue,zorder=3);a.text(.12,.90,'target',va='center',color=blue)
a.annotate('',pos[0],(.07,.87),arrowprops=dict(arrowstyle='->',color=blue,lw=1.4))
a.annotate('',pos[1],pos[0],arrowprops=dict(arrowstyle='->',color=blue,lw=1.4))
a.annotate('',pos[2],pos[0],arrowprops=dict(arrowstyle='->',color=blue,lw=1.4,
                                           connectionstyle='arc3,rad=-.18'))
a.annotate('',pos[3],pos[1],arrowprops=dict(arrowstyle='->',color=blue,lw=1.4,
                                           ls='--',connectionstyle='arc3,rad=-.22'))
a.scatter(.60,.79,s=34,marker='x',color=orange,lw=1.5,zorder=5)
a.text(.50,.51,'target-reachable information backbone',ha='center',color=blue)
for p,q in zip(pos[:-1],pos[1:]):
 a.plot([p[0],q[0]],[.25,.25],color=orange,lw=2)
for xx,_ in pos:a.scatter(xx,.25,s=70,color='white',edgecolor=orange,lw=1)
a.plot([pos[1][0],pos[3][0]],[.25,.25],color=orange,lw=1.2,ls=':',alpha=.7)
a.scatter(.72,.25,s=34,marker='x',color=orange,lw=1.5,zorder=5)
a.text(.50,.11,'task-admissible physical influence',ha='center',color=orange)
a.text(.73,.36,'non-local information ancestry\ndoes not certify spacing',ha='center',fontsize=7,color=grey)

b=ax[1];b.set_title('b  Conditional modal theory',loc='left',fontweight='bold')
x=np.linspace(.25,3.0,160);info=.30+.42/x;mismatch=.20+.15*x;psi=np.maximum(info,mismatch)
b.plot(x,info,color=blue,lw=1.4,label='information age');b.plot(x,mismatch,color=orange,lw=1.4,label='mismatch exposure');b.plot(x,psi,color=green,lw=1.8,label='$\\Psi(\\rho)$')
b.axhline(.68,color=grey,ls=':',lw=.9);b.fill_between(x,0,.68,where=psi<=.68,color=green,alpha=.10)
b.set(xlabel='participation $\\rho$',ylabel='normalised burden',ylim=(0,1.05));b.legend(frameon=False,loc='upper right',fontsize=7)
b.text(.03,.04,'finite window shown;\nother classes can be monotone\nor infeasible',
       transform=b.transAxes,fontsize=7,color=grey,va='bottom')

c=ax[2];c.set_title('c  Low-order UAV window',loc='left',fontweight='bold')
v8=pd.read_csv(ROOT/'uav_causal_validation/results/uav_v8_curve_summary.csv')
for pol,co,la in [('all_coupled_with_trim_sharing',orange,'all coupled'),('heterogeneity_constrained_participation_gate',green,'selective influence')]:
 q=v8[v8.policy==pol].sort_values('rho');c.plot(q.rho,q.success_rate,'o-',color=co,lw=1.3,ms=2.8,label=la);c.fill_between(q.rho,q.wilson95_low,q.wilson95_high,color=co,alpha=.12,lw=0)
c.axvspan(.95,1.4,color=green,alpha=.06);c.set(xlabel='participation $\\rho$',ylabel='task success',ylim=(-.03,1.03));c.legend(frameon=False,loc='lower center');c.text(.03,.95,'40 streams per point',transform=c.transAxes,va='top',fontsize=7)

d=ax[3];d.set_title('d  Six-DOF UAV response',loc='left',fontweight='bold')
u=pd.read_csv(ROOT/'uav_v54_sixdof/results/V54_HELDOUT.csv')
for pol,co,la in [('all_coupled',blue,'all coupled'),('two_layer_supervisor',orange,'two-layer')]:
 q=u[u.policy==pol].groupby('rho').success.agg(['sum','count','mean']);lo,hi=wilson(q['sum'],q['count']);d.errorbar(q.index,q['mean'],yerr=[q['mean']-lo,hi-q['mean']],fmt='o-',capsize=2,color=co,lw=1.3,ms=2.8,label=la)
d.set(xlabel='participation $\\rho$',ylabel='task success',ylim=(-.03,1.03));d.legend(frameon=False,loc='upper left');d.text(.97,.05,'broad trend; n=12/point',ha='right',transform=d.transAxes,fontsize=7,color=grey)

e=ax[4];e.set_title('e  Within-model class mismatch',loc='left',fontweight='bold')
dev=pd.read_csv(ROOT/'vehicle_v47_dense/results/V47_DEVELOPMENT.csv');test=pd.read_csv(ROOT/'vehicle_v47_dense/results/V47_HELDOUT.csv')
for dat,co,ls,la in [(dev,grey,'--','development'),(test,blue,'-','held-out')]:
 q=dat[dat.policy=='all_coupled'].groupby('rho').task_success.mean();e.plot(q.index,q.values,ls=ls,color=co,lw=1.5,label=la)
e.axhline(.8,color=grey,ls=':',lw=.8);e.axvline(.8667,color=grey,ls='--',lw=.8);e.axvline(.7333,color=blue,ls=':',lw=1)
e.set(xlabel='participation $\\rho$',ylabel='task success',ylim=(.2,1.03));e.legend(frameon=False,loc='lower right');e.text(.03,.94,'boundary shift = 0.13',transform=e.transAxes,fontsize=7)

f=ax[5];f.set_title('f  Vehicle interventions',loc='left',fontweight='bold')
vs=pd.read_csv(ROOT/'vehicle_v46/results/V46_POLICY_SUMMARY.csv').set_index('policy')
for pol,co,la in [('all_coupled',blue,'all coupled'),('residual_gate',purple,'residual gate'),('two_layer',orange,'two-layer'),('one_step_barrier_filter',green,'physical barrier')]:
 row=vs.loc[pol];f.scatter(row.median_margin_m,row.success_rate,s=32,color=co,label=la,alpha=.9)
f.axvline(0,color=grey,ls=':',lw=.8);f.set(xlabel='median spacing margin (m)',ylabel='task success',ylim=(.3,1.04));f.text(.03,.95,'108 paired conditions',transform=f.transAxes,va='top',fontsize=7,color=grey);f.legend(frameon=False,loc='lower right',fontsize=7)

for a in ax:
 for s in a.spines.values():s.set_linewidth(.6)
fig.subplots_adjust(left=.07,right=.99,bottom=.10,top=.96,wspace=.38,hspace=.48)
for ext in ('pdf','svg'):fig.savefig(OUT/f'Fig1_v55_decisive_evidence.{ext}')
fig.savefig(OUT/'Fig1_v55_decisive_evidence.png',dpi=400)
print('wrote redesigned Fig1_v55_decisive_evidence.{pdf,svg,png}')
