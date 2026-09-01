#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;O=H/"results"
rows=list(csv.DictReader((O/"V57_HELDOUT.csv").open()))
policies=["all_coupled","physical_filter","two_layer_physical_filter"]
labels={"all_coupled":"all coupled","physical_filter":"physical filter","two_layer_physical_filter":"two-layer + filter"}
colors={"all_coupled":"#2F6B9A","physical_filter":"#D9822B","two_layer_physical_filter":"#2A9D8F"}
rhos=sorted({float(x["rho"]) for x in rows});z=1.96
plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"legend.fontsize":7,"svg.fonttype":"none"})
fig,axs=plt.subplots(1,2,figsize=(7.2,2.75),constrained_layout=True)
for p in policies:
    ys=[];lo=[];hi=[];cl=[]
    for rho in rhos:
        q=[x for x in rows if x["policy"]==p and float(x["rho"])==rho];n=len(q);ph=sum(int(x["success"]) for x in q)/n
        den=1+z*z/n;cen=(ph+z*z/(2*n))/den;rad=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        ys.append(ph);lo.append(ph-max(0,cen-rad));hi.append(min(1,cen+rad)-ph);cl.append(np.median([float(x["minimum_clearance_m"]) for x in q]))
    axs[0].errorbar(rhos,ys,yerr=[lo,hi],marker="o",lw=1.4,capsize=2,color=colors[p],label=labels[p])
    axs[1].plot(rhos,cl,marker="o",lw=1.4,color=colors[p],label=labels[p])
axs[0].set(xlabel="participation $\\rho$",ylabel="task success",ylim=(-.03,1.03),title="a  New-stream task outcome")
axs[1].axhline(.45,color="#555555",ls="--",lw=1,label="clearance threshold")
axs[1].set(xlabel="participation $\\rho$",ylabel="median minimum clearance (m)",title="b  Physical task margin")
for ax in axs:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#DDDDDD",lw=.5)
axs[0].legend(frameon=False,loc="upper left");axs[1].legend(frameon=False,loc="lower right")
fig.savefig(O/"Fig_V57_task_filter.pdf",bbox_inches="tight")
fig.savefig(O/"Fig_V57_task_filter.svg",bbox_inches="tight")
fig.savefig(O/"Fig_V57_task_filter.png",dpi=400,bbox_inches="tight")
