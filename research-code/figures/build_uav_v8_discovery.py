#!/usr/bin/env python3
"""Nature-style V8 mechanism and held-out response figure."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/"uav_causal_validation/results"
OUT=Path(__file__).resolve().parent
curve=list(csv.DictReader((RES/"uav_v8_curve_summary.csv").open()))
mech=list(csv.DictReader((RES/"uav_v8_mechanism_summary.csv").open()))
colors={"all_coupled_with_trim_sharing":"#3B6FB6","heterogeneity_constrained_participation_gate":"#D55E00","independent_constrained_tracking":"#009E73"}
labels={"all_coupled_with_trim_sharing":"All coupled","heterogeneity_constrained_participation_gate":"Heterogeneity gate","independent_constrained_tracking":"Independent tracking"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.labelsize":7,"axes.titlesize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"axes.linewidth":.65,"svg.fonttype":"none","pdf.fonttype":42})
fig,axs=plt.subplots(2,2,figsize=(7.08,5.2),constrained_layout=True)

ax=axs[0,0]
for policy in colors:
    z=sorted((r for r in curve if r["policy"]==policy),key=lambda r:float(r["rho"]))
    x=np.array([float(r["rho"]) for r in z]); y=np.array([float(r["success_rate"]) for r in z])
    lo=np.array([float(r["wilson95_low"]) for r in z]); hi=np.array([float(r["wilson95_high"]) for r in z])
    ax.plot(x,y,"o-",lw=1.3,ms=3,color=colors[policy],label=labels[policy]);ax.fill_between(x,lo,hi,color=colors[policy],alpha=.12,lw=0)
ax.axvspan(.35,.55,color="#B2182B",alpha=.07);ax.axvspan(1.1,1.9,color="#4DAF4A",alpha=.07);ax.axvspan(2.5,2.8,color="#B2182B",alpha=.07)
ax.set(xlabel="Participation strength, $\\rho$",ylabel="Task success fraction",ylim=(-.04,1.05),title="a  Frozen dense participation response")
ax.legend(frameon=False,loc="lower center",ncol=1)

ax=axs[0,1]
arms=["homogeneous","all_heterogeneous_without_feedforward_sharing","trim_only","all_heterogeneous"]
arm_labels=["Homogeneous","No mismatch sharing","Trim only","All heterogeneous"]
rhos=sorted({float(r["rho"]) for r in mech})
M=np.array([[float(next(r["success_rate"] for r in mech if r["arm"]==a and float(r["rho"])==rho)) for rho in rhos] for a in arms])
im=ax.imshow(M,aspect="auto",vmin=0,vmax=1,cmap="RdYlGn",origin="upper")
ax.set_xticks(range(len(rhos)),[f"{x:g}" for x in rhos]);ax.set_yticks(range(len(arms)),arm_labels)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=7,color="black")
ax.set(xlabel="Participation strength, $\\rho$",title="b  Mechanism ablation (success fraction)")
fig.colorbar(im,ax=ax,fraction=.046,pad=.03,label="Success fraction")

ax=axs[1,0]
for policy in colors:
    z=sorted((r for r in curve if r["policy"]==policy),key=lambda r:float(r["rho"]))
    ax.plot([float(r["rho"]) for r in z],[float(r["median_tail_error_m"]) for r in z],"o-",lw=1.3,ms=3,color=colors[policy],label=labels[policy])
ax.axhline(.14,color="black",ls="--",lw=.9,label="Frozen task tube")
ax.set(xlabel="Participation strength, $\\rho$",ylabel="Median tail error (m)",title="c  Failure and recovery use a physical error scale")

ax=axs[1,1]
for rho,marker in [(1.25,"o"),(2.5,"s")]:
    for policy in colors:
        r=next(r for r in curve if r["policy"]==policy and abs(float(r["rho"])-rho)<1e-9)
        ax.scatter(float(r["median_control_energy"]),float(r["median_tail_error_m"]),s=32,marker=marker,color=colors[policy],edgecolor="white",linewidth=.5)
ax.axhline(.14,color="black",ls="--",lw=.9)
ax.text(.02,.97,"○ intermediate   □ high participation",transform=ax.transAxes,va="top",fontsize=7)
ax.set(xlabel="Median acceleration-squared effort (m² s⁻³)",ylabel="Median tail error (m)",title="d  Recovery incurs a control-cost trade-off")
for ax in axs.flat:
    ax.spines[["top","right"]].set_visible(False);ax.tick_params(width=.6,length=3)
fig.savefig(OUT/"Fig24_uav_v8_discovery.pdf",bbox_inches="tight")
fig.savefig(OUT/"Fig24_uav_v8_discovery.svg",bbox_inches="tight")
fig.savefig(OUT/"Fig24_uav_v8_discovery.png",dpi=300,bbox_inches="tight")
print(OUT/"Fig24_uav_v8_discovery.pdf")
