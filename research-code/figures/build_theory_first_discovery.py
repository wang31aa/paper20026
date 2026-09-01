#!/usr/bin/env python3
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "participation_window_validation/results.csv"
OUT = ROOT / "figures/Fig22_theory_first_discovery"
rows = list(csv.DictReader(DATA.open()))
colors = {"interior_window": "#0072B2", "monotone_benefit": "#009E73", "topology_dominant": "#D55E00"}
labels = {"interior_window": "finite window", "monotone_benefit": "monotone benefit", "topology_dominant": "contraction-dominant"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": .7,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2, hspace=.14)

# a: causal mechanism
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
def box(x, y, w, h, text, fc):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",facecolor=fc,edgecolor="#333333",linewidth=.8)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8)
box(.06,.40,.23,.18,"Participation\n$\\rho$","#E8E8E8")
box(.40,.66,.26,.18,"Contraction\n$a_i(\\rho)$","#D9F0D3")
box(.40,.40,.26,.18,"Mismatch exposure\n$D_i(\\rho)$","#FDD0A2")
box(.40,.14,.26,.18,"Information age\n$V_i h/\\rho$","#DADAEB")
box(.76,.40,.19,.18,"Task capability\n$\\Psi(\\rho)$","#C6DBEF")
for y in (.75,.49,.23): ax.add_patch(FancyArrowPatch((.29,.49),(.40,y),arrowstyle="-|>",mutation_scale=10,color="#555"))
for y in (.75,.49,.23): ax.add_patch(FancyArrowPatch((.66,y),(.76,.49),arrowstyle="-|>",mutation_scale=10,color="#555"))
ax.text(.53,.94,"one channel, competing effects",ha="center",weight="bold",fontsize=9)
ax.text(.02,.98,"a",weight="bold",fontsize=10,va="top")

# b: frozen theorem-internal predictions
ax = fig.add_subplot(gs[0, 1])
for scenario in colors:
    rr=[r for r in rows if r["scenario"]==scenario]
    x=[float(r["rho"]) for r in rr]; y=[float(r["capability_function"]) for r in rr]
    ax.plot(x,y,lw=1.8,color=colors[scenario],label=labels[scenario])
ax.axhline(1,color="#222",ls="--",lw=1,label="task threshold")
ax.set_xscale("log"); ax.set_xlim(.11,4.2); ax.set_ylim(0,2.15)
ax.xaxis.set_major_locator(FixedLocator([.12,.25,.5,1,2,4]))
ax.xaxis.set_major_formatter(FixedFormatter(["0.12","0.25","0.5","1","2","4"]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("participation strength, $\\rho$"); ax.set_ylabel("joint capability function, $\\Psi$")
ax.legend(frameon=False,fontsize=7,loc="upper right"); ax.set_title("Prespecified predictions include the counterexample",fontsize=9,weight="bold")
ax.text(-.17,1.04,"b",transform=ax.transAxes,weight="bold",fontsize=10)

# c: mathematical scope
ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
scope=[("Exact scalar function","fixed decoupled modes\nbox resources; declared controller","#C6DBEF"),
       ("Exact set-valued boundary","non-normal / switching network\nrobust reach-and-stay kernel","#D9F0D3"),
       ("Physical task decision","information → reachability →\nphysical feasibility","#FDD0A2")]
for j,(title,sub,fc) in enumerate(scope):
    y=.72-j*.30; box(.08,y,.84,.21,title+"\n"+sub,fc)
    if j<2: ax.add_patch(FancyArrowPatch((.5,y),(.5,y-.08),arrowstyle="-|>",mutation_scale=10,color="#555"))
ax.text(.50,.98,"the claim narrows as assumptions widen",ha="center",weight="bold",fontsize=9)
ax.text(.02,.98,"c",weight="bold",fontsize=10,va="top")

# d: retained evidence and promotion gate
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
domains=[("Network","exact-flow",1), ("Robot","closed-loop SIL",2), ("Vehicle","model + records",2),
         ("UAV","source replay pending",1), ("Motor","object ID",1), ("Grid","exploratory",0),
         ("Circuit","record ordering",1), ("Water/RTHS","qualification",1)]
ax.set_xlim(0,1); ax.set_ylim(-.8,len(domains)+1)
for i,(name,role,level) in enumerate(domains):
    y=len(domains)-i-.25
    ax.text(.02,y,name,va="center",fontsize=7.5)
    ax.text(.25,y,role,va="center",fontsize=7,color="#444")
    for k in range(3):
        ax.scatter(.73+.085*k,y,s=45,facecolor=("#0072B2" if k<level else "white"),edgecolor="#555",linewidth=.6)
ax.text(.73,len(domains)+.25,"qualified",fontsize=7,ha="center")
ax.text(.815,len(domains)+.25,"causal",fontsize=7,ha="center")
ax.text(.91,len(domains)+.25,"physical",fontsize=7,ha="center")
ax.plot([.02,.96],[-.05,-.05],color="#D55E00",lw=1.2)
ax.text(.49,-.42,"Promotion gate: two active HIL/physical domains\n+ one frozen leave-domain-out test",ha="center",va="center",fontsize=7,color="#A33",weight="bold")
ax.set_title("Evidence roles are not pooled",fontsize=9,weight="bold",pad=8)
ax.text(-.04,1.04,"d",transform=ax.transAxes,weight="bold",fontsize=10)

fig.savefig(OUT.with_suffix(".pdf"),bbox_inches="tight")
fig.savefig(OUT.with_suffix(".svg"),bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"),dpi=300,bbox_inches="tight")
print(OUT)
