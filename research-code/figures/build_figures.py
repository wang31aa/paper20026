#!/usr/bin/env python3
"""Build four evidence-traceable figures from the v6 audit outputs.

No values are embedded from the manuscript and no failed runs are removed.
Every quantitative panel is computed from code/results or theory/result.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
THEORY = ROOT / "theory" / "result.json"
ABLATION_RESULTS = ROOT / "baseline_ablation" / "results"
EXTENSION_RESULTS = ROOT / "extension_study" / "results"
CERTIFIED_RESULTS = ROOT / "certified_benchmark" / "results"
OUT = Path(__file__).resolve().parent

# Nature-compatible compact typography and colour-blind-safe accents.
INK = "#202124"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8A8F98"
LIGHT = "#F3F5F7"
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "nature-tac2023-v7",
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def panel_letters(axes):
    for letter, ax in zip(string.ascii_lowercase, np.ravel(axes)):
        ax.text(-0.10, 1.20, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=10, va="top", ha="left")


def clean(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9DDE2", lw=0.5, alpha=0.65)
        ax.set_axisbelow(True)


def save(fig, stem, min_text=8.6, tight=True):
    # Materialize tick and colour-bar labels before applying the final-size gate.
    fig.canvas.draw()
    for item in fig.findobj(match=Text):
        item.set_fontsize(max(item.get_fontsize(), min_text))
    bbox = "tight" if tight else None
    with mpl.rc_context({"savefig.bbox": bbox}):
        fig.savefig(OUT / f"{stem}.svg", metadata={"Date": None}, bbox_inches=bbox)
        fig.savefig(OUT / f"{stem}.pdf",
                    metadata={"CreationDate": None, "ModDate": None}, bbox_inches=bbox)
        fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches=bbox)
        fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches=bbox)
    plt.close(fig)


def fig1_workflow():
    """Schematic-led overview of the supported certificate architecture."""
    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, .95], wspace=.12)
    fig.subplots_adjust(left=.035, right=.985, top=.94, bottom=.10, wspace=.12)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_chain = fig.add_subplot(gs[0, 1])
    for ax in (ax_net, ax_chain):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # a, physical/information layer. Receiver-row arrows point source -> receiver.
    pos = {"s": (.13, .54), "1": (.48, .78), "2": (.72, .78),
           "3": (.48, .30), "4": (.72, .30), "5": (.88, .54)}
    edges = [("s", "1", True), ("s", "3", True), ("1", "2", False),
             ("1", "4", False), ("3", "4", False), ("2", "5", False),
             ("4", "5", False)]
    for src, dst, pinned in edges:
        ax_net.annotate("", pos[dst], pos[src],
                        arrowprops=dict(arrowstyle="-|>", lw=.85,
                                        color=VERMILION if pinned else "#65717E",
                                        shrinkA=13, shrinkB=13,
                                        connectionstyle="arc3,rad=.05"))
    ax_net.scatter(*pos["s"], s=820, color=VERMILION, edgecolor="white", lw=.8, zorder=3)
    ax_net.text(*pos["s"], "$s$", ha="center", va="center", color="white",
                weight="bold", fontsize=7.0, zorder=4)
    for node in "12345":
        ax_net.scatter(*pos[node], s=470, color=SKY, edgecolor="white", lw=.8, zorder=3)
        ax_net.text(*pos[node], node, ha="center", va="center", color=INK,
                    weight="bold", fontsize=7.0, zorder=4)
    ax_net.plot([], [], color=VERMILION, lw=1.2, label="target information")
    ax_net.plot([], [], color="#65717E", lw=1.2, label="follower exchange")
    ax_net.legend(loc="lower center", bbox_to_anchor=(.50, .17), ncol=2,
                  frameon=False, handlelength=1.5, columnspacing=1.2, fontsize=6.5)
    ax_net.text(.50, .065,
                r"distributed target estimate $\widehat s_i$"
                r" $\longrightarrow$ local control $u_i$",
                ha="center", fontsize=6.6, color=INK)
    ax_net.text(-.02, .99, "a", weight="bold", fontsize=8, va="top")
    ax_net.text(.04, .98, "Distributed information flow", weight="bold",
                fontsize=7.0, va="top")

    # b, equation-to-certificate chain without audit-card decoration.
    boxes = [
        (.78, "Directed graph", r"$GL_1+L_1^\mathsf{T}G\succ0$", BLUE),
        (.55, "Local increment bound", r"IQCs; $Q\succ0$; $QH=H^\mathsf{T}Q\succ0$", PURPLE),
        (.32, "Bounded disturbances", "mismatch + observer/implementation\nerror $v_i$", ORANGE),
        (.09, "Ultimate tracking bound", r"$\limsup_{t\to\infty}\Vert e(t)\Vert_2\leq\Delta_{\rm stat}$", GREEN),
    ]
    for y, head, body, color in boxes:
        if head == "Ultimate tracking bound":
            ax_chain.add_patch(FancyBboxPatch((.09, y-.015), .82, .17,
                              boxstyle="round,pad=.012,rounding_size=.014",
                              facecolor="#ECF7F2", edgecolor="none"))
        ax_chain.plot([.10, .10], [y+.02, y+.13], color=color, lw=.9,
                      solid_capstyle="round")
        ax_chain.text(.15, y+.095, head, weight="bold", color=color, fontsize=6.8)
        ax_chain.text(.15, y+.025, body, color=INK, fontsize=6.5, linespacing=.9)
    for y in (.755, .525, .295):
        ax_chain.annotate("", (.50, y-.055), (.50, y-.005),
                          arrowprops=dict(arrowstyle="-|>", color="#65717E", lw=.8))
    ax_chain.text(-.02, .99, "b", weight="bold", fontsize=8, va="top")
    ax_chain.text(.04, .98, "Certificate chain", weight="bold",
                  fontsize=7.0, va="top")
    save(fig, "Fig1_certificate_framework", min_text=7.0, tight=False)


def fig2_robustness(raw):
    trees = [f"random_tree_{i}" for i in range(5)]
    hs = sorted(raw.heterogeneity.unique())
    med = raw.pivot_table(index="topology", columns="heterogeneity", values="final_error", aggfunc="median").loc[trees, hs]
    q95 = raw.pivot_table(index="topology", columns="heterogeneity", values="q95_error", aggfunc="median").loc[trees, hs]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"width_ratios":[1.05, 1.05, 1.15]})
    for ax, mat, title in [(axes[0], med, "Median final error"), (axes[1], q95, "Median 95th-percentile error")]:
        im=ax.pcolormesh(np.arange(6)-.5, np.arange(6)-.5, mat.values,
                         cmap="viridis", norm=LogNorm(vmin=min(med.values.min(),q95.values.min()),
                         vmax=max(med.values.max(),q95.values.max())), shading="flat")
        ax.set_xticks(range(5), [f"{h:g}" for h in hs]); ax.set_yticks(range(5), [f"T{i+1}" for i in range(5)])
        ax.set_xlabel("Heterogeneity multiplier"); ax.set_ylabel("Directed tree")
        ax.set_title(title)
        # Exact cell values remain in Source Data; colour avoids unreadable
        # cross-cell numeric strings at final journal width.
    data=[raw.loc[raw.heterogeneity==h,"final_error"].values for h in hs]
    assert all(np.all(d > 0) for d in data), "log-scale errors must be positive"
    bp=axes[2].boxplot(data, positions=range(5), widths=.56, patch_artist=True, showfliers=False,
                      medianprops=dict(color=INK,lw=1), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    for p,c in zip(bp["boxes"],[SKY,BLUE,GREEN,ORANGE,VERMILION]): p.set(facecolor=c,alpha=.65,edgecolor="white")
    for j,d in enumerate(data):
        # Deterministic display-only jitter; no observations are sampled or removed.
        jitter = .17 * np.sin(np.arange(len(d)) * 2.399963229728653)
        axes[2].scatter(j+jitter,d,s=3,c=INK,alpha=.18,lw=0)
    # The asserted source metrics are strictly positive before log-axis rendering.
    axes[2].set_xticks(range(5),[f"{h:g}" for h in hs]); axes[2].set_yscale("log")
    axes[2].set_xlabel("Heterogeneity multiplier"); axes[2].set_ylabel("Final stacked error")
    axes[2].set_title("All seeds and trees (n=100/group)"); clean(axes[2])
    panel_letters(axes); fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.84,wspace=.50)
    save(fig,"Fig2_robustness")


def fig3_numerics_and_certificates(raw, conv, theory):
    fig, axes = plt.subplots(1,3,figsize=(7.2,2.55),gridspec_kw={"width_ratios":[1.2,1,1.05]})
    ax=axes[0]
    finite=conv[conv.finite].copy()
    assert (finite.final_error > 0).all() and (finite.dt > 0).all()
    for seed,g in finite.groupby("seed"):
        g=g.sort_values("dt"); ax.plot(g.dt,g.final_error,marker="o",ms=3,lw=.8,color=BLUE,alpha=.55)
    bad=conv[~conv.finite]
    ax.scatter(bad.dt,np.repeat(finite.final_error.max()*2,len(bad)),marker="x",s=28,color=VERMILION,label="diverged")
    ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xticks([.008,.004,.002,.001],["0.008","0.004","0.002","0.001"])
    ax.set_xlabel("RK4 step size, dt"); ax.set_ylabel("Final stacked error")
    ax.set_title("Step-size convergence (5 matched seeds)"); ax.legend(frameon=False); clean(ax)
    # Availability is not imputed: NaN certificates remain failures.
    avail=raw.assign(ok=np.isfinite(raw.asymptotic_oracle_envelope)).pivot_table(index="topology",columns="heterogeneity",values="ok",aggfunc="mean")
    avail=avail.loc[[f"random_tree_{i}" for i in range(5)],sorted(raw.heterogeneity.unique())]
    ax=axes[1]; im=ax.pcolormesh(np.arange(6)-.5, np.arange(6)-.5, avail.values,
                                 vmin=0,vmax=1,cmap=mpl.colors.ListedColormap(["#D55E00","#009E73"]),shading="flat")
    ax.set_xticks(range(5),[f"{x:g}" for x in avail.columns]); ax.set_yticks(range(5),[f"T{i+1}" for i in range(5)])
    ax.set_xlabel("Heterogeneity multiplier"); ax.set_ylabel("Directed tree"); ax.set_title("Diagnostic certificate availability")
    # Exact availability counts remain in Source Data; repeated in-cell labels
    # are intentionally omitted at final journal width.
    ax=axes[2]
    vals=[theory["coupling_margin"],theory["a"]]
    ax.bar([0,1],vals,color=[GREEN,BLUE],width=.58)
    ax.axhline(0,color=INK,lw=.7)
    ax.set_xticks([0,1],["coupling\nmargin","decay rate\na"])
    ax.set_ylabel("Value"); ax.set_title("Static-bound quantities")
    # The finite-tail value and its status are stated in the caption rather than
    # printed over the data bars.
    clean(ax); panel_letters(axes); fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.80,wspace=.52)
    save(fig,"Fig3_numerics_certificates")


def fig4_representative(raw, rep, tops):
    key="random_tree_0_h1"; t=rep[key+"_t"]; X=rep[key+"_X"]; err=rep[key+"_error"]
    row=raw[(raw.topology=="random_tree_0")&(raw.seed==0)&(raw.heterogeneity==1.)].iloc[0]
    env=row.asymptotic_oracle_envelope
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),gridspec_kw={"height_ratios":[1,1]})
    ax=axes[0,0]
    ax.semilogy(t,err,color=BLUE,lw=1.4,label=r"$\Vert x_{1:7}-x_8\Vert_2$")
    ax.axhline(env,color=VERMILION,ls="--",lw=1.1,label="oracle finite-horizon envelope")
    ax.set_xlabel("Time"); ax.set_ylabel("Stacked error norm"); ax.set_title("Representative error (T1, seed 0, heterogeneity 1)")
    ax.legend(frameon=False); clean(ax)
    ax=axes[0,1]
    colors=[BLUE,GREEN,ORANGE]
    for k,c in enumerate(colors):
        followers=X[:, :7, k]; lo=np.min(followers,axis=1); hi=np.max(followers,axis=1)
        ax.fill_between(t,lo,hi,color=c,alpha=.12)
        ax.plot(t,X[:,7,k],color=c,lw=1.15,label=f"oracle target $x_{k+1}$")
    ax.set_xlabel("Time"); ax.set_ylabel("State"); ax.set_title("Oracle target and follower ranges"); ax.legend(ncol=1,frameon=False); clean(ax)
    ax=axes[1,0]; tail=t>=.8*t[-1]
    ax.plot(t[tail],err[tail],color=BLUE,lw=1.4)
    ax.axhline(env,color=VERMILION,ls="--",lw=1.1)
    ax.set_xlabel("Time (last 20%)"); ax.set_ylabel("Stacked error norm")
    ax.set_title(f"Final 20% of simulation; maximum = {err[tail].max():.2e}")
    clean(ax)
    ax=axes[1,1]; L=tops["random_tree_0"]
    pos=np.array([[.16,.70],[.32,.90],[.43,.57],[.61,.88],[.70,.52],[.88,.78],[.90,.30],[.14,.16]])
    for child in range(7):
        for parent in range(8):
            if L[child,parent]<0:
                ax.annotate("",pos[child],pos[parent],arrowprops=dict(arrowstyle="-|>",lw=.8,color=GREY,shrinkA=9,shrinkB=9,connectionstyle="arc3,rad=.06"))
                delta=pos[child]-pos[parent]; normal=np.array([-delta[1],delta[0]])
                normal=normal/(np.linalg.norm(normal)+1e-12)
                mid=(pos[child]+pos[parent])/2+.025*normal
                ax.text(*mid,f"{abs(L[child,parent]):.2g}",fontsize=7.0,color="#555A60",ha="center",va="center",bbox=dict(fc="white",ec="none",pad=.15,alpha=.88))
    ax.scatter(pos[:7,0],pos[:7,1],s=130,c=SKY,edgecolor="white",lw=.8,zorder=3)
    ax.scatter(pos[7,0],pos[7,1],s=150,c=VERMILION,edgecolor="white",lw=.8,zorder=3)
    for i,(x,y) in enumerate(pos): ax.text(x,y,str(i+1),ha="center",va="center",fontsize=7,color="white",weight="bold",zorder=4)
    ax.text(pos[7,0],pos[7,1]-.12,"oracle target",ha="center",va="top",fontsize=7,color=VERMILION,weight="bold")
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off"); ax.set_title("Exact directed tree T1 (parent → child)")
    panel_letters(axes); fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.90,hspace=.54,wspace=.34)
    save(fig,"Fig4_oracle_diagnostic")


def fig6_ablation(raw, meta):
    variants = meta["variants"]
    labels = {"oracle_static":"Oracle\nstatic", "distributed_full":"Distributed\nfull",
              "state_only_known_model":"State only\nknown model", "no_model_local":"No-model\nlocal",
              "distributed_uncertified":"Distributed\nuncertified"}
    colors = dict(zip(variants,[GREY,BLUE,GREEN,VERMILION,ORANGE]))
    hs = meta["heterogeneity_scales"]
    fig, axes = plt.subplots(1,3,figsize=(7.2,3.05),gridspec_kw={"width_ratios":[1.25,1,1]})
    ax=axes[0]
    for v in variants:
        med=raw[raw.variant==v].groupby("heterogeneity").tail20_max_tracking_error.median().reindex(hs)
        ax.plot(hs,med,marker="o",ms=3.3,lw=1.2,color=colors[v],label=labels[v].replace("\n"," "))
    ax.axhline(meta["tracking_threshold"],color=INK,ls="--",lw=.8,label="supervision threshold")
    ax.set(xlabel="Heterogeneity multiplier",ylabel="Median tail tracking max",title="Tracking outcome")
    ax.set_yscale("log"); ax.set_xticks(hs); clean(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax=axes[1]
    g=raw.groupby("variant").agg(err=("tail20_max_tracking_error","median"),effort=("control_l2","median")).loc[variants]
    for v,r in g.iterrows():
        ax.scatter(r.effort,r.err,s=34,color=colors[v],edgecolor="white",lw=.5,zorder=3)
    ax.set(xlabel="Median control effort, L2 norm",ylabel="Median tail tracking max",title="Accuracy–effort trade-off")
    ax.set_yscale("log"); ax.set_xlim(23,42.5); clean(ax)
    ax=axes[2]
    selected=["distributed_full","no_model_local"]
    x=np.arange(len(hs)); width=.34
    for j,v in enumerate(selected):
        med=raw[raw.variant==v].groupby("heterogeneity").missed_violation_rate.median().reindex(hs)
        ax.bar(x+(j-.5)*width,med,width,color=colors[v],label=labels[v].replace("\n"," "))
    ax.set_xticks(x,[f"{h:g}" for h in hs]); ax.set_ylim(0,1)
    ax.set(xlabel="Heterogeneity multiplier",ylabel="Median missed-violation rate",title="Supervision failures")
    ax.legend(frameon=False,fontsize=6); clean(ax)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5,.985),
               frameon=False, fontsize=6.1, ncol=3, columnspacing=1.5,
               handlelength=2.0)
    panel_letters(axes); fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.70,wspace=.48)
    save(fig,"Fig6_ablation")


def fig7_extension(raw, meta):
    models=meta["models"]; scenarios=meta["scenarios"]; sizes=meta["sizes"]
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),gridspec_kw={"width_ratios":[1,1.15,1.05]})
    # Failure counts: all predeclared runs retained.
    fail=(raw.assign(failed=(~raw.finite)|raw.domain_exit).groupby(["model","scenario"]).failed.sum().unstack().reindex(index=models,columns=scenarios))
    ax=axes[0]; ax.pcolormesh(np.arange(5)-.5, np.arange(3)-.5, fail.values,
                              cmap=mpl.colors.ListedColormap(["#E8F3EE",VERMILION]),vmin=0,vmax=10,shading="flat")
    ax.set_xticks(range(4),[s.capitalize() for s in scenarios],rotation=30,ha="right");ax.set_yticks(range(2),[m.capitalize() for m in models])
    for i in range(2):
        for j in range(4): ax.text(j,i,f"{int(fail.iloc[i,j])}/30",ha="center",va="center",fontsize=7,weight="bold",color="white" if fail.iloc[i,j] else INK)
    ax.set_title("Failures across sizes and seeds")
    # Chua observer errors: delay divergence is the negative result.
    ax=axes[1]; x=np.arange(3); width=.18
    for j,s in enumerate(scenarios):
        vals=raw[(raw.model=="chua")&(raw.scenario==s)].groupby("n").final_observer.median().reindex(sizes)
        ax.bar(x+(j-1.5)*width,vals,width,color=[BLUE,SKY,VERMILION,ORANGE][j],label=s.capitalize())
    ax.set_yscale("log"); ax.set_xticks(x,[str(n) for n in sizes]); ax.set(xlabel="Network size, N",ylabel="Median final observer error",title="Chua observer under impairments")
    ax.legend(frameon=False,ncol=2,fontsize=6); clean(ax)
    # Scaling uses clean runs only and normalises the stacked norm by sqrt(N).
    ax=axes[2]
    for m,c,marker in [("chua",BLUE,"o"),("lorenz",VERMILION,"s")]:
        d=raw[(raw.model==m)&(raw.scenario=="clean")]
        med=d.groupby("n").tail20_max_tracking_per_sqrt_n.median().reindex(sizes)
        lo=d.groupby("n").tail20_max_tracking_per_sqrt_n.quantile(.25).reindex(sizes)
        hi=d.groupby("n").tail20_max_tracking_per_sqrt_n.quantile(.75).reindex(sizes)
        ax.plot(sizes,med,color=c,marker=marker,lw=1.3,label=m.capitalize())
        ax.fill_between(sizes,lo,hi,color=c,alpha=.14)
    ax.set_yscale("log");ax.set_xticks(sizes);ax.set(xlabel="Network size, N",ylabel="Tail tracking max / N^0.5",title="Per-agent size scaling")
    ax.legend(frameon=False);clean(ax)
    ax.text(.98,-.24,"Lorenz: empirical extrapolation only",transform=ax.transAxes,
            ha="right",va="top",fontsize=6,color=VERMILION,style="italic")
    panel_letters(axes);fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.80,wspace=.48)
    save(fig,"Fig7_extension")


def fig8_certified(raw, meta, series, dt):
    tops=["chain","star","branch","cyclic"]; levels=meta["preregistered"]["levels"]
    tcolors=dict(zip(tops,[BLUE,GREEN,ORANGE,PURPLE])); markers=dict(zip(levels,["o","s","^"]))
    fig=plt.figure(figsize=(7.2,4.65))
    gs=fig.add_gridspec(2,3,height_ratios=[.84,1],hspace=.58,wspace=.48)
    ax_chain=fig.add_subplot(gs[0,:2]); ax_scatter=fig.add_subplot(gs[0,2])
    ax_heat=fig.add_subplot(gs[1,0]); ax_traj=fig.add_subplot(gs[1,1]); ax_dt=fig.add_subplot(gs[1,2])
    axes=[ax_chain,ax_scatter,ax_heat,ax_traj,ax_dt]
    # Analytic chain, illustrated by one exact metadata entry.
    c=meta["certificates"]["star"]["0.15"]; R=meta["preregistered"]["R"]
    nodes=[(0.02,"Invariant ball",f"R = {R:.3f}"),(0.22,"Mismatch",f"max |Wᵢ| = {max(c['W']):.3f}"),
           (0.43,"Weighted input",f"Ω = {c['omega']:.3f}"),(0.64,"Dissipation",f"rate = {c['rate']:.3f}"),
           (0.84,"Certificate",f"Δ = {c['delta']:.4f}")]
    for i,(x,title,value) in enumerate(nodes):
        color=[BLUE,SKY,GREEN,ORANGE,VERMILION][i]
        ax_chain.add_patch(FancyBboxPatch((x,.31),.15,.42,boxstyle="round,pad=.012,rounding_size=.02",facecolor=LIGHT,edgecolor=color,lw=1.2))
        ax_chain.text(x+.075,.60,title,ha="center",weight="bold",fontsize=7,color=INK)
        ax_chain.text(x+.075,.43,value,ha="center",fontsize=7,color=color)
        if i<4: ax_chain.annotate("",(nodes[i+1][0]-.008,.52),(x+.16,.52),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=1))
    ax_chain.set(xlim=(0,1),ylim=(0,1));ax_chain.axis("off");ax_chain.set_title("Analytic certificate chain (star topology, heterogeneity 0.15)")
    ax_chain.text(.5,.08,"All terms are analytic; no simulated maximum enters Δ.",ha="center",fontsize=6.5,style="italic",color="#555A60")
    # All 120 certificate-versus-observation pairs.
    for top in tops:
        for lev in levels:
            d=raw[(raw.topology==top)&np.isclose(raw.heterogeneity,lev)]
            ax_scatter.scatter(d.delta,d.tail20_max_error,s=17,color=tcolors[top],marker=markers[lev],alpha=.7,edgecolor="white",lw=.25)
    lim=[raw[["delta","tail20_max_error"]].min().min()*.8,raw[["delta","tail20_max_error"]].max().max()*1.25]
    ax_scatter.plot(lim,lim,"--",color=INK,lw=.8,label="equality")
    ax_scatter.set(xscale="log",yscale="log",xlim=lim,ylim=lim,xlabel="Analytic Δ",ylabel="Observed tail max",title="All 120 registered runs");clean(ax_scatter);ax_scatter.legend(frameon=False)
    # Median tail/certificate ratio, deliberately showing conservatism.
    ratio=(raw.assign(ratio=raw.tail20_max_error/raw.delta).pivot_table(index="topology",columns="heterogeneity",values="ratio",aggfunc="median").loc[tops,levels])
    im=ax_heat.pcolormesh(np.arange(4)-.5, np.arange(5)-.5, ratio.values,
                          vmin=0,vmax=1,cmap="viridis",shading="flat")
    ax_heat.set_xticks(range(3),[f"{x:g}" for x in levels]);ax_heat.set_yticks(range(4),[x.capitalize() for x in tops])
    for i in range(4):
        for j in range(3): ax_heat.text(j,i,f"{ratio.iloc[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if ratio.iloc[i,j]<.5 else INK)
    ax_heat.set(xlabel="Heterogeneity",title="Median tail max / Δ")
    # Exact saved representative: columns t, Euclidean error, comparison ratio, node ratio, target radius, observer error.
    z=series["star_0.15_seed0"]; delta=float(raw[(raw.topology=="star")&np.isclose(raw.heterogeneity,.15)&(raw.seed==0)].delta.iloc[0])
    l1=ax_traj.semilogy(z[:,0],z[:,1],color=BLUE,lw=1.2,label="stacked error")
    l2=ax_traj.axhline(delta,color=GREEN,lw=.9,ls=":",label=r"asymptotic $\Delta$")
    ax_traj.set(xlabel="Time",ylabel="Error norm",title="Trajectory and all-time coverage");clean(ax_traj)
    ax_ratio=ax_traj.twinx(); l3=ax_ratio.plot(z[:,0],z[:,2],color=VERMILION,lw=1.1,ls="--",label="comparison ratio")
    ax_ratio.axhline(1,color=INK,lw=.6,ls=":");ax_ratio.set_ylim(0,1.08);ax_ratio.set_ylabel("")
    ax_ratio.tick_params(axis="y",colors=VERMILION,labelsize=6);ax_ratio.spines["top"].set_visible(False)
    ax_traj.legend(l1+[l2]+l3,[x.get_label() for x in l1+[l2]+l3],frameon=False,fontsize=6.0,loc="upper right")
    # Matched endpoint audit against dt=.001.
    base=dt[np.isclose(dt.dt,.001)].set_index(["topology","seed"]).final_error
    for step,col in [(.004,ORANGE),(.002,BLUE),(.001,GREEN)]:
        d=dt[np.isclose(dt.dt,step)].set_index(["topology","seed"]).final_error
        rel=((d-base).abs()/base).values
        x=np.full(len(rel),step); ax_dt.scatter(x,rel+1e-16,s=18,color=col,alpha=.7,edgecolor="white",lw=.3)
    ax_dt.set(xscale="log",yscale="log",xlabel="RK4 step size, dt",ylabel="Relative endpoint difference",title="Matched dt audit")
    ax_dt.set_xticks([.004,.002,.001],[".004",".002",".001"]);ax_dt.invert_xaxis();clean(ax_dt)
    for letter,ax in zip("abcde",axes): ax.text(-.12,1.17,letter,transform=ax.transAxes,weight="bold",fontsize=10,va="top")
    fig.text(.5,.015,"Purpose-built dissipative supplement — not a rescue of the archived Chua certificates.",ha="center",color=VERMILION,fontsize=7.5,weight="bold")
    fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.91)
    save(fig,"Fig8_certified_benchmark")


def main():
    required=[RESULTS/"raw_runs.csv",RESULTS/"dt_convergence.csv",RESULTS/"raw_representative_trajectories.npz",RESULTS/"topologies.npz",THEORY]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing source data: "+", ".join(missing))
    raw=pd.read_csv(required[0]); conv=pd.read_csv(required[1])
    rep=np.load(required[2]); tops=np.load(required[3]); theory=json.loads(THEORY.read_text())
    assert len(raw)==500 and raw.finite.all()
    assert raw.groupby(["topology","heterogeneity"]).size().eq(20).all()
    assert len(conv)==20 and (~conv[conv.dt==.008].finite).all()
    fig1_workflow(); fig2_robustness(raw); fig3_numerics_and_certificates(raw,conv,theory); fig4_representative(raw,rep,tops)
    abr=pd.read_csv(ABLATION_RESULTS/"raw_runs.csv"); abm=json.loads((ABLATION_RESULTS/"metadata.json").read_text())
    exr=pd.read_csv(EXTENSION_RESULTS/"raw_runs.csv"); exm=json.loads((EXTENSION_RESULTS/"metadata.json").read_text())
    assert len(abr)==300 and abr.groupby(["variant","heterogeneity"]).size().eq(20).all()
    assert len(exr)==240 and exr.groupby(["model","n","scenario"]).size().eq(10).all()
    assert int(((~exr.finite)|exr.domain_exit).sum())==30
    fig6_ablation(abr,abm); fig7_extension(exr,exm)
    cbr=pd.read_csv(CERTIFIED_RESULTS/"raw_runs.csv"); cbm=json.loads((CERTIFIED_RESULTS/"metadata.json").read_text())
    cbz=np.load(CERTIFIED_RESULTS/"representative_timeseries.npz"); cbdt=pd.read_csv(CERTIFIED_RESULTS/"dt_audit.csv")
    assert len(cbr)==120 and cbr.finite.all() and len(cbz.files)==120
    assert (cbr.tail20_max_error<=cbr.delta).all() and cbr.max_certificate_ratio.le(1+1e-12).all()
    assert len(cbdt)==24 and cbdt.finite.all()
    fig8_certified(cbr,cbm,cbz,cbdt)
    print("Built 7 figures as editable SVG/PDF and 600-dpi PNG/TIFF from validated sources.")

if __name__ == "__main__": main()
