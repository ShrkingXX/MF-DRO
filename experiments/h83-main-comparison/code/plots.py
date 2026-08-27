"""H83 figures. Reuses analyse.py's loader so the plotted SR is the same
trace-recomputed, cross-checked quantity the tables report -- not a second
implementation that could drift from it."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import load, grid, BENCH, METHODS, SEEDS
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":0.25,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.bbox":"tight"})
C={"MF-DRO":"#C0392B","SF-DRO":"#E67E22","MF-MES":"#2471A3",
   "MF-MI-Greedy":"#27AE60","MF-GP-UCB":"#7D3C98"}
G=np.linspace(0,200,201)
D,missing,failures=load()
if failures: print("SR CROSS-CHECK FAILED:", failures[:5]); sys.exit(1)
print(f"loaded {len(D)} runs; {len(missing)} missing")

def units(b):
    o=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    return (1.0,"absolute simple regret") if abs(o)<1e-9 else (100.0/abs(o),"relative regret (%)")

# ---------- Figure 1: mean SR vs cost ----------
fig,axes=plt.subplots(1,4,figsize=(15.5,3.4))
for ax,b in zip(axes,BENCH):
    k,lab=units(b)
    for m in METHODS:
        curves=[grid(D[(b,m,s)]["cost"],D[(b,m,s)]["sr"],G)*k for s in SEEDS if (b,m,s) in D]
        if not curves: continue
        A=np.vstack(curves); mu=np.nanmean(A,0); se=np.nanstd(A,0,ddof=1)/np.sqrt(A.shape[0])
        ax.plot(G,mu,color=C[m],lw=1.6,label=m)
        ax.fill_between(G,mu-se,mu+se,color=C[m],alpha=0.15,lw=0)
    ax.set_yscale("log"); ax.set_title(b); ax.set_xlabel("cost (post-init)")
    ax.set_ylabel(lab)          # per panel: Ackley is ABSOLUTE, others relative %
fig.legend(handles=[Line2D([],[],color=C[m],lw=2,label=m) for m in METHODS],
           loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.13))
fig.subplots_adjust(wspace=0.38)
fig.suptitle("H83 mean simple regret vs cost (5 seeds, shaded = ±1 s.e.)",y=1.06,fontsize=11)
fig.savefig(os.path.join(OUT,"fig1_mean_sr_vs_cost.png")); plt.close(fig)

# ---------- Figure 2: seed-by-seed ----------
fig,axes=plt.subplots(4,5,figsize=(16,10.5),sharex=True)
for i,b in enumerate(BENCH):
    k,lab=units(b)
    for j,s in enumerate(SEEDS):
        ax=axes[i,j]
        for m in METHODS:
            if (b,m,s) not in D: continue
            r=D[(b,m,s)]; ax.plot(r["cost"],r["sr"]*k,color=C[m],lw=1.3)
        ax.set_yscale("log")
        if i==0: ax.set_title(f"seed {s}")
        if j==0: ax.set_ylabel(f"{b}\n{lab}",fontsize=7.5)
        if i==len(BENCH)-1: ax.set_xlabel("cost (post-init)")
fig.legend(handles=[Line2D([],[],color=C[m],lw=2,label=m) for m in METHODS],
           loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("H83 simple regret vs cost, per seed",y=1.0,fontsize=12)
fig.savefig(os.path.join(OUT,"fig2_seed_by_seed_sr_vs_cost.png")); plt.close(fig)

# ---------- Figure 3: MF-DRO query value + fidelity per iteration ----------
fig,axes=plt.subplots(4,5,figsize=(16,10))
for i,b in enumerate(BENCH):
    opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    for j,s in enumerate(SEEDS):
        ax=axes[i,j]
        p=os.path.join(OUT,f"{b}__MF-DRO__seed{s}.json")
        if not os.path.exists(p): ax.axis("off"); continue
        q=[e for e in json.load(open(p))["queries"] if not e.get("is_init")]
        it=np.arange(len(q)); y=np.array([e["y"] for e in q]); fid=np.array([e["fid"] for e in q])
        # y is stored so that LARGER IS BETTER: the regret convention is
        # regret = -max(y_HF) - known_optimal_value, so the best attainable y
        # equals -known_optimal_value. Plot y as-is against a line at -opt.
        # Negating y here (an earlier version did) silently made
        # maximum.accumulate track the WORST query instead of the best.
        yr=y
        ax.scatter(it[fid==0],yr[fid==0],s=9,c="#5DADE2",marker="x",lw=0.8,label="LF")
        ax.scatter(it[fid==1],yr[fid==1],s=11,c="#C0392B",marker="o",
                   edgecolors="none",label="HF")
        run=np.maximum.accumulate(np.where(fid==1,yr,-np.inf))
        ax.plot(it,np.where(np.isfinite(run),run,np.nan),color="k",lw=1.0,alpha=0.7)
        ax.axhline(-opt,color="k",ls="--",lw=0.8,alpha=0.6)
        lfp=100.0*np.mean(fid==0)
        ax.text(0.97,0.05,f"LF {lfp:.0f}%",transform=ax.transAxes,ha="right",fontsize=7.5,
                bbox=dict(fc="w",ec="none",alpha=0.7,pad=1.5))
        if i==0: ax.set_title(f"seed {s}")
        if j==0: ax.set_ylabel(f"{b}\nqueried value",fontsize=8)
        if i==len(BENCH)-1: ax.set_xlabel("optimization iteration")
fig.legend(handles=[Line2D([],[],color="#C0392B",marker="o",ls="",label="HF query"),
                    Line2D([],[],color="#5DADE2",marker="x",ls="",label="LF query"),
                    Line2D([],[],color="k",lw=1,label="running best HF"),
                    Line2D([],[],color="k",ls="--",lw=1,label="true optimum")],
           loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("H83 MF-DRO: queried value and fidelity per iteration",y=1.0,fontsize=12)
fig.savefig(os.path.join(OUT,"fig3_mfdro_query_fidelity.png")); plt.close(fig)
print("wrote fig1_mean_sr_vs_cost.png, fig2_seed_by_seed_sr_vs_cost.png, fig3_mfdro_query_fidelity.png")
