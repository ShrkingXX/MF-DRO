"""Mean SR-vs-cost: MF-DRO (with the calibrated ROI) against the h83 baselines.

"MF-DRO" here IS the calibrated-ROI configuration (use_roi=True,
roi_beta_mode='quantile', roi_target_accept=0.10) -- that is the method. The
older no-ROI runs are h83's and are not drawn; see fig1 in h83 for those.

Uses h83's own sr_curve/grid so every method sits on one metric: SR
grid-interpolated onto a common cost axis. The MF-DRO+ROI arm is assembled from
h84 (Hartmann, Borehole) and h86 (Currin, Ackley) -- the same configuration in
both.
"""
import sys, os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark

G=np.linspace(0,200,201); SEEDS=(42,43,44,45,46)
H83=os.path.join(REPO,'experiments','h83-main-comparison','results')
H84=os.path.join(REPO,'experiments','h84-roi-strategy','results')
H86=os.path.join(REPO,'experiments','h86-roi-full','results')
ROI={"Hartmann_6D":H84,"Borehole_8D":H84,"Currin_2D":H86,"Ackley_10D":H86}
BENCH=("Currin_2D","Hartmann_6D","Borehole_8D","Ackley_10D")
BASE=("SF-DRO","MF-MES","MF-MI-Greedy","MF-GP-UCB")
C={"MF-MES":"#2471A3","SF-DRO":"#E67E22","MF-MI-Greedy":"#27AE60","MF-GP-UCB":"#7D3C98"}
DRO_OLD="#E8836F"; DRO_NEW="#8E1B0F"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":0.25,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.bbox":"tight"})

def curves(b,paths):
    opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    k=1.0 if abs(opt)<1e-9 else 100.0/abs(opt)
    out=[]
    for p in paths:
        if not os.path.exists(p): continue
        c,sr=sr_curve(json.load(open(p)),opt)
        out.append(grid(c,sr,G)*k)
    return np.vstack(out) if out else None

def band(ax,A,color,lw,ls="-",z=2,alpha_f=0.13):
    mu=np.nanmean(A,0); se=np.nanstd(A,0,ddof=1)/np.sqrt(A.shape[0])
    ax.plot(G,mu,color=color,lw=lw,ls=ls,zorder=z)
    ax.fill_between(G,mu-se,mu+se,color=color,alpha=alpha_f,lw=0,zorder=z-1)

fig,axes=plt.subplots(1,4,figsize=(15.5,3.6))
for ax,b in zip(axes,BENCH):
    opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    lab="absolute simple regret" if abs(opt)<1e-9 else "relative regret (%)"
    for m in BASE:
        A=curves(b,[f'{H83}/{b}__{m}__seed{s}.json' for s in SEEDS])
        if A is not None: band(ax,A,C[m],1.4,z=2,alpha_f=0.10)
    B=curves(b,[f'{ROI[b]}/{b}__ROI-Q10__seed{s}.json' for s in SEEDS])
    n_roi=0 if B is None else B.shape[0]
    if B is not None: band(ax,B,DRO_NEW,2.6,z=5,alpha_f=0.16)
    ax.set_yscale("log"); ax.set_xlabel("cost (post-init)"); ax.set_ylabel(lab)
    t=b if n_roi==5 else (f"{b}  (MF-DRO n={n_roi}/5)" if n_roi
                          else f"{b}  — MF-DRO STILL RUNNING")
    ax.set_title(t,fontsize=10)
fig.subplots_adjust(wspace=0.38)
H=[Line2D([],[],color=DRO_NEW,lw=2.6,label="MF-DRO")]+ \
  [Line2D([],[],color=C[m],lw=1.4,label=m) for m in BASE]
fig.legend(handles=H,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.13))
fig.suptitle("Mean simple regret vs cost (5 seeds, shaded ±1 s.e.)",y=1.06,fontsize=11)
out=os.path.join(H86,"fig_roi_mean_sr_vs_cost.png")
fig.savefig(out); print("wrote",out)
