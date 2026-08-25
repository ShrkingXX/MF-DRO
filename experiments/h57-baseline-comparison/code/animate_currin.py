"""Animate a Currin 2D run: HF landscape + the real queried points as they
arrive. Reads any H57 trace (final result OR live checkpoint) and writes a GIF.

usage: animate_currin.py <trace.json> [out.gif]
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from benchmarks import get_benchmark

src=sys.argv[1]
out=sys.argv[2] if len(sys.argv)>2 else src.replace(".json",".gif")
d=json.load(open(src)); Q=d["queries"]
meta=d.get("_meta") or {k:d.get(k) for k in ("bench","method","seed")}
bench=meta.get("bench","Currin_2D"); method=meta.get("method","?"); seed=meta.get("seed","?")
assert bench=="Currin_2D", f"this animator is 2D-only, got {bench}"

hf=get_benchmark("Currin_2D_HF"); lf=get_benchmark("Currin_2D_LF")
lo=np.array(hf["domain_min"]); hi=np.array(hf["domain_max"])
f_hf=hf["make_objective"](); f_lf=lf["make_objective"]()
G=160
gx=np.linspace(lo[0],hi[0],G); gy=np.linspace(lo[1],hi[1],G)
GX,GY=np.meshgrid(gx,gy)
P=torch.tensor(np.stack([GX.ravel(),GY.ravel()],1),dtype=torch.float64)
with torch.no_grad():
    ZH=f_hf(P).reshape(-1).numpy().reshape(G,G)
    ZL=f_lf(P).reshape(-1).numpy().reshape(G,G)
star=np.unravel_index(np.argmax(ZH),ZH.shape)
x_star=(GX[star],GY[star])
# The benchmark's own optimum, NOT the grid max: a 160x160 grid underestimates
# the continuous optimum, which made a real query beat it and produced a
# negative "regret" of -0.0008. known_optimal_value is stored negated.
true_opt=-float(hf["known_optimal_value"])

X=np.array([q["x"] for q in Q]); Y=np.array([q["y"] for q in Q])
F=np.array([q["fid"] for q in Q]); IN=np.array([bool(q.get("is_init",False)) for q in Q])
C=np.array([q.get("cost_cum",0.0) for q in Q])
# incumbent = best HF value seen so far (LF observations never set the incumbent)
inc=np.full(len(Q),np.nan); best=-np.inf; bi=-1; binc=[]
for i in range(len(Q)):
    if F[i]==1 and Y[i]>best: best=Y[i]; bi=i
    inc[i]=best if np.isfinite(best) else np.nan; binc.append(bi)
reg=true_opt-inc

fig=plt.figure(figsize=(12.4,5.6))
gs=fig.add_gridspec(2,2,width_ratios=[1.25,1],height_ratios=[1,1],hspace=.42,wspace=.24)
axM=fig.add_subplot(gs[:,0]); axL=fig.add_subplot(gs[0,1]); axR=fig.add_subplot(gs[1,1])
for ax,Z,t in ((axM,ZH,"HF"),(axL,ZL,"LF")):
    ax.contourf(GX,GY,Z,levels=40,cmap="viridis",alpha=.9)
    ax.contour(GX,GY,Z,levels=12,colors="k",linewidths=.25,alpha=.35)
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1])
axM.plot(*x_star,marker="*",ms=22,mfc="white",mec="k",mew=1.4,ls="none",zorder=6)
axM.set_title(f"{method} · seed {seed} · Currin 2D (HF landscape)",fontsize=11)
axL.set_title("LF landscape",fontsize=9); axL.tick_params(labelsize=7)
axR.set_xlabel("cumulative cost"); axR.set_ylabel("simple regret"); axR.tick_params(labelsize=8)
axR.set_xlim(0,max(C.max(),1)*1.02); axR.grid(alpha=.25)
fin=reg[np.isfinite(reg)]
if fin.size: axR.set_ylim(max(0,fin.min()-.05*(fin.max()-fin.min()+1e-9)),fin.max()*1.05+1e-9)

past_l,=axM.plot([],[],"o",ms=4.5,mfc="none",mec="#d0d0d0",mew=.9,ls="none",zorder=3)
past_h,=axM.plot([],[],"o",ms=6.5,mfc="none",mec="w",mew=1.3,ls="none",zorder=4)
init_m,=axM.plot([],[],"s",ms=4.5,mfc="none",mec="#9a9a9a",mew=.8,ls="none",zorder=2)
cur,=axM.plot([],[],"o",ms=13,mfc="none",mec="#ff2d55",mew=2.6,ls="none",zorder=8)
incm,=axM.plot([],[],"P",ms=12,mfc="#00e5ff",mec="k",mew=1.0,ls="none",zorder=7)
trail,=axM.plot([],[],"-",lw=.9,color="w",alpha=.45,zorder=3)
curve,=axR.plot([],[],"-",lw=1.7,color="#111"); dot,=axR.plot([],[],"o",ms=6,color="#ff2d55")
txt=axM.text(.015,.982,"",transform=axM.transAxes,va="top",ha="left",fontsize=9,color="w",
             bbox=dict(fc="k",alpha=.62,ec="none",pad=3.5))
axM.plot([],[],"s",ms=5,mfc="none",mec="#9a9a9a",ls="none",label="initial design")
axM.plot([],[],"o",ms=5,mfc="none",mec="#d0d0d0",ls="none",label="past LF")
axM.plot([],[],"o",ms=6,mfc="none",mec="w",ls="none",label="past HF")
axM.plot([],[],"o",ms=8,mfc="none",mec="#ff2d55",mew=2,ls="none",label="current query")
axM.plot([],[],"P",ms=8,mfc="#00e5ff",mec="k",ls="none",label="incumbent (best HF)")
axM.plot([],[],"*",ms=12,mfc="white",mec="k",ls="none",label="true optimum")
axM.legend(loc="lower right",fontsize=7,framealpha=.82,ncol=2)

def upd(k):
    sl=slice(0,k+1)
    m_in=IN[sl]; m_lf=(~IN[sl])&(F[sl]==0); m_hf=(~IN[sl])&(F[sl]==1)
    init_m.set_data(X[sl][m_in,0],X[sl][m_in,1])
    past_l.set_data(X[sl][m_lf,0],X[sl][m_lf,1])
    past_h.set_data(X[sl][m_hf,0],X[sl][m_hf,1])
    cur.set_data([X[k,0]],[X[k,1]])
    j=max(0,k-14); trail.set_data(X[j:k+1,0],X[j:k+1,1])
    if binc[k]>=0: incm.set_data([X[binc[k],0]],[X[binc[k],1]])
    curve.set_data(C[sl],reg[sl]); dot.set_data([C[k]],[reg[k]])
    txt.set_text(f"iter {k+1}/{len(Q)}   {'HF' if F[k] else 'LF'}"
                 f"{'  (init)' if IN[k] else ''}\ncost {C[k]:.0f}   "
                 f"regret {reg[k]:.4f}" if np.isfinite(reg[k]) else
                 f"iter {k+1}/{len(Q)}   {'HF' if F[k] else 'LF'}\ncost {C[k]:.0f}")
    return init_m,past_l,past_h,cur,trail,incm,curve,dot,txt

fps=max(4,min(20,len(Q)//10))
FuncAnimation(fig,upd,frames=len(Q),blit=False).save(out,writer=PillowWriter(fps=fps),dpi=96)
print(f"[gif] {out}  frames={len(Q)}  fps={fps}  final_regret={reg[-1]:.4f}  "
      f"HF={int((F==1).sum())} LF={int((F==0).sum())}")
