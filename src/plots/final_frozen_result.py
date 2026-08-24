"""Final frozen-evaluation figure: per-seed traces + means, both rewards."""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
OUT=os.path.join(REPO,"to_human","figs"); os.makedirs(OUT,exist_ok=True)
H17=f"{REPO}/experiments/h17-joint-mes-frozen-eval/results"
H1=f"{REPO}/experiments/h1-leak-fix-validation/results"
S=list(range(42,52))
def fin(p):
    d=json.load(open(p)); c=d.get("hf_regret_curve") or d.get("regret_curve"); return float(c[-1])
arms=[("MF-DRO\njoint MES",[fin(f"{H17}/MF-DRO__seed{s}.json") for s in S],"#1f7a5a"),
      ("MF-DRO\nimprovement",[fin(f"{H1}/MF-DRO__seed{s}.json") for s in S],"#4a90d9"),
      ("MF-MI-Greedy",[fin(f"{H1}/MF-MI-Greedy__seed{s}.json") for s in S],"#c9862c"),
      ("MF-GP-UCB",[fin(f"{H1}/MF-GP-UCB__seed{s}.json") for s in S],"#b23223")]
fig,ax=plt.subplots(figsize=(8,4.6))
for i,(lab,v,c) in enumerate(arms):
    v=np.array(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v))
    ax.scatter(np.full(len(v),i)+np.linspace(-.13,.13,len(v)),v,s=26,color=c,
               alpha=.55,zorder=3,edgecolor="none")
    ax.errorbar(i,m,yerr=se,fmt="_",color=c,markersize=34,capsize=7,lw=2.4,zorder=4)
    ax.text(i,-0.09,f"{m:.3f}\n±{se:.3f}",ha="center",fontsize=8.5,color=c,fontweight="bold")
bl=min(np.mean(arms[2][1])-np.std(arms[2][1],ddof=1)/np.sqrt(10),
       np.mean(arms[3][1])-np.std(arms[3][1],ddof=1)/np.sqrt(10))
ax.axhline(bl,ls="--",c="k",lw=1.1)
ax.text(3.42,bl+0.03,f"success bar\n(best baseline mean−SE = {bl:.3f})",
        ha="right",fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels([a[0] for a in arms],fontsize=9)
ax.set_ylabel("final simple regret at matched cost ≈ 200")
ax.set_ylim(-0.15,2.5); ax.set_xlim(-.5,3.5)
ax.set_title("Frozen evaluation · 10 seeds · both MF-DRO arms FAIL the success test",
             fontsize=11,loc="left")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); p=os.path.join(OUT,"final_frozen_result.png")
fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig); print(p)
