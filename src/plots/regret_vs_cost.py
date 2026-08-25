"""Regret-vs-cost trajectories: anytime performance of the post-fix variants."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
H1=f"{REPO}/experiments/h1-leak-fix-validation/results"
H17=f"{REPO}/experiments/h17-joint-mes-frozen-eval/results"
S=range(42,52); BUD=200.0; G=np.linspace(0,BUD,401)
BLUE,ORANGE,GREEN,RED="#0072B2","#E69F00","#009E73","#D55E00"
def curve(p):
    d=json.load(open(p))
    r=np.array(d.get("hf_regret_curve") or d.get("regret_curve"),float)
    c=np.array(d["cost_curve"],float); n=min(len(r),len(c)); r,c=r[:n],c[:n]
    return np.array([r[max(np.searchsorted(c,g,side="right")-1,0)] for g in G])
arms=[("MF-DRO / joint MES",[f"{H17}/MF-DRO__seed{s}.json" for s in S],GREEN),
      ("MF-DRO / improvement",[f"{H1}/MF-DRO__seed{s}.json" for s in S],BLUE),
      ("MF-MI-Greedy",[f"{H1}/MF-MI-Greedy__seed{s}.json" for s in S],ORANGE),
      ("MF-GP-UCB",[f"{H1}/MF-GP-UCB__seed{s}.json" for s in S],RED)]
fig,ax=plt.subplots(figsize=(6.2,3.8))
for lab,fs,c in arms:
    C=np.stack([curve(f) for f in fs]); m=C.mean(0); se=C.std(0,ddof=1)/np.sqrt(len(fs))
    ax.plot(G,m,color=c,lw=1.9,label=lab)
    ax.fill_between(G,m-se,m+se,color=c,alpha=.16,lw=0)
ax.set_xlabel("post-initialisation cost"); ax.set_ylabel("simple regret (mean ± SE, 10 seeds)")
ax.set_xlim(0,BUD); ax.set_ylim(0,2.1)
ax.set_title("Anytime performance: regret vs cost",fontsize=10.5,loc="left")
ax.legend(fontsize=8,frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(f"{REPO}/paper/figs/fig4_regret_vs_cost.{ext}" if ext=="pdf"
                else f"{REPO}/to_human/figs/regret_vs_cost.png", bbox_inches="tight", dpi=150)
print("wrote paper/figs/fig4_regret_vs_cost.pdf and to_human/figs/regret_vs_cost.png")
