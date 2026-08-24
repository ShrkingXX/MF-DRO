"""Publication figures (vector PDF) for paper/main.tex."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.rcParams.update({"font.size":9,"axes.linewidth":.8,"pdf.fonttype":42})
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
OUT=os.path.join(REPO,"paper","figs"); os.makedirs(OUT,exist_ok=True)
# Okabe-Ito (colourblind-safe); also distinguishable in greyscale by value
BLUE,ORANGE,GREEN,RED,GREY="#0072B2","#E69F00","#009E73","#D55E00","#666666"

def fig1_attenuation():
    """Figure 1: the mechanism. Signal decays; the decision never moves."""
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(7.0,2.7),gridspec_kw={"width_ratios":[1.45,1]})
    stages=["state $s$","hidden $h$","coeffs $w$"]; vals=[0.2155,0.0745,0.0219]
    ax.bar(range(3),vals,color=[BLUE,ORANGE,RED],width=.6)
    for i,v in enumerate(vals):
        ax.text(i,v+.008,f"{v:.4f}",ha="center",fontsize=8.5)
    for i,lab in [(0,r"$0.346\times$"),(1,r"$0.294\times$")]:
        ax.annotate("",xy=(i+.92,vals[i+1]+.022),xytext=(i+.08,vals[i]+.022),
                    arrowprops=dict(arrowstyle="->",lw=1.1,color=GREY))
        ax.text(i+.5,vals[i]+.033,lab,ha="center",fontsize=8.5,color=GREY)
    ax.set_xticks(range(3)); ax.set_xticklabels(stages)
    ax.set_ylabel("relative spread across\nreal-iteration states")
    ax.set_ylim(0,.27); ax.spines[["top","right"]].set_visible(False)
    ax.set_title(r"(a) $\sim$10$\times$ attenuation, split evenly",fontsize=9.5,loc="left")

    # (b) what a full run's state change buys you
    ax2.axis("off")
    rows=[(r"state change over a full run",r"$L_2 = 1.4968$",BLUE),
          (r"rotation of $w$",r"$2.04^\circ$",ORANGE),
          (r"change in $\|w\|$",r"$2.3\%$",ORANGE),
          (r"score-vector correlation",r"$1.000000$",RED),
          (r"\textbf{decisions changed}",r"$\mathbf{0/12}$",RED)]
    for j,(l,v,c) in enumerate(rows):
        y=.86-j*.19
        ax2.text(0,y,l.replace(r"\textbf{","").replace("}",""),fontsize=8.6,va="center")
        ax2.text(1.0,y,v.replace(r"\mathbf{","").replace("}","").replace("$",""),
                 fontsize=9.2,va="center",ha="right",color=c,
                 fontweight="bold" if j==4 else "normal")
        if j<4: ax2.plot([0,1.0],[y-.095,y-.095],lw=.5,color="#DDD")
    ax2.set_xlim(-.02,1.05); ax2.set_ylim(0,1)
    ax2.set_title("(b) and none of it reaches the decision",fontsize=9.5,loc="left")
    fig.tight_layout(); p=f"{OUT}/fig1_mechanism.pdf"
    fig.savefig(p,bbox_inches="tight"); plt.close(fig); return p

def fig2_interventions():
    rows=[("adaLN conditioning",0),("deny score head GP features",0),
          ("RTG swept in realized band",0),(r"RTG floor $\alpha_{rtg}$",None),
          ("un-normalized RTG",None),(r"real history $T{=}1\to8$",0),
          ("+ DT-style RTG decrement",None),("joint-MES reward (live)",0),
          ("deterministic dynamics",0),("+ stochastic behaviour policy",0),
          (r"MLP score head $[h;cf]$",0)]
    fig,ax=plt.subplots(figsize=(5.4,3.2))
    y=np.arange(len(rows))[::-1]
    for yi,(lab,v) in zip(y,rows):
        if v is None:
            ax.text(.4,yi,"void — manipulation check failed",fontsize=7.8,
                    va="center",color=GREY,style="italic")
        else:
            ax.barh(yi,.45,color=RED,height=.5)
            ax.text(.9,yi,"0%",va="center",fontsize=8,color=RED)
    ax.axvline(30,ls="--",c="k",lw=.9)
    ax.text(30.6,len(rows)-1.4,"pre-registered\nbar (30%)",fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=8)
    ax.set_xlim(0,40); ax.set_xlabel("% of probes on which the proposed argmax moved")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); p=f"{OUT}/fig2_interventions.pdf"
    fig.savefig(p,bbox_inches="tight"); plt.close(fig); return p

def fig3_frozen():
    H17=f"{REPO}/experiments/h17-joint-mes-frozen-eval/results"
    H1=f"{REPO}/experiments/h1-leak-fix-validation/results"
    S=range(42,52)
    def fin(p):
        d=json.load(open(p)); c=d.get("hf_regret_curve") or d.get("regret_curve"); return float(c[-1])
    arms=[("MF-DRO\njoint MES",[fin(f"{H17}/MF-DRO__seed{s}.json") for s in S],GREEN),
          ("MF-DRO\nimprovement",[fin(f"{H1}/MF-DRO__seed{s}.json") for s in S],BLUE),
          ("MF-MI-Greedy",[fin(f"{H1}/MF-MI-Greedy__seed{s}.json") for s in S],ORANGE),
          ("MF-GP-UCB",[fin(f"{H1}/MF-GP-UCB__seed{s}.json") for s in S],RED)]
    fig,ax=plt.subplots(figsize=(5.4,3.0))
    for i,(lab,v,c) in enumerate(arms):
        v=np.array(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v))
        ax.scatter(np.full(len(v),i)+np.linspace(-.12,.12,len(v)),v,s=16,color=c,
                   alpha=.5,zorder=3,edgecolor="none")
        ax.errorbar(i,m,yerr=se,fmt="_",color=c,markersize=26,capsize=5,lw=1.9,zorder=4)
    mig=np.array(arms[2][1]); ucb=np.array(arms[3][1])
    bar=min(mig.mean()-mig.std(ddof=1)/np.sqrt(10),ucb.mean()-ucb.std(ddof=1)/np.sqrt(10))
    ax.axhline(bar,ls="--",c="k",lw=1)
    ax.text(3.45,bar+.05,f"success bar {bar:.3f}",ha="right",fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([a[0] for a in arms],fontsize=8)
    ax.set_ylabel("final simple regret at matched cost")
    ax.set_ylim(0,2.5); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); p=f"{OUT}/fig3_frozen.pdf"
    fig.savefig(p,bbox_inches="tight"); plt.close(fig); return p

if __name__=="__main__":
    for f in (fig1_attenuation,fig2_interventions,fig3_frozen): print(f())
