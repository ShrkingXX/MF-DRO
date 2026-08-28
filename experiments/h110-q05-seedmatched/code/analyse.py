"""H110: does the BEST setting close more of the gap, seed-matched at n=10?
Committed with 0 results on disk. EXPLICIT paths; no globbing across experiments.

READ h109 FIRST. These runs execute the patched src/ that h109 is testing. If
h109's P1 failed, every number below is contaminated and must be discarded."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
DEV=(42,43,44,45,46); NEW=(52,53,54,55,56)
E=lambda d,n,s: f"{REPO}/experiments/{d}/results/{B}__{n}__seed{s}.json"
P_Q05={**{s:E("h107-q05-confirm","ROI-Q05",s) for s in DEV},
       **{s:E("h110-q05-seedmatched","ROI-Q05",s) for s in NEW}}
P_Q10={**{s:E("h84-roi-strategy","ROI-Q10",s) for s in DEV},
       **{s:E("h106-roi-closes-the-gap","ROI-Q10",s) for s in NEW}}
P_CTL={**{s:E("h83-main-comparison","MF-DRO",s) for s in DEV},
       **{s:E("h89-hffloor-confirm","CONTROL",s) for s in NEW}}
P_MES={**{s:E("h83-main-comparison","MF-MES",s) for s in DEV},
       **{s:E("h92-mfmes-borehole","MF-MES",s) for s in NEW}}
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT

if __name__=="__main__":
    seeds=[s for s in DEV+NEW if all(os.path.exists(d[s]) for d in (P_Q05,P_Q10,P_CTL,P_MES))]
    missing=[s for s in DEV+NEW if s not in seeds]
    print("H110 -- does q=0.05 close more of the gap than q=0.10, seed-matched at n=10?\n")
    print("  *** READ h109 FIRST. If its P1 failed, everything below is contaminated. ***\n")
    print(f"  usable seeds: {len(seeds)}/10   missing: {missing if missing else 'none'}")
    if missing:
        print("\n  INCOMPLETE -- every bar refuses."); sys.exit()
    q05=np.array([at200(P_Q05[s]) for s in seeds]); q10=np.array([at200(P_Q10[s]) for s in seeds])
    ctl=np.array([at200(P_CTL[s]) for s in seeds]); mes=np.array([at200(P_MES[s]) for s in seeds])
    print(f"\n  {'seed':>5}{'Q05':>8}{'Q10':>8}{'no-ROI':>9}{'MF-MES':>9}{'Q05-noROI':>11}{'Q05-Q10':>9}")
    for i,s in enumerate(seeds):
        print(f"  {s:>5}{q05[i]:>8.2f}{q10[i]:>8.2f}{ctl[i]:>9.2f}{mes[i]:>9.2f}"
              f"{q05[i]-ctl[i]:>+11.2f}{q05[i]-q10[i]:>+9.2f}")
    def st(d): 
        m,sd=d.mean(),d.std(ddof=1); return m,sd,abs(m)/sd,int((d<0).sum())
    print("\n  REGISTERED BARS")
    m,sd,r,w=st(q05-ctl)
    r1=(m<=-3.0) and (r>=0.5)
    print(f"    R1 PRIMARY  Q05 vs no-ROI: mean {m:+.2f} sd {sd:.2f} ratio {r:.2f} better {w}/10")
    print(f"                bar = mean <= -3.0 AND ratio >= 0.50  ->  {'MET' if r1 else 'FAILED'}")
    base=ctl.mean()-mes.mean(); g05=q05.mean()-mes.mean(); g10=q10.mean()-mes.mean()
    c05=100*(base-g05)/base; c10=100*(base-g10)/base
    print(f"    R2  gap closed: q=0.10 {c10:.0f}%   q=0.05 {c05:.0f}%  ->  "
          f"{'MET' if c05>c10 else 'FAILED'}")
    print(f"        (registered WEAKLY -- the same setting's closure moved 45%->57% between seed sets)")
    print(f"    R3 NEGATIVE Q05 mean {q05.mean():.2f} vs MF-MES {mes.mean():.2f}  ->  "
          f"{'MET (still behind)' if q05.mean()>mes.mean() else '*** REFUTED -- re-verify ***'}")
    dm=q05-mes; lose=int((dm>0).sum())
    print(f"    R4  paired vs MF-MES: Q05 better {int((dm<0).sum())}/10, loses {lose}/10, "
          f"median {np.median(dm):+.2f}  ->  {'MET' if lose>=6 else 'FAILED'}")
    print(f"        (the mean-based percentage above must never be quoted without this line)")
    d=q05-q10; m2,sd2,r2,w2=st(d)
    print(f"\n  Q05 vs Q10 head-to-head: mean {m2:+.2f} sd {sd2:.2f} ratio {r2:.2f} better {w2}/10"
          f"  {'SEPARABLE' if r2>1 else 'not separable'}")
    print(f"  prior: h97 -1.52 (4/5), h107 -1.57 (4/5)")
