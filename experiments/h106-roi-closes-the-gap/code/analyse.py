"""H106: does the ROI reduce the ONE gap that is real, seed-matched at n=10?
Committed with 1 of 5 runs on disk. Bars per protocol; EXPLICIT paths only."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
DEV=(42,43,44,45,46); NEW=(52,53,54,55,56)
# EXPLICIT source per cell -- cross-experiment globbing is banned.
P_ROI={**{s:f"{REPO}/experiments/h84-roi-strategy/results/{B}__ROI-Q10__seed{s}.json" for s in DEV},
       **{s:f"{REPO}/experiments/h106-roi-closes-the-gap/results/{B}__ROI-Q10__seed{s}.json" for s in NEW}}
P_CTL={**{s:f"{REPO}/experiments/h83-main-comparison/results/{B}__MF-DRO__seed{s}.json" for s in DEV},
       **{s:f"{REPO}/experiments/h89-hffloor-confirm/results/{B}__CONTROL__seed{s}.json" for s in NEW}}
P_MES={**{s:f"{REPO}/experiments/h83-main-comparison/results/{B}__MF-MES__seed{s}.json" for s in DEV},
       **{s:f"{REPO}/experiments/h92-mfmes-borehole/results/{B}__MF-MES__seed{s}.json" for s in NEW}}
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
def have(s): return all(os.path.exists(d[s]) for d in (P_ROI,P_CTL,P_MES))
def stat(d):
    d=np.asarray(d); m,sd=d.mean(),(d.std(ddof=1) if len(d)>1 else float('nan'))
    return m,sd,(abs(m)/sd if sd==sd and sd>0 else float('nan')),int((d<0).sum()),len(d)

if __name__=="__main__":
    seeds=[s for s in DEV+NEW if have(s)]
    missing=[s for s in DEV+NEW if not have(s)]
    print("H106 -- does the ROI close the one real gap, seed-matched at n=10?\n")
    print(f"  seeds usable: {len(seeds)}/10   missing: {missing if missing else 'none'}")
    if missing:
        print("\n  INCOMPLETE -- every bar refuses. (A partial arm has read better than its")
        print("  completed self three times in this project.)")
        sys.exit()
    roi=np.array([at200(P_ROI[s]) for s in seeds])
    ctl=np.array([at200(P_CTL[s]) for s in seeds])
    mes=np.array([at200(P_MES[s]) for s in seeds])
    print(f"\n  {'seed':>5}{'ROI':>9}{'no-ROI':>9}{'MF-MES':>9}{'ROI-noROI':>11}{'ROI-MES':>10}")
    for i,s in enumerate(seeds):
        print(f"  {s:>5}{roi[i]:>9.2f}{ctl[i]:>9.2f}{mes[i]:>9.2f}{roi[i]-ctl[i]:>+11.2f}{roi[i]-mes[i]:>+10.2f}")
    print("\n  REGISTERED BARS")
    m,sd,r,w,n=stat(roi-ctl)
    q1 = (m<0) and (r>=0.5)
    print(f"    Q1 PRIMARY  ROI vs no-ROI: mean {m:+.2f} sd {sd:.2f} ratio {r:.2f} better {w}/{n}")
    print(f"                bar = negative mean AND ratio >= 0.50  ->  {'MET' if q1 else 'FAILED'}")
    gap_before = ctl.mean()-mes.mean(); gap_after = roi.mean()-mes.mean()
    print(f"    Q2 NEGATIVE ROI does NOT close the gap to MF-MES:")
    print(f"                no-ROI {ctl.mean():.2f} vs MES {mes.mean():.2f}  gap {gap_before:+.2f}")
    print(f"                ROI    {roi.mean():.2f} vs MES {mes.mean():.2f}  gap {gap_after:+.2f}"
          f"   ({100*(gap_before-gap_after)/gap_before:.0f}% of the gap closed)")
    print(f"                -> {'MET (still behind)' if roi.mean()>mes.mean() else '*** REFUTED -- re-verify before believing ***'}")
    dv=roi[:5]-ctl[:5]; nw=roi[5:]-ctl[5:]
    print(f"    Q3 halves   42-46 mean {dv.mean():+.2f} | 52-56 mean {nw.mean():+.2f}"
          f" | split {abs(dv.mean()-nw.mean()):.2f}")
    print(f"                bar = split < 2.0  ->  {'MET' if abs(dv.mean()-nw.mean())<2.0 else 'FAILED'}")
    if abs(dv.mean()-nw.mean())>=2.0:
        print("                *** Q3 FAILED -> the halves may NOT be pooled. Two named")
        print("                    candidates: seed-set dependence, or code drift (h84's arm")
        print("                    predates the working-tree patches; comparability was")
        print("                    REASONED, not measured). Do not attribute to seeds by default.")
    if not q1:
        print("\n    *** Q1 FAILED -> q=0.10 does not measurably reduce the one real deficit.")
        print("        This licenses 'q=0.10 does not', NOT 'the ROI does not' -- q=0.05 is")
        print("        separably better and is untested at these seeds. ***")
