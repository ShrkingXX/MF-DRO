"""H113: do the ROI and the L1 loss compose, or share a bottleneck?
Committed before any run. Three of the four 2x2 cells are reused at matched seeds;
the use_roi=False branch is byte-identical across every contributing commit
(md5 ff70f008c0ac, h105) and the patched src reproduces h84 exactly (h109)."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; BAR=0.59
SETS={"42-46":(42,43,44,45,46), "47-51":(47,48,49,50,51)}
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
R113=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
E=lambda *p: os.path.join(REPO,"experiments",*p)
# Explicit per-cell, per-seed-set sources. No globbing across experiment dirs.
SRC={
 "BASE": {"42-46":E("h83-main-comparison","results",f"{B}__MF-DRO__seed%d.json"),
          "47-51":E("h90-borehole-confirm","results",f"{B}__NO-ROI__seed%d.json")},
 "ROI":  {"42-46":E("h84-roi-strategy","results",f"{B}__ROI-Q10__seed%d.json"),
          "47-51":E("h90-borehole-confirm","results",f"{B}__ROI-Q10__seed%d.json")},
 "L1":   {"42-46":E("h108-l1-confirm","results",f"{B}__L1-LOSS__seed%d.json"),
          "47-51":E("h102-boundary-loss","results",f"{B}__L1-LOSS__seed%d.json")},
 "BOTH": {"42-46":os.path.join(R113,f"{B}__ROI-L1__seed%d.json"),
          "47-51":os.path.join(R113,f"{B}__ROI-L1__seed%d.json")},
}
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
def cell(name):
    out={}
    for tag,seeds in SETS.items():
        for s in seeds:
            p=SRC[name][tag] % s
            if os.path.exists(p): out[s]=at200(p)
    return out
if __name__=="__main__":
    print("H113 -- do the calibrated ROI and the L1 location loss compose?\n")
    both=cell("BOTH")
    have=sorted(both)
    print("  DOUBLE GATE -- both manipulations must be OBSERVED, not read from config.")
    if not have: print("    no ROI-L1 runs yet\n"); sys.exit()
    void=False
    for s in have:
        d=json.load(open(SRC["BOTH"]["42-46"] % s))
        acc=(d.get("roi_summary") or {}).get("accept_frac")
        ll=(d.get("L_loc_per_iter") or [])
        # AMENDED (see protocol): the tail statistic is depressed by the ROI's
        # own concentration of the teacher targets, so it cannot separate "L1
        # did not fire" from "L1 fired and the ROI narrowed its targets". The
        # FIRST-FIVE-iteration value cannot be depressed that way -- MSE arms
        # give 0.05-0.06 there, L1 arms 0.19-0.22. Amended before any regret
        # number was read; the analysis exits at this gate.
        llf=float(np.mean(ll[:5])) if len(ll)>=5 else float('nan')
        lltail=float(np.mean(ll[-10:])) if len(ll)>=10 else float('nan')
        a_ok = acc is not None and 0.095<=acc<=0.105
        l_ok = llf>0.10
        void = void or not (a_ok and l_ok)
        print(f"    seed {s}: accept_frac={acc if acc is None else round(acc,4)} {'OK' if a_ok else 'BAD'}"
              f"   L_loc(first5)={llf:.4f} {'OK' if l_ok else 'BAD'}   [tail {lltail:.4f}, ROI-depressed]")
    if void: print("\n    *** GATE FAILED -- a manipulation did not fire. Arm VOID. ***"); sys.exit(1)
    print("    -> PASS (both fired)\n")
    base,roi,l1=cell("BASE"),cell("ROI"),cell("L1")
    seeds=[s for s in sorted(set(both)&set(base)&set(roi)&set(l1))]
    missing=[s for tag,ss in SETS.items() for s in ss if s not in both]
    print(f"  {'seed':>5}{'base':>8}{'ROI':>8}{'L1':>8}{'BOTH':>8}{'BOTH-base':>11}{'BOTH-ROI':>10}")
    for s in seeds:
        print(f"  {s:>5}{base[s]:>8.2f}{roi[s]:>8.2f}{l1[s]:>8.2f}{both[s]:>8.2f}"
              f"{both[s]-base[s]:>+11.2f}{both[s]-roi[s]:>+10.2f}")
    if missing: print(f"\n  INCOMPLETE -- pending {missing}"); sys.exit()
    dB=np.array([both[s]-base[s] for s in seeds])
    dR=np.array([both[s]-roi[s] for s in seeds])
    eR=np.array([roi[s]-base[s] for s in seeds]); eL=np.array([l1[s]-base[s] for s in seeds])
    n=len(seeds)
    print(f"\n  n={n}")
    print(f"    ROI alone  vs base: {eR.mean():+.2f}  better {int((eR<0).sum())}/{n}")
    print(f"    L1 alone   vs base: {eL.mean():+.2f}  better {int((eL<0).sum())}/{n}")
    print(f"    BOTH       vs base: {dB.mean():+.2f}  better {int((dB<0).sum())}/{n}")
    print(f"    BOTH       vs ROI : {dR.mean():+.2f}  sd {dR.std(ddof=1):.2f}  better {int((dR<0).sum())}/{n}")
    print("\n  REGISTERED BARS")
    print(f"    P1 (BOTH beats base): {'MET' if (dB.mean()<0 and int((dB<0).sum())>=8) else 'FAILED'}")
    sep = abs(dR.mean())>BAR and max(int((dR<0).sum()),n-int((dR<0).sum()))>=8
    if sep and dR.mean()<0:
        print(f"    P2 (no direction registered): **BOTH BEATS ROI ALONE** by {abs(dR.mean()):.2f}, clearing the bar.")
    elif sep:
        print(f"    P2: *** BOTH is WORSE than ROI alone by {dR.mean():.2f} -- they INTERFERE. ***")
    else:
        print(f"    P2: INDISTINGUISHABLE from ROI alone ({abs(dR.mean()):.2f} vs {BAR} bar, "
              f"{int((dR<0).sum())}/{n}).")
    print(f"\n  P3 -- DESCRIPTIVE, no threshold registered:")
    print(f"    additive would be {eR.mean()+eL.mean():+.2f} (ROI {eR.mean():+.2f} + L1 {eL.mean():+.2f})")
    print(f"    shared bottleneck would be about {eR.mean():+.2f} (no better than ROI alone)")
    print(f"    MEASURED: {dB.mean():+.2f}")
    print(f"    Report the position, do not name it. 'Partially additive' is not a finding.")
