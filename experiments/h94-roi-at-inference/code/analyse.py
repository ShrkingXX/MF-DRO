"""H94: does applying the ROI to the QUERY beat applying it to the TEACHER?

Committed BEFORE any H94 run exists. Same frozen metric as h83/h84/h87/h90.
Arms A (NO-ROI) and B (ROI-Q10) are READ FROM h90 at the same seeds; C and D
are h94's own. Every bar refuses to evaluate on incomplete data rather than
printing a verdict from a partial arm -- three times in this project a partial
arm read better than its completed self.
"""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(47,48,49,50,51)
H94=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
H90=os.path.join(REPO,"experiments","h90-borehole-confirm","results")
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
MES_BOREHOLE=6.40   # h83 seeds 42-46; see P4's caveat below

def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT

def arm(where, name):
    """-> {seed: value} for the seeds that have a completed run."""
    out={}
    for s in SEEDS:
        p=os.path.join(where,f"{B}__{name}__seed{s}.json")
        if os.path.exists(p): out[s]=at200(p)
    return out

def paired(a, b):
    """a minus b on the seeds BOTH have. Returns (seeds, diffs)."""
    ks=sorted(set(a)&set(b))
    return ks, np.array([a[k]-b[k] for k in ks])

def show(label, a, b, ref=None):
    ks,d=paired(a,b)
    print(f"\n  {label}   (n={len(ks)}/5)")
    if not ks: print("    no complete pairs"); return None
    for k in ks: print(f"    seed {k}: {a[k]:6.2f} vs {b[k]:6.2f}   {a[k]-b[k]:+7.2f}")
    print(f"    paired mean {d.mean():+.2f}  sd {d.std(ddof=1) if len(d)>1 else float('nan'):.2f}"
          f"  better {int((d<0).sum())}/{len(d)}")
    if ref: print(f"    {ref}")
    return (ks,d)

def snapstats():
    rows=[]
    for name in ("ROI-PROJECT","SNAP-CONTROL"):
        for s in SEEDS:
            p=os.path.join(H94,f"{B}__{name}__seed{s}.json")
            if not os.path.exists(p): continue
            ri=(json.load(open(p)).get("roi_inference") or {})
            if ri: rows.append((name,s,ri.get("snapped_frac"),ri.get("mean_snap_dist"),
                                ri.get("mean_accept_frac"),ri.get("n_empty_roi")))
    return rows

if __name__=="__main__":
    A=arm(H90,"NO-ROI"); Bq=arm(H90,"ROI-Q10")
    C=arm(H94,"ROI-PROJECT"); D=arm(H94,"SNAP-CONTROL")
    print("H94 -- the ROI applied to the QUERY vs to the TEACHER. Borehole, seeds 47-51.")
    print(f"  arms present: NO-ROI {len(A)}/5 (h90)  ROI-Q10 {len(Bq)}/5 (h90)  "
          f"ROI-PROJECT {len(C)}/5  SNAP-CONTROL {len(D)}/5")

    print("\n  P5 FIRST -- did the manipulation intervene at all?")
    rows=snapstats()
    if not rows:
        print("    CANNOT EVALUATE -- no run has written roi_inference yet.")
        p5=None
    else:
        print(f"    {'arm':<14}{'seed':>5}{'snapped':>10}{'dist':>8}{'accept':>9}{'emptyROI':>10}")
        for n,s,f,dd,ac,ne in rows:
            print(f"    {n:<14}{s:>5}{(f if f is not None else float('nan')):>10.3f}"
                  f"{(dd if dd is not None else float('nan')):>8.3f}"
                  f"{(ac if ac is not None else float('nan')):>9.3f}{ne:>10}")
        pf=[f for n,s,f,_,_,_ in rows if n=="ROI-PROJECT" and f is not None]
        if len(pf)<len(C) or not C:
            print("    P5: CANNOT EVALUATE -- ROI-PROJECT arm incomplete")
            p5=None
        else:
            p5 = max(pf) > 0.5
            print(f"    P5 (some run snapped >50% of queries): {'MET' if p5 else 'FAILED'}"
                  f"   max snapped_frac={max(pf):.3f}")
            if not p5:
                print("    *** P5 FAILED -> the DT's raw output was already almost always")
                print("        admissible, so ROI-PROJECT is near-identical to ROI-Q10 by")
                print("        construction. C and D are INCONCLUSIVE, not negative. ***")

    r1=show("P1 PRIMARY  C(ROI-PROJECT) - D(SNAP-CONTROL)  [isolates the ROI]", C, D)
    r2=show("P2          C(ROI-PROJECT) - A(NO-ROI)", C, A)
    r3=show("P3 input    B(ROI-Q10)     - A(NO-ROI)   [the imitation channel]", Bq, A,
            ref="h90's own comparison, repeated here for the P3 ratio")
    rD=show("context     D(SNAP-CONTROL)- A(NO-ROI)   [snapping alone]", D, A)

    print("\n  REGISTERED BARS")
    done_C = len(C)==5; done_D = len(D)==5; done_A = len(A)==5; done_B = len(Bq)==5
    if not (done_C and done_D) or r1 is None:
        print("    P1: CANNOT EVALUATE -- C and/or D incomplete")
    else:
        ks,d=r1; p1 = int((d<0).sum())>=4 and d.mean()<0
        print(f"    P1 (C beats D on >=4/5 AND negative mean): {'MET' if p1 else 'FAILED'}")
        if not p1 and rD is not None and rD[1].mean()<0:
            print("    *** D helps but C does not beat it -> quantizing onto a finite pool")
            print("        did the work, NOT the ROI. That is the EXCLUDED pool mechanism")
            print("        resurfacing, and it is not a contribution. Report it as such. ***")
    if not (done_C and done_A) or r2 is None:
        print("    P2: CANNOT EVALUATE -- C and/or A incomplete")
    else:
        ks,d=r2; print(f"    P2 (C beats A on >=4/5 AND negative mean): "
                       f"{'MET' if (int((d<0).sum())>=4 and d.mean()<0) else 'FAILED'}")
    if not (done_C and done_A and done_B) or r2 is None or r3 is None:
        print("    P3: CANNOT EVALUATE -- needs A, B and C complete")
    else:
        ca=abs(r2[1].mean()); ba=abs(r3[1].mean())
        print(f"    P3 (|C-A| > |B-A|: the query constraint beats the imitation channel): "
              f"{'MET' if ca>ba else 'FAILED'}   |C-A|={ca:.2f}  |B-A|={ba:.2f}")
    if not done_C:
        print("    P4: CANNOT EVALUATE -- C incomplete")
    else:
        cm=float(np.mean(list(C.values())))
        print(f"    P4 (C still does NOT beat MF-MES): "
              f"{'MET (still behind)' if cm>MES_BOREHOLE else '*** REFUTED -- re-verify before believing ***'}"
              f"   C mean={cm:.2f} vs {MES_BOREHOLE}")
        print("      CAVEAT registered in advance: 6.40 is MF-MES at seeds 42-46, and h89")
        print("      showed seeds differ by up to 3.67 pts on this benchmark. A borderline")
        print("      P4 needs MF-MES re-run at 47-51 before any claim either way.")
