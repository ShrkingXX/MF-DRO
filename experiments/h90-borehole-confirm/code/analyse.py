"""H90: does the Borehole ROI gain survive fresh seeds? Committed BEFORE the
treatment arm finishes, as h87's was. Same metric as everything else."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Borehole_8D"; SEEDS=(47,48,49,50,51)
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
if __name__=="__main__":
    rows=[];miss=[]
    for s in SEEDS:
        a=os.path.join(R,f"{B}__ROI-Q10__seed{s}.json"); c=os.path.join(R,f"{B}__NO-ROI__seed{s}.json")
        if os.path.exists(a) and os.path.exists(c): rows.append((s,at200(a),at200(c)))
        else: miss.append(s)
    print("H90 -- Borehole ROI gain at FRESH seeds, q=0.10 fixed in advance.\n")
    if not rows: print("  no complete pairs yet"); sys.exit()
    print(f"  {'seed':>5}{'ROI-Q10':>10}{'no ROI':>9}{'diff':>9}")
    for s,a,c in rows: print(f"  {s:>5}{a:>10.2f}{c:>9.2f}{a-c:>+9.2f}")
    d=np.array([a-c for _,a,c in rows]); wins=int((d<0).sum())
    print(f"\n  n={len(d)}  paired mean {d.mean():+.2f}  "
          f"sd {d.std(ddof=1) if len(d)>1 else float('nan'):.2f}  ROI better {wins}/{len(d)}")
    print(f"  seeds 42-46 for reference: mean -4.22, sd 2.43, 5/5")
    if miss: print(f"  INCOMPLETE -- pairs pending: {miss}"); sys.exit()
    print("\n  REGISTERED BARS")
    p1 = wins>=4 and d.mean()<0
    print(f"    P1 (negative on >=4/5 AND negative mean): {'MET' if p1 else 'FAILED'}")
    print(f"    P2 (margin shrinks vs -4.22): "
          f"{'MET' if abs(d.mean())<4.22 or d.mean()>0 else 'NOT MET'}")
    roi=np.array([a for _,a,_ in rows])
    print(f"    P3 (still does NOT beat MF-MES 6.40): "
          f"{'MET (still behind)' if roi.mean()>6.40 else '*** REFUTED -- investigate ***'}")
    # --- REFINE-100 arm, bar registered in the protocol addendum before any result ---
    rrows=[];rmiss=[]
    for s in SEEDS:
        a=os.path.join(R,f"{B}__REFINE-100__seed{s}.json"); c=os.path.join(R,f"{B}__NO-ROI__seed{s}.json")
        if os.path.exists(a) and os.path.exists(c): rrows.append((s,at200(a),at200(c)))
        else: rmiss.append(s)
    print("\n  TEACHER REFINEMENT, third seed set")
    if rmiss: print(f"    INCOMPLETE -- pairs pending: {rmiss}")
    elif rrows:
        for s,a,c in rrows: print(f"    seed {s}: REFINE {a:6.2f}  no-ROI {c:6.2f}  {a-c:+7.2f}")
        rd=np.array([a-c for _,a,c in rrows]); rw=int((rd<0).sum())
        print(f"    n={len(rd)}  paired mean {rd.mean():+.2f}  refinement better {rw}/{len(rd)}")
        print(f"    record: seeds 42-46 mean -5.85 5/5 | seeds 52-56 mean -2.11 4/5")
        p4 = rw>=3 and rd.mean()<0
        print(f"    P4 (>=3/5 AND negative mean): {'MET' if p4 else 'FAILED'}")
        read = "STABLE" if rd.mean()<=-1.0 and rw>=4 else ("DECAYING" if p4 else "DEAD")
        print(f"    reading: {read}  (STABLE ~-2.1 4/5 | DECAYING ~-1.0 3/5 | else dead)")
        print(f"    P5 (still above strongest baseline, not competitive): "
              f"{'MET' if np.array([a for _,a,_ in rrows]).mean()>10.07 else '*** REFUTED -- investigate ***'}")
        if not p4:
            print("\n    *** P4 FAILED -> teacher refinement joins the ROI flip and the HF")
            print("        floor as withdrawn. NO intervention tried this session survives")
            print("        fresh seeds; the session's answer is uniformly negative. ***")
    if not p1:
        print("\n  *** P1 FAILED -> per the protocol's falsifier, the Borehole gain is")
        print("      WITHDRAWN, and the ROI has NO surviving regret result on any")
        print("      benchmark. Only the controllability argument remains. State that")
        print("      plainly in findings.md and the report. ***")
