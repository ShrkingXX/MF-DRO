"""H65 verdict. PRIMARY is VARIANCE, not mean -- committed before data because
h61's mean and variance gains came apart. Refuses partial arms."""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
S = (44, 46, 48)

def main():
    from benchmarks import get_benchmark
    fs = -float(get_benchmark("Hartmann_6D_HF")["known_optimal_value"])
    base = {s: json.load(open(os.path.join(H57, f"Hartmann_6D__MF-DRO__seed{s}.json"))) for s in S}
    got = {}
    for s in S:
        f = os.path.join(RES, f"Hartmann_6D__REFINE__seed{s}.json")
        if os.path.exists(f): got[s] = json.load(open(f))
    if len(got) < 3:
        print(f"  INCOMPLETE {len(got)}/3 -- WITHHELD. Verdict not evaluated."); return
    bv = np.array([base[s]["final_regret"] for s in S])
    rv = np.array([got[s]["final_regret"] for s in S])
    bs, rs = bv.max() - bv.min(), rv.max() - rv.min()
    w = int((rv < bv).sum())
    print(f"  {'arm':<8}" + "".join(f"{'s'+str(s):>10}" for s in S)
          + f"{'mean':>10}{'rel':>8}{'spread':>9}")
    print(f"  {'BASE':<8}" + "".join(f"{x:>10.4f}" for x in bv)
          + f"{bv.mean():>10.4f}{bv.mean()/fs:>7.1%}{bs:>9.4f}")
    print(f"  {'REFINE':<8}" + "".join(f"{x:>10.4f}" for x in rv)
          + f"{rv.mean():>10.4f}{rv.mean()/fs:>7.1%}{rs:>9.4f}")
    print("\n  LOCKED PREDICTIONS")
    print(f"  1 PRIMARY (variance)  REFINE spread < BASE spread: "
          f"{'MET' if rs < bs else 'NOT MET'}  ({rs:.4f} vs {bs:.4f})")
    print(f"  2 SECONDARY (mean)    REFINE beats BASE >=2/3: {'MET' if w>=2 else 'NOT MET'} ({w}/3)")
    print(f"  3 BOREHOLE-SPECIFIC   neither moves: "
          f"{'fired' if rs>=bs and w<2 else 'no'}")
    print(f"  4 HARMFUL             spread INCREASES: {'FIRED' if rs>bs else 'no'}")
    print("\n  h61 reference (Borehole): BASE spread 8.62 -> REFINE 2.57, a 3.4x contraction.")
    print("  Prediction 1 was chosen as PRIMARY because h61's mean gain was matched by")
    print("  POOL600 at 10x the spread -- mean is the weaker signal for this mechanism.")

if __name__ == "__main__":
    main()
