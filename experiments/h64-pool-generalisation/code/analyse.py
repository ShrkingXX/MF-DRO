"""H64 verdict. PRIMARY is Hartmann; Currin is a NO-HARM CHECK, not a test.
Refuses partial arms. Written before any result existed."""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
S = (44, 46, 48)

def main():
    from benchmarks import get_benchmark
    print("  H64 -- does the teacher-pool fix generalise? PRE-REGISTERED: Hartmann NULL")
    print("  (widening 200->600 buys Hartmann 1.00x acquisition value; Borehole 1.44x)\n")
    verdicts = {}
    for b in ("Hartmann_6D", "Currin_2D"):
        fs = -float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        base = {s: json.load(open(os.path.join(H57, f"{b}__MF-DRO__seed{s}.json"))) for s in S}
        got = {}
        for s in S:
            f = os.path.join(RES, f"{b}__POOL600__seed{s}.json")
            if os.path.exists(f): got[s] = json.load(open(f))
        role = "PRIMARY" if b == "Hartmann_6D" else "NO-HARM CHECK (saturated)"
        print(f"  {b}  [{role}]")
        if len(got) < 3:
            print(f"    INCOMPLETE {len(got)}/3 -- WITHHELD\n"); verdicts[b] = None; continue
        bv = np.array([base[s]["final_regret"] for s in S])
        pv = np.array([got[s]["final_regret"] for s in S])
        w = int((pv < bv).sum()); verdicts[b] = (w, bv, pv)
        print(f"    {'BASE':<9}" + "".join(f"{x:>10.4f}" for x in bv)
              + f"{bv.mean():>10.4f}{bv.mean()/fs:>7.1%}  sd {bv.std(ddof=1):.4f}")
        print(f"    {'POOL600':<9}" + "".join(f"{x:>10.4f}" for x in pv)
              + f"{pv.mean():>10.4f}{pv.mean()/fs:>7.1%}  sd {pv.std(ddof=1):.4f}   wins {w}/3\n")
    h = verdicts.get("Hartmann_6D")
    if h is None:
        print("  Verdict withheld: Hartmann (the primary) is incomplete."); return
    w, bv, pv = h
    print("  LOCKED PREDICTIONS")
    print(f"  1 PRIMARY    POOL600 beats BASE on Hartmann >=2/3: {'MET' if w>=2 else 'NOT MET'} ({w}/3)")
    print(f"  3 BOREHOLE-SPECIFIC (no Hartmann movement): {'fired' if w<2 else 'no'}")
    print(f"  4 HARMFUL on Hartmann: {'FIRED' if w<=1 and pv.mean()>bv.mean() else 'no'}")
    print()
    if w >= 2:
        print("  NOTE: this CONTRADICTS the pre-registered null. The acquisition-value")
        print("  channel measured 1.00x for Hartmann at N=600, so a real gain here means")
        print("  a wider pool helps through some OTHER channel, and the mechanism is")
        print("  UNEXPLAINED. That was written down in advance and must not be explained away.")
    else:
        print("  Consistent with the pre-registered null: no acquisition gain -> no regret gain.")
        print("  h61's Borehole win is then benchmark-specific for an acquisition-value reason.")
    c = verdicts.get("Currin_2D")
    if c: print(f"\n  Currin ({c[0]}/3) is a saturated-benchmark no-harm check and is NOT support either way.")

if __name__ == "__main__":
    main()
