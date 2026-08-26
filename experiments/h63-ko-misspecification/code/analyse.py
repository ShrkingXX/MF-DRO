"""H63 verdict. The discriminating signal is the CONTRAST -- a Borehole gain
LARGER than Hartmann's -- not a Borehole win alone. Refuses partial arms.
Written with 5/6 cells on disk and the Hartmann arm incomplete."""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
S = (44, 46, 48)
SLOPE = {"Borehole_8D": (1.2566, "OUTSIDE (0,1) -- unrepresentable"),
         "Hartmann_6D": (0.9792, "inside (0,1) -- representable, CONTROL")}

def main():
    from benchmarks import get_benchmark
    print("  H63 -- KO rho misspecification. Discriminating signal is the CONTRAST.\n")
    res = {}
    for b in ("Borehole_8D", "Hartmann_6D"):
        fs = -float(get_benchmark(f"{b}_HF")["known_optimal_value"])
        base = {s: json.load(open(os.path.join(H57, f"{b}__MF-DRO__seed{s}.json"))) for s in S}
        got = {}
        for s in S:
            f = os.path.join(RES, f"{b}__RHOTRUE__seed{s}.json")
            if os.path.exists(f): got[s] = json.load(open(f))
        sl, note = SLOPE[b]
        print(f"  {b}   true slope {sl}  [{note}]")
        if len(got) < 3:
            print(f"    INCOMPLETE {len(got)}/3 -- WITHHELD\n"); res[b] = None; continue
        bv = np.array([base[s]["final_regret"] for s in S])
        rv = np.array([got[s]["final_regret"] for s in S])
        def mix(d):
            q = [e for e in d["queries"] if not e.get("is_init")]
            nh = sum(1 for e in q if e["fid"] == 1); return nh / max(len(q), 1)
        w = int((rv < bv).sum())
        gain = (bv.mean() - rv.mean()) / fs
        res[b] = (w, gain, bv, rv)
        print(f"    {'BASE':<9}{bv.mean():>10.4f}{bv.mean()/fs:>8.1%}   HF "
              + " ".join(f"{mix(base[s]):.0%}" for s in S))
        print(f"    {'RHOTRUE':<9}{rv.mean():>10.4f}{rv.mean()/fs:>8.1%}   HF "
              + " ".join(f"{mix(got[s]):.0%}" for s in S)
              + f"   wins {w}/3   gain {gain:+.1%}\n")
    bo, ha = res.get("Borehole_8D"), res.get("Hartmann_6D")
    if bo is None or ha is None:
        print("  Verdict withheld: the contrast needs BOTH benchmarks complete."); return
    print("  LOCKED PREDICTIONS")
    print(f"  1 PRIMARY        RHOTRUE beats BASE on Borehole >=2/3: "
          f"{'MET' if bo[0]>=2 else 'NOT MET'} ({bo[0]}/3)")
    disc = bo[1] > ha[1]
    print(f"  2 DISCRIMINATING Borehole gain > Hartmann gain: "
          f"{'MET' if disc else 'NOT MET'}   ({bo[1]:+.1%} vs {ha[1]:+.1%})")
    print(f"  3 NULL           neither moves: "
          f"{'fired' if bo[0]<2 and ha[0]<2 else 'no'}")
    print()
    if bo[0] >= 2 and disc:
        print("  -> Supports range-violation: the benchmark whose rho is unrepresentable")
        print("     gains, the control whose rho is representable gains less or loses.")
        print("     STILL CONFOUNDED four ways (value, adaptivity, ensemble diversity,")
        print("     fidelity mix) -- confounds 2-4 act on BOTH benchmarks, which is why")
        print("     the CONTRAST and not the level is the signal.")
    elif bo[0] >= 2 and not disc:
        print("  -> NOT support. Both benchmarks gain similarly, so the effect is one of")
        print("     the confounds (most plausibly the fidelity shift, measured at")
        print("     99%->6-31% HF on Borehole and 81%->16% on Hartmann), not rho's range.")
    else:
        print("  -> rho misspecification excluded as the Borehole mechanism.")
    print("\n  Context: on the same Borehole cells, h61's POOL600 reached 19.5% and")
    print("  REFINE 19.3% -- both better than RHOTRUE's 21.8%. Whatever rho contributes,")
    print("  teacher optimisation contributes more.")

if __name__ == "__main__":
    main()
