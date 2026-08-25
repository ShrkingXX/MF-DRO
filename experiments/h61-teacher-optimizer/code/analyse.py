"""H61 verdict against its locked predictions. Refuses partial arms."""
import os, sys, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
S = (44, 46, 48)

def main():
    from benchmarks import get_benchmark
    fs = -float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
    base = {s: json.load(open(os.path.join(H57, f"Borehole_8D__MF-DRO__seed{s}.json"))) for s in S}
    bv = np.array([base[s]["final_regret"] for s in S])
    def mix(d):
        q = [e for e in d["queries"] if not e.get("is_init")]
        nh = sum(1 for e in q if e["fid"] == 1); return nh, len(q) - nh
    print(f"  {'arm':<10}{'s44':>10}{'s46':>10}{'s48':>10}{'mean':>10}{'rel':>7}{'wins':>6}   HF/LF")
    print(f"  {'BASE':<10}" + "".join(f"{x:>10.2f}" for x in bv)
          + f"{bv.mean():>10.2f}{bv.mean()/fs:>6.1%}{'--':>6}   "
          + " ".join(f"{a}/{b}" for a, b in [mix(base[s]) for s in S]))
    arms = {}
    for arm in ("POOL600", "REFINE"):
        got = {}
        for s in S:
            f = os.path.join(RES, f"Borehole_8D__{arm}__seed{s}.json")
            if os.path.exists(f): got[s] = json.load(open(f))
        if len(got) < 3:
            print(f"  {arm:<10}{('INCOMPLETE ' + str(len(got)) + '/3 -- WITHHELD'):>44}"); continue
        v = np.array([got[s]["final_regret"] for s in S]); arms[arm] = v
        w = sum(1 for s in S if got[s]["final_regret"] < base[s]["final_regret"])
        print(f"  {arm:<10}" + "".join(f"{x:>10.2f}" for x in v)
              + f"{v.mean():>10.2f}{v.mean()/fs:>6.1%}{str(w)+'/3':>6}   "
              + " ".join(f"{a}/{b}" for a, b in [mix(got[s]) for s in S]))
    if len(arms) < 2:
        print("\n  Verdict withheld: both arms required."); return
    print("\n  LOCKED PREDICTIONS")
    r = arms["REFINE"]; p = arms["POOL600"]
    w = int((r < bv).sum())
    print(f"  1 PRIMARY   REFINE vs BASE: {w}/3 wins, {r.mean():.2f} vs {bv.mean():.2f}")
    ok2 = w >= 2 and bv.mean() > p.mean() > r.mean()
    print(f"  2 EXPECTED  REFINE beats BASE >=2/3 AND POOL600 lands between: "
          f"{'MET' if ok2 else 'NOT MET'}")
    print(f"              (BASE {bv.mean():.2f} | POOL600 {p.mean():.2f} | REFINE {r.mean():.2f})")
    print(f"  3 NULL      neither moves: {'fired' if w<2 and (p<bv).sum()<2 else 'no'}")
    print(f"  4 HARMFUL   a sharper teacher is worse: "
          f"{'FIRED for REFINE' if w<=1 else 'no'}")
    print("\n  Note: H61's liveness check already falsified the HARMFUL branch's stated")
    print("  premise -- refinement WIDENED the query spread (0.2907 vs BASE 0.2534)")
    print("  rather than concentrating it. If HARMFUL fires it needs a new explanation.")

if __name__ == "__main__":
    main()
