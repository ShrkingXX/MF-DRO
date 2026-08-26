"""H66 verdict: does h64's Hartmann north-star result survive n=10?

Written with POOL600 at 0/7. Refuses partial arms. Reports the Wilcoxon
UNCONDITIONALLY, including when it is non-significant beside a favourable win
count -- that combination was pre-registered as expected, not as a refutation.
"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
H64 = os.path.join(REPO, "experiments", "h64-pool-generalisation", "results")
H57 = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")
NEW = [42, 43, 45, 47, 49, 50, 51]
OLD = [44, 46, 48]

def load(seed, arm):
    """Reused cells come from h64/h57; new cells from h66. Same code, same config."""
    if seed in OLD:
        d = H64 if arm == "POOL600" else H57
        p = os.path.join(d, f"Hartmann_6D__{arm}__seed{seed}.json")
    else:
        p = os.path.join(RES, f"Hartmann_6D__{arm}__seed{seed}.json")
    return json.load(open(p))["final_regret"] if os.path.exists(p) else None

def main():
    from benchmarks import get_benchmark
    fs = -float(get_benchmark("Hartmann_6D_HF")["known_optimal_value"])
    S = sorted(OLD + NEW)
    P = {s: load(s, "POOL600") for s in S}
    M = {s: load(s, "MF-MES") for s in S}
    miss = [s for s in S if P[s] is None or M[s] is None]
    if miss:
        have = len([s for s in S if P[s] is not None])
        print(f"  INCOMPLETE -- POOL600 {have}/10, missing seeds {miss}. WITHHELD.")
        return
    p = np.array([P[s] for s in S]); m = np.array([M[s] for s in S])
    w = int((p < m).sum())
    print(f"  {'seed':>5}{'POOL600':>11}{'MF-MES':>11}{'winner':>10}")
    for i, s in enumerate(S):
        print(f"  {s:>5}{p[i]:>11.4f}{m[i]:>11.4f}{('POOL600' if p[i]<m[i] else 'MF-MES'):>10}")
    print(f"\n  {'POOL600':<10}mean {p.mean():.4f}  rel {p.mean()/fs:.1%}  sd {p.std(ddof=1):.4f}")
    print(f"  {'MF-MES':<10}mean {m.mean():.4f}  rel {m.mean()/fs:.1%}  sd {m.std(ddof=1):.4f}")
    print(f"\n  paired wins: POOL600 {w}/10")
    try:
        from scipy.stats import wilcoxon
        pv = wilcoxon(p, m).pvalue
        print(f"  Wilcoxon signed-rank p = {pv:.4f}   (reported unconditionally)")
    except Exception as e:
        pv = float("nan"); print(f"  Wilcoxon unavailable: {e!r}")
    print("\n  LOCKED PREDICTIONS")
    ok = (p.mean() < m.mean()) and w >= 6
    print(f"  1 PRIMARY   mean below MF-MES AND >=6/10 wins: {'MET' if ok else 'NOT MET'}")
    print(f"  2 FAILURE   <=5/10 wins -> WITHDRAW the north-star arrival: "
          f"{'FIRED -- WITHDRAW' if w <= 5 else 'no'}")
    base_sd = 0.2395
    print(f"  3 VARIANCE  POOL600 sd below BASE's {base_sd}: "
          f"{'MET' if p.std(ddof=1) < base_sd else 'NOT MET'} ({p.std(ddof=1):.4f})")
    print()
    if w <= 5:
        print("  h64's 2/3 was noise. The claim that a DRO variant beats the best")
        print("  Hartmann baseline is WITHDRAWN, and this must be reported as")
        print("  prominently as the original claim was.")
    elif ok and np.isfinite(pv) and pv >= 0.05:
        print("  Direction holds at n=10 but Wilcoxon is not significant. This was")
        print("  PRE-REGISTERED as expected: h17-vs-h31 needed 82 seeds for 80% power on")
        print("  a smaller effect. The win count and direction carry the claim; the")
        print("  non-significance is stated, not buried, and is not itself a refutation.")
    elif ok:
        print("  Replicated at n=10 with a significant signed-rank test.")
    else:
        print("  Mixed: mean or win-count criterion not met. Report as unresolved.")

if __name__ == "__main__":
    main()
