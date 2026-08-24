"""
Analysis for h1-leak-fix-validation. Applies the FROZEN success test from
PROTOCOL.md and verifies cost-matching before reporting anything.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
METHODS = ["MF-DRO", "MF-MI-Greedy", "MF-GP-UCB"]


def load_all():
    out = {}
    for m in METHODS:
        runs = []
        for s in SEEDS:
            p = os.path.join(RESULTS, f"{m}__seed{s}.json")
            if os.path.exists(p):
                runs.append(json.load(open(p)))
        out[m] = runs
    return out


def main():
    data = load_all()

    print("=" * 78)
    print("COST MATCHING CHECK (must pass before any regret comparison)")
    print("=" * 78)
    cap_bound_any = False
    for m in METHODS:
        runs = data[m]
        if not runs:
            print(f"  {m}: NO RUNS")
            continue
        costs = [r["final_cost"] for r in runs]
        capped = [r["seed"] for r in runs if r.get("iter_cap_bound")]
        cap_bound_any |= bool(capped)
        print(f"  {m:<14} n={len(runs):>2}  final_cost mean={np.mean(costs):7.1f} "
              f"min={np.min(costs):7.1f} max={np.max(costs):7.1f}"
              + (f"  ** ITER-CAP BOUND on seeds {capped} -- NOT cost-matched **"
                 if capped else ""))
    if cap_bound_any:
        print("\n  WARNING: at least one run stopped on the iteration guard, not the")
        print("  cost budget. Those runs are not cost-matched; treat with caution.")

    print()
    print("=" * 78)
    print("PER-SEED FINAL SIMPLE REGRET")
    print("=" * 78)
    hdr = f"{'seed':<6}" + "".join(f"{m:>16}" for m in METHODS)
    print(hdr)
    for s in SEEDS:
        row = f"{s:<6}"
        for m in METHODS:
            r = next((x for x in data[m] if x["seed"] == s), None)
            row += f"{r['final_regret']:>16.4f}" if r else f"{'--':>16}"
        print(row)

    print()
    print("=" * 78)
    print("SUMMARY  (mean +/- SE, SE = std/sqrt(n))")
    print("=" * 78)
    stats = {}
    for m in METHODS:
        runs = data[m]
        if not runs:
            continue
        v = np.array([r["final_regret"] for r in runs], dtype=float)
        mean, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        n_hf = [r["n_hf_queries"] for r in runs if r.get("n_hf_queries") is not None]
        n_imp = [r["incumbent_improved_count"] for r in runs]
        wall = [r["wall_time_s"] for r in runs]
        stats[m] = (mean, se, len(v))
        print(f"  {m:<14} n={len(v):>2}  regret {mean:7.4f} +/- {se:6.4f}   "
              f"mean_n_HF={np.mean(n_hf) if n_hf else float('nan'):6.1f}  "
              f"mean_n_improved={np.mean(n_imp):5.2f}  "
              f"mean_wall={np.mean(wall):7.1f}s")

    print()
    print("=" * 78)
    print("FROZEN SUCCESS TEST: MF-DRO mean+SE  <  best-baseline mean-SE")
    print("=" * 78)
    if "MF-DRO" in stats and len(stats) > 1:
        dro_mean, dro_se, _ = stats["MF-DRO"]
        base = {m: stats[m] for m in stats if m != "MF-DRO"}
        best_m = min(base, key=lambda m: base[m][0])
        b_mean, b_se, _ = base[best_m]
        lhs, rhs = dro_mean + dro_se, b_mean - b_se
        passed = lhs < rhs
        print(f"  MF-DRO mean+SE      = {lhs:.4f}")
        print(f"  best baseline       = {best_m} (mean {b_mean:.4f})")
        print(f"  best baseline mean-SE = {rhs:.4f}")
        print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} "
              f"({lhs:.4f} {'<' if passed else '>='} {rhs:.4f})")
        print(f"\n  Locked prediction (MF-DRO mean < 1.0): "
              f"{'MET' if dro_mean < 1.0 else 'NOT MET'} (mean={dro_mean:.4f})")
        print(f"  Pre-fix MF-DRO reference was ~1.31 (NOT comparable -- leak active).")

    print()
    print("=" * 78)
    print("REGRET vs COST and vs HF-QUERY COUNT (both axes, per PROTOCOL)")
    print("=" * 78)
    for m in METHODS:
        for r in data[m][:3]:
            rc = r.get("hf_regret_curve") or r.get("regret_curve") or []
            cc = r.get("cost_curve") or []
            ft = r.get("fidelity_trace") or []
            if not rc:
                continue
            cum_hf = np.cumsum(ft) if ft else np.zeros(len(rc))
            step = max(1, len(rc) // 6)
            idx = list(range(0, len(rc), step))
            print(f"  {m} seed{r['seed']}:")
            print(f"    (cost, regret): "
                  f"{[(round(cc[i],1), round(rc[i],3)) for i in idx]}")
            print(f"    (n_HF, regret): "
                  f"{[(int(cum_hf[i]), round(rc[i],3)) for i in idx]}")


if __name__ == "__main__":
    main()
