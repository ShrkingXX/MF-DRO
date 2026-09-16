"""h208 readout. Applies the criteria registered in ../protocol.md to whatever
results have landed. usage: readout.py [--csv]"""
import os, sys, json, glob
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(H, "..", "results"))
ARMS = ["ESON-L4", "L1", "L2", "L4", "L8", "L8-TRUNC"]
SEEDS = list(range(42, 52))
N_EXPECTED = 505


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, "Ackley_10D__*__seed*.json"))):
        r = json.load(open(f))
        runs[(r["arm"], r["seed"])] = r
    return runs


def main():
    runs = load()
    print(f"loaded {len(runs)}/60 runs\n")

    # SC1 -- exactly 505 evaluations, else the base loop skipped iterations silently
    bad = [(k, r["n_eval"]) for k, r in runs.items() if r["n_eval"] != N_EXPECTED]
    print(f"SC1 n_eval==505: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    valid = {k: r for k, r in runs.items() if r["n_eval"] == N_EXPECTED}

    # SC2 -- realised rollout length
    print("SC2 realised rollout length (mean/min/max over all rollouts of the run):")
    for a in ARMS:
        rs = [r for (arm, s), r in valid.items() if arm == a]
        if not rs:
            continue
        mn = np.mean([r["realised_len_mean"] for r in rs])
        lo = min(r["realised_len_min"] for r in rs); hi = max(r["realised_len_max"] for r in rs)
        L = rs[0]["rollout_length"]; es = rs[0]["early_stop"]
        ok = (not es and lo == hi == L) or es
        print(f"   {a:9s} L={L} early_stop={es!s:5s} mean={mn:5.2f} min={lo} max={hi}  {'ok' if ok else 'MISMATCH'}")

    # SC3 -- L8-TRUNC's rtg0 vs L8's at iteration 0 (same simulator, same seed => identical)
    sc3 = []
    for s in SEEDS:
        a, b = valid.get(("L8", s)), valid.get(("L8-TRUNC", s))
        if a and b:
            sc3.append(abs(a["diag"][0]["rtg0_mean"] - b["diag"][0]["rtg0_mean"]))
    if sc3:
        print(f"SC3 |rtg0(L8) - rtg0(L8-TRUNC)| at iter 0, max over {len(sc3)} seeds: {max(sc3):.2e}  {'PASS' if max(sc3) < 1e-9 else 'FAIL'}")

    # SC4 -- identical initial design across arms within a seed
    sc4_ok = True
    for s in SEEDS:
        rs = [r for (arm, ss), r in valid.items() if ss == s]
        if len(rs) < 2:
            continue
        x0 = np.array(rs[0]["all_x"][:5]); y0 = np.array(rs[0]["all_y"][:5])
        for r in rs[1:]:
            if not (np.allclose(np.array(r["all_x"][:5]), x0) and np.allclose(np.array(r["all_y"][:5]), y0)):
                sc4_ok = False; print(f"   SC4 MISMATCH seed {s}: {r['arm']} vs {rs[0]['arm']}")
    print(f"SC4 identical initial design within seed: {'PASS' if sc4_ok else 'FAIL'}\n")

    # ---- main table: final TRUE simple regret at 505 evaluations ----
    print("Final simple regret (f_true at the observed incumbent, 505 evals). n = seeds landed.")
    print(f"{'arm':9s} {'n':>2s} {'mean':>7s} {'se':>6s} {'sd':>6s} {'median':>7s} {'wall(m)':>8s}  per-seed (42..51)")
    tab = {}
    for a in ARMS:
        vals = []; walls = []
        for s in SEEDS:
            r = valid.get((a, s))
            vals.append(r["final_regret"] if r else np.nan)
            if r: walls.append(r["_wall_s"] / 60)
        v = np.array(vals); ok = ~np.isnan(v); tab[a] = v
        if ok.sum() == 0:
            print(f"{a:9s}  0"); continue
        print(f"{a:9s} {ok.sum():2d} {np.nanmean(v):7.3f} {np.nanstd(v, ddof=1)/np.sqrt(ok.sum()) if ok.sum()>1 else np.nan:6.3f} "
              f"{np.nanstd(v, ddof=1) if ok.sum()>1 else np.nan:6.3f} {np.nanmedian(v):7.3f} {np.mean(walls):8.1f}  "
              + " ".join(f"{x:6.2f}" if not np.isnan(x) else "   -- " for x in v))

    # ---- registered paired contrasts ----
    def paired(a, b):
        d = tab[a] - tab[b]; d = d[~np.isnan(d)]
        if len(d) < 2:
            return None
        m, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(len(d))
        return dict(n=len(d), mean=m, sd=sd, se=se, d=m / sd if sd > 0 else np.nan,
                    n_a_better=int((d < 0).sum()), within2se=abs(m) < 2 * se)

    print("\nPaired contrasts (first - second; negative = first arm has LOWER regret):")
    contrasts = [("L8", "L1"), ("L8", "L8-TRUNC"), ("L8-TRUNC", "L1"), ("L4", "L1"), ("L2", "L1"), ("ESON-L4", "L4"), ("ESON-L4", "L1")]
    P = {}
    for a, b in contrasts:
        p = paired(a, b); P[(a, b)] = p
        if p is None:
            print(f"   {a:>9s} - {b:<9s}: (no pairs yet)"); continue
        print(f"   {a:>9s} - {b:<9s}: n={p['n']:2d} mean={p['mean']:+7.3f} sd={p['sd']:6.3f} se={p['se']:5.3f} "
              f"d={p['d']:+5.2f} {a} better on {p['n_a_better']}/{p['n']}  |mean|<2se: {p['within2se']}")

    # ---- verdict per protocol ----
    p81, p8t = P.get(("L8", "L1")), P.get(("L8", "L8-TRUNC"))
    if p81 and p8t:
        n81 = p81["within2se"]; n8t = p8t["within2se"]
        if n81 and n8t:
            v = "NULL -- P1 holds: the dose is flat. Later-step rollout data does not move the final regret."
        elif (not n81) and p81["mean"] < 0 and (not n8t) and p8t["mean"] < 0:
            v = "DATA-MATTERS -- L8 beats both L1 and L8-TRUNC by >2se: positions 1-7 as training data improve the query."
        elif (not n81) and n8t:
            v = "LABEL-ONLY -- L8 differs from L1 but not from L8-TRUNC: the RTG label horizon matters, the later-step data does not."
        else:
            v = "OTHER pattern -- report as is (see contrasts above)."
        print(f"\nVERDICT ({p81['n']} paired seeds): {v}")
    else:
        print("\nVERDICT: not enough paired runs yet.")

    # ---- P2 wall-clock ----
    print("\nP2 wall-clock (min, mean over landed seeds):")
    for a in ARMS:
        w = [r["_wall_s"] / 60 for (arm, s), r in valid.items() if arm == a]
        if w:
            print(f"   {a:9s} {np.mean(w):6.1f}  (n={len(w)})")

    if "--csv" in sys.argv:
        out = os.path.join(RES, "summary.csv")
        with open(out, "w") as f:
            f.write("arm,seed,final_regret_true,final_regret_obs,realised_len_mean,wall_min\n")
            for (a, s), r in sorted(valid.items()):
                f.write(f"{a},{s},{r['final_regret']:.6f},{r['final_regret_obs']:.6f},{r['realised_len_mean']:.3f},{r['_wall_s']/60:.2f}\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
