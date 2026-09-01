"""h139 P1 analysis. Written BEFORE the run exists.

P1 (locked): FIX2's acceptance DECLINES across the run -- last-third mean minus
first-third mean is negative, effect >= 1.0, >= 4/5 seeds. FALSIFIED if effect
< 1.0. Verified partitioned with tools/check_gate.py at registration.

The trajectory is recovered by grouping roi_stats records on `n_real_iter`, the
field h136 gates. If that field is absent the run predates the patch and this
script refuses rather than silently falling back to the run-level mean -- a
run-level mean is exactly the quantity that cannot answer P1.
"""
import json, os, sys
import numpy as np

def trajectory(path):
    r = json.load(open(path))
    # The worker aggregates roi_stats by n_real_iter into `roi_per_iter`
    # (h90/worker.py). Older runs, and any run predating the h136-gated tag,
    # carry only the run-level `roi_summary` mean -- refuse rather than fall
    # back to it, since a single number cannot answer P1.
    rp = r.get("roi_per_iter")
    if not rp or not rp.get("accept_frac"):
        rs = r.get("roi_summary") or {}
        raise SystemExit(f"REFUSING: {os.path.basename(path)} has no roi_per_iter "
                         f"(run-level accept_frac={rs.get('accept_frac')}). P1 needs the "
                         f"per-iteration array; a run-level mean cannot answer it.")
    return np.asarray(rp["n_real_iter"], float), np.asarray(rp["accept_frac"], float)

def main():
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    paths = sorted(p for p in (os.path.join(RES, f) for f in os.listdir(RES))
                   if p.endswith(".json") and "ROI-FIX2" in p)
    if not paths:
        raise SystemExit("no ROI-FIX2 results yet")
    print(f"h139 P1 -- FIX2 per-iteration acceptance, {len(paths)} run(s)\n")
    diffs = []
    for p in paths:
        it, acc = trajectory(p)
        t = len(acc) // 3
        first, last = acc[:t].mean(), acc[-t:].mean()
        diffs.append(last - first)
        print(f"  {os.path.basename(p)[:44]:46s} iters {len(acc):4d}  "
              f"first-third {first:.4f}  last-third {last:.4f}  delta {last-first:+.4f}")
    d = np.array(diffs)
    if len(d) > 1:
        sd = d.std(ddof=1); e = abs(d.mean()) / sd if sd > 0 else float("nan")
        print(f"\n  mean delta {d.mean():+.4f}  sd {sd:.4f}  effect {e:.2f}  "
              f"declining {int((d < 0).sum())}/{len(d)}")
        holds = d.mean() < 0 and e >= 1.0 and int((d < 0).sum()) >= 4
        print(f"  P1 (declines, effect>=1.0, >=4/5): {'HOLDS' if holds else 'FAILS'}")
        if d.mean() > 0 and e >= 1.0:
            print("  -> RISES separably: the analytic argument is WRONG and FIX2 is a de facto")
            print("     WIDENING schedule. Retract 'the stall is caused by late over-restriction'")
            print("     and void h133's treatment of FIX2 as a rung at q=0.21.")
    else:
        print(f"\n  single run: delta {d[0]:+.4f}. P1 needs >=4/5 seeds for a verdict;")
        print(f"  this is DIRECTIONAL EVIDENCE ONLY, not the registered test.")

if __name__ == "__main__":
    main()
