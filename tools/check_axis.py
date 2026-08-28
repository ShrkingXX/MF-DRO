#!/usr/bin/env python3
"""Verify that result files are on a MATCHED cost axis before comparing them.

WHY
  `cost_curve` does not mean the same thing in every arm:
    src/policy/mf_dro.py        stores POST-INIT cost
    src/baselines/mf_mes_takeno stores CUMULATIVE cost (includes initial design)
  They differ by exactly the initial design -- 40 on Borehole. Reading both at
  the stored `cost_curve == 200` compares MF-DRO at 200 post-init against MF-MES
  at 160 post-init, silently handing MF-DRO 25% more budget in a COMPETITIVE
  comparison, through a key with the same name in both files.

  Unlike every other read-point error this project has hit, this one changes the
  BUDGET rather than the units, and it biases toward our own method.

WHAT IS SAFE
  h83's `sr_curve` recomputes `cost_cum - init_cost` from the query trace, so it
  places every method on the post-init axis whatever the file stores. Use it.
  This tool exists to confirm that, and to catch any arm that does not comply.

USAGE
  tools/check_axis.py <result.json> [<result.json> ...]
"""
import json, sys, os

def axis(path):
    r = json.load(open(path))
    q = r["queries"]; m = r["_meta"]
    init = sum((m["c_H"] if e["fid"] else m["c_L"]) for e in q if e.get("is_init"))
    post = [e for e in q if not e.get("is_init")]
    recomputed_end = float(post[-1]["cost_cum"]) - init if post else float("nan")
    stored = r.get("cost_curve")
    return dict(
        name=os.path.basename(path),
        recomputed=recomputed_end,                      # what sr_curve uses
        stored=(float(stored[-1]) if stored else float("nan")),
        raw=float(q[-1]["cost_cum"]),
        init=init,
        n_post_hf=sum(1 for e in post if e["fid"]),
    )

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = [axis(p) for p in sys.argv[1:]]
    print(f"{'file':46s}{'post-init end':>15s}{'stored':>10s}{'raw':>9s}{'init':>7s}{'nHF':>6s}  stored==post-init?")
    bad = False
    for r in rows:
        ok = abs(r["stored"] - r["recomputed"]) < 1e-6
        if not ok: bad = True
        print(f"{r['name'][:44]:46s}{r['recomputed']:15.2f}{r['stored']:10.2f}{r['raw']:9.2f}"
              f"{r['init']:7.1f}{r['n_post_hf']:6d}  {'yes' if ok else 'NO -- CUMULATIVE'}")
    ends = [r["recomputed"] for r in rows]
    spread = max(ends) - min(ends)
    print(f"\n  post-init axis spread across these runs: {spread:.2f}")
    if spread > 5.0:
        print("  WARNING: axis ends differ by more than overshoot; these are NOT matched budgets.")
        bad = True
    else:
        print("  Post-init axes are matched -- a paired read via sr_curve is valid.")
    if bad:
        print("\n  At least one arm stores CUMULATIVE cost. Never read its stored `cost_curve`")
        print("  against an MF-DRO arm's. Use h83 sr_curve, which rebuilds the post-init axis.")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
