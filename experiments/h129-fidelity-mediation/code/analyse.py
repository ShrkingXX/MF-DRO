"""h129 analysis. WRITTEN AND COMMITTED BEFORE h127 PRODUCED A SINGLE RESULT FILE.

The point of committing this early is that P1 and P2 are locked numeric
predictions, so every choice that could flex them -- which seeds, which control,
which fraction (count vs cost), which read point -- is fixed in advance rather
than settled after seeing the data.

Read point: post-init, matching h83 `sr_curve` (cost_cum - init_cost).
Fraction:   post-init HF fraction BY COUNT, not by cost. The two differ
            materially (Borehole control 0.8829 by count, 0.9372 by cost) and
            every h129 number is by count.
"""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CTRL = os.path.join(ROOT, "experiments/h83-main-comparison/results/Borehole_8D__MF-DRO__seed%d.json")
Q30  = os.path.join(ROOT, "experiments/h127-roi-q30-boundary/results/Borehole_8D__ROI-Q30__seed%d.json")

P1_CENTRE, P1_HALFWIDTH = 0.808, 0.020      # locked
P2_BENEFIT              = 2.21              # locked, rel% of optimum
P2_BOUNDS               = (1.31, 4.22)      # locked: must lie between the measured doses


def hf_fraction(path):
    """Post-init HF fraction BY COUNT."""
    r = json.load(open(path))
    post = [e for e in r["queries"] if not e.get("is_init")]
    if not post:
        return None
    return sum(e["fid"] for e in post) / len(post)


def main():
    paired, missing = {}, []
    for s in range(42, 47):                  # only 42-46 have an h83 control
        if os.path.exists(Q30 % s) and os.path.exists(CTRL % s):
            paired[s] = (hf_fraction(CTRL % s), hf_fraction(Q30 % s))
        else:
            missing.append(s)

    # Report every seed, including the unpaired ones, per the reporting rule.
    unpaired = [s for s in range(47, 52) if os.path.exists(Q30 % s)]
    print(f"h127 seeds present: paired {sorted(paired)} | unpaired(no h83 control) {unpaired}")
    if missing:
        print(f"MISSING from the paired set: {missing}  <- P1/P2 are NOT evaluable until these land")
    if len(paired) < 5:
        print("\nGATE: fewer than 5 paired seeds. Reporting state only, no verdict.")
        return

    ctrl = np.array([paired[s][0] for s in sorted(paired)])
    q30  = np.array([paired[s][1] for s in sorted(paired)])
    d    = q30 - ctrl

    print(f"\n  per-seed: " + "  ".join(f"{s}:{paired[s][0]:.3f}->{paired[s][1]:.3f}" for s in sorted(paired)))
    print(f"  control {ctrl.mean():.4f}   q=0.30 {q30.mean():.4f}   shift {d.mean():+.4f}"
          f"   sd {d.std(ddof=1):.4f}   effect {abs(d.mean())/d.std(ddof=1):.2f}")

    lo, hi = P1_CENTRE - P1_HALFWIDTH, P1_CENTRE + P1_HALFWIDTH
    inside = lo <= q30.mean() <= hi
    print(f"\nP1  predicted {P1_CENTRE:.3f} +- {P1_HALFWIDTH:.3f}  ({lo:.3f}-{hi:.3f})")
    print(f"    observed  {q30.mean():.4f}  ->  {'INSIDE' if inside else 'OUTSIDE'}")
    print(f"    falsifiers: 0.883 (no effect) / 0.739 (a step, not a dose-response)")

    # The honesty check P3 taught us: is the band narrower than the data supports?
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print(f"    paired sem {sem:.4f} vs band half-width {P1_HALFWIDTH:.3f}"
          f"  -> band is {'NARROWER than the data supports' if sem > P1_HALFWIDTH else 'consistent with'}"
          f" the noise (ratio {sem/P1_HALFWIDTH:.1f}x)")
    if sem > P1_HALFWIDTH:
        print("    => report as CONSISTENT, not as CONFIRMATION, exactly as for P3.")

    print("\nP2 requires the regret benefit at q=0.30 vs the same control, rel% @cost_curve 200.")
    print(f"    predicted {P2_BENEFIT:.2f}%, and must lie within {P2_BOUNDS} to support mediation.")
    print("    NOT computed here: it needs h83's frozen sr_curve/grid, which this script")
    print("    deliberately does not reimplement -- a second implementation of the metric is")
    print("    exactly how a read-point mismatch gets introduced.")


if __name__ == "__main__":
    main()
