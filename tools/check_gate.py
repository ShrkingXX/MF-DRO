#!/usr/bin/env python3
"""Check that a registered PASS condition and FALSIFIER partition the outcome space.

WHY THIS EXISTS
  Three bar-design failures in one day, none caught in advance:
    h131 P1  pass: effect < 0.5      falsify: effect >= 1.0   -> 0.86 had no verdict
    h134 P4  pass: 3/3 arms          falsify: >=2/3 wrong     -> 1/3 wrong had none
    h132 P4  pass invoked a bar in regret points against a normalised score --
             a clause that could never fire, i.e. a pass condition covering
             everything.
  Each time the two clauses were written minutes apart and never compared. The
  rule is NOT "write a falsifier". It is that pass and falsifier must PARTITION:
  every possible outcome must map to exactly one registered verdict. Otherwise
  some results have no verdict and whatever is said about them is authored after
  seeing them, which is what pre-registration exists to prevent.

USAGE
  tools/check_gate.py --stat "effect size" --pass "<0.5" --falsify ">=1.0"
  tools/check_gate.py --stat "arms in direction" --pass ">=3" --falsify "<=1" --int
"""
import argparse, re, sys

OPS = {">=": "ge", ">": "gt", "<=": "le", "<": "lt"}

def parse(c):
    m = re.fullmatch(r"\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*", c)
    if not m:
        sys.exit(f"cannot parse condition {c!r}; use e.g. '>=1.0' or '<0.5'")
    return m.group(1), float(m.group(2))

def covers(op, v, x):
    return {">=": x >= v, ">": x > v, "<=": x <= v, "<": x < v}[op]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stat", required=True, help="name of the statistic, for the message")
    p.add_argument("--pass", dest="p", required=True, help="e.g. '<0.5'")
    p.add_argument("--falsify", dest="f", required=True, help="e.g. '>=1.0'")
    p.add_argument("--int", action="store_true", help="statistic is integer-valued")
    p.add_argument("--calibrated-by", dest="cal", default=None,
                   help="the COMMAND that computed the threshold's calibrating value, "
                        "not the value. Peer's rule: a bar carrying its code makes "
                        "applying it to a different quantity a diff, not a judgement call.")
    a = p.parse_args()
    if not a.cal:
        print("WARNING: no --calibrated-by given.\n"
              "  Three of today's bar failures came from calibrating a threshold on one\n"
              "  QUANTITY and registering it against another (h134 P1: 0.40 from\n"
              "  first-iter-vs-last-iter, registered against first-third-vs-last-third\n"
              "  means, which differ by an order of magnitude). Record the command that\n"
              "  produced the calibrating number so the mismatch is a diff, not a\n"
              "  judgement call.\n")
    po, pv = parse(a.p); fo, fv = parse(a.f)

    # Probe the boundary region densely for outcomes covered by neither / both.
    lo, hi = min(pv, fv), max(pv, fv)
    step = 1 if a.int else (hi - lo) / 200 or 0.01
    xs, x = [], lo - step
    while x <= hi + step:
        xs.append(round(x, 6)); x += step
    xs += [lo - 10 * step, hi + 10 * step]

    gap  = [x for x in xs if not covers(po, pv, x) and not covers(fo, fv, x)]
    both = [x for x in xs if covers(po, pv, x) and covers(fo, fv, x)]

    print(f"statistic : {a.stat}")
    print(f"PASS      : {a.stat} {a.p}")
    print(f"FALSIFIED : {a.stat} {a.f}")
    bad = False
    if both:
        bad = True
        print(f"\nCONTRADICTION: values satisfy BOTH, e.g. {both[0]} .. {both[-1]}")
    if gap:
        bad = True
        g = sorted(gap)
        print(f"\nGAP: no registered verdict for {a.stat} in [{g[0]}, {g[-1]}]")
        print("     A result landing here would be interpreted AFTER seeing it.")
        print("     Fix: widen the pass condition, widen the falsifier, or register")
        print("     the middle band explicitly as INDETERMINATE before running.")
    if not bad:
        print("\nOK -- pass and falsifier partition the outcome space.")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
