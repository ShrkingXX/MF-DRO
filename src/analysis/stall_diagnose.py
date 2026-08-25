"""Why is the incumbent stalled? Attributes a stall to one of four causes using
only the tracked trace (x, y, fidelity, is_init) -- no re-runs.

The incumbent updates ONLY on HF queries, so a stall has a trivial explanation
that must be ruled out before any statement about search quality:

  HF-STARVED     ~no HF queries since the last improvement. The incumbent could
                 not have moved. This is a FIDELITY-ALLOCATION failure and says
                 nothing about where the policy searched.
  NEAR-MISS      HF queries land just below the incumbent. Right region, not
                 over the bar.
  MISDIRECTED    HF queries land far from x* AND far below the incumbent, while
                 the incumbent is comparatively close to x*. Searching the wrong
                 place.
  CLUSTERED      HF queries tightly grouped and not improving -- exploitation
                 that has converged to a non-optimal point.

Normalised distances throughout (Borehole's raw span is ~52530 on one axis).
"""
import sys, os, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")

def _bench(bench):
    # x* is NOT in the benchmark dict for Currin or Borehole -- see
    # src/analysis/optima_cache.py, which recovers it by multi-start L-BFGS-B and
    # validates against the known optimal VALUE.
    from benchmarks import get_benchmark
    from src.analysis.optima_cache import x_star
    b = get_benchmark(f"{bench}_HF")
    xs, _ = x_star(bench)
    return (np.array(b["domain_min"], float), np.array(b["domain_max"], float),
            -float(b["known_optimal_value"]), np.asarray(xs, float))

def diagnose(path, live, method="MF-DRO"):
    d = json.load(open(path)); m = d.get("_meta") or d
    if m.get("method") != method: return None
    q = [e for e in d.get("queries", []) if not e.get("is_init")]
    if len(q) < 3: return None
    lo, hi, fstar, xstar = _bench(m["bench"])
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    X = (np.array([e["x"] for e in q], float) - lo) / span
    Y = np.array([e["y"] for e in q], float); F = np.array([e["fid"] for e in q])
    best, last, bx, bv = -np.inf, -1, None, np.nan
    for i in range(len(q)):
        if F[i] == 1 and Y[i] > best: best, last, bx, bv = Y[i], i, X[i], Y[i]
    if last < 0:  # never had an HF improvement at all
        last, bx, bv = -1, None, np.nan
    S = slice(last + 1, len(q))                 # the stall window
    n_stall = len(q) - last - 1
    if n_stall == 0: return None
    Fs, Xs, Ys = F[S], X[S], Y[S]
    n_hf = int((Fs == 1).sum())
    hf_y = Ys[Fs == 1]; hf_x = Xs[Fs == 1]
    xs_n = ((xstar - lo) / span) if xstar.size else None
    def dstar(P): return np.linalg.norm(P - xs_n, axis=-1) if xs_n is not None else np.full(len(P), np.nan)
    r = dict(bench=m["bench"], seed=m["seed"], live=live, n_stall=n_stall,
             n_hf_stall=n_hf, frac_hf=n_hf / n_stall,
             inc_val=bv, inc_dstar=float(dstar(bx[None])[0]) if bx is not None and xs_n is not None else np.nan,
             hf_ygap=float(np.mean(bv - hf_y)) if n_hf and np.isfinite(bv) else np.nan,
             hf_ygap_min=float(np.min(bv - hf_y)) if n_hf and np.isfinite(bv) else np.nan,
             hf_dstar=float(np.mean(dstar(hf_x))) if n_hf else np.nan,
             hf_dinc=float(np.mean(np.linalg.norm(hf_x - bx, axis=1))) if n_hf and bx is not None else np.nan,
             spread=float(np.mean([np.linalg.norm(Xs[i]-Xs[j]) for i in range(len(Xs))
                                   for j in range(i+1, len(Xs))])) if len(Xs) > 1 else np.nan)
    # VALUE-based, not distance-based. The earlier classifier called a stall
    # MISDIRECTED when its HF queries sat farther from x* than the incumbent.
    # That test is invalid: on Hartmann the best value reachable within 0.3 of x*
    # (2.9196) equals the best beyond 1.0 away (2.8338), and in Borehole's 8D
    # domain 6000 uniform samples contain ZERO points within 0.3. See
    # src/analysis/value_reference.py. Distance to x* is kept as a reported
    # column but no longer decides anything.
    from src.analysis.value_reference import pct
    frange = abs(fstar) if fstar else 1.0
    r["hf_pct"] = float(np.mean([pct(m["bench"], v) for v in hf_y])) if n_hf else np.nan
    r["inc_pct"] = pct(m["bench"], bv) if np.isfinite(bv) else np.nan
    if n_hf <= 1:
        r["cause"] = "HF-STARVED"
    elif r["hf_ygap_min"] < 0.02 * frange:
        r["cause"] = "NEAR-MISS"
    elif np.isfinite(r["hf_pct"]) and r["hf_pct"] < 0.50:
        # Querying values worse than a coin flip against uniform sampling.
        r["cause"] = "LOW-VALUE-SEARCH"
    elif np.isfinite(r["hf_dinc"]) and r["hf_dinc"] < 0.15:
        r["cause"] = "CLUSTERED"
    else:
        r["cause"] = "PLATEAU"          # high-percentile queries, none over the bar
    return r

def main(method="MF-DRO", min_stall=5):
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(RES, f"*{method}*.json"))):
        r = diagnose(f, False, method)
        if r: rows.append(r); seen.add((r["bench"], r["seed"]))
    for f in sorted(glob.glob(os.path.join(RES, "ckpt", f"*{method}*.json"))):
        r = diagnose(f, True, method)
        if r and (r["bench"], r["seed"]) not in seen: rows.append(r)
    rows = [r for r in rows if r["n_stall"] >= min_stall]
    if not rows:
        print(f"  no {method} cell with a stall >= {min_stall} queries"); return
    rows.sort(key=lambda r: -r["n_stall"])
    print(f"  {'benchmark':<13}{'seed':>5}{'stall':>6}{'HF in':>7}{'%HF':>6}"
          f"{'best gap':>10}{'q pctile':>10}{'inc pctile':>12}{'d*(hf)':>8}  cause")
    for r in rows:
        print(f"  {r['bench']:<13}{r['seed']:>5}{r['n_stall']:>6}{r['n_hf_stall']:>7}"
              f"{r['frac_hf']:>6.0%}{r['hf_ygap_min']:>10.3f}{r['hf_pct']:>10.1%}"
              f"{r['inc_pct']:>12.1%}{r['hf_dstar']:>8.3f}  {r['cause']}")
    print("\n  best gap   = smallest (incumbent - HF query value) in the stall; ~0 = near miss")
    print("  q pctile   = mean percentile of stalled HF query VALUES vs 20k Sobol samples")
    print("               of the domain. <50% means searching worse than uniform sampling.")
    print("  inc pctile = the incumbent's own percentile -- how high the bar already is")
    print("  d*(hf)     = distance to x*, REPORTED ONLY. It does not classify: on Hartmann")
    print("               the best value within 0.3 of x* equals the best beyond 1.0.\n")
    from collections import Counter
    for c, n in Counter(r["cause"] for r in rows).most_common():
        print(f"    {c:<14} {n} cell(s)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MF-DRO",
         int(sys.argv[2]) if len(sys.argv) > 2 else 5)
