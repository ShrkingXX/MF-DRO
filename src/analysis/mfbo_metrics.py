"""Metrics for multi-fidelity BO comparison, following the MFBO literature.

- Simple regret (SR)      : Takeno et al. 2020 -- f* - best HF value QUERIED
- Inference regret (IR)   : Takeno et al. 2020 -- f* - f(argmax mu_H), the
                            model's RECOMMENDATION. Falls when the surrogate
                            believes the right thing, not only when a good point
                            is evaluated. Clamped to <= SR per their convention.
- Cost-aligned traces     : regret as a step function of cumulative cost, so
                            methods spending different amounts per query are
                            comparable (Best Practices, Nat Comput Sci 2025).
- Discount Delta          : budget MFBO needs to reach a target regret, relative
                            to a reference method. Delta > 0 means cheaper.
"""
import json
import numpy as np

def load_run(path):
    """Load one run.

    GRANULARITY WARNING (verified, not assumed): the baselines log one
    regret/cost entry per *round*, where a round bundles several queries --
    MF-MI-Greedy on Currin produced cost_curve [3,9,15,21] (deltas 3,6,6,6 =
    1 HF + 3 LF per round) alongside a 13-entry fidelity_trace. MF-DRO logs one
    entry per *query*, so its traces are 1:1.

    Consequences handled here:
      * cost_curve is post-init cumulative for BOTH families (checked: the
        baseline starts at 3.0, not at the 30.0 initial-design cost), so
        cost-alignment is valid across methods.
      * fidelity_trace must NOT be truncated to the regret length -- that would
        silently keep only the first few sub-steps and corrupt any HF-share
        statistic. It is kept whole and reported separately.
    """
    d = json.load(open(path))
    sr = np.asarray(d.get("hf_regret_curve") or d.get("regret_curve"), float)
    ir = d.get("inference_regret_curve")
    ir = np.asarray(ir, float) if ir else np.full(len(sr), np.nan)
    cost = np.asarray(d["cost_curve"], float)          # post-init cumulative
    fid = np.asarray(d.get("fidelity_trace", []))      # per QUERY, not per round
    n = min(len(sr), len(cost))
    if len(ir) < n:
        ir = np.full(n, np.nan)
    return dict(sr=sr[:n], ir=ir[:n], cost=cost[:n],
                fid=fid,                                # full, untruncated
                n_queries=int(len(fid)), n_logged=int(n),
                hf_share=(float((fid == 1).mean()) if len(fid) else float("nan")))

def align_to_cost(values, cost, grid):
    """Step-interpolate a best-so-far trace onto a common cost grid.

    Before the first query the trace is undefined; we hold the first value,
    which is conservative (it cannot make a method look better than it is).
    """
    out = np.empty(len(grid), float)
    for i, g in enumerate(grid):
        j = np.searchsorted(cost, g, side="right") - 1
        out[i] = values[j] if j >= 0 else values[0]
    return out

def cost_grid(budget, n=401):
    return np.linspace(0.0, float(budget), n)

def auc(traces, grid):
    """Cost-weighted mean regret: area under regret-vs-cost / budget."""
    return np.trapz(traces, grid, axis=-1) / (grid[-1] - grid[0])

def budget_to_target(values, cost, target):
    """Cheapest cumulative cost at which regret first reaches `target`.
    Returns inf if never reached -- callers must handle censoring explicitly
    rather than silently dropping those runs."""
    hit = np.nonzero(values <= target)[0]
    return float(cost[hit[0]]) if len(hit) else float("inf")

def discount(method_runs, reference_runs, target):
    """Delta = 1 - budget(method)/budget(reference) at a shared target regret.

    Delta > 0  : method reaches the target more cheaply (the MFBO claim).
    Returns (delta, n_used, n_censored) -- censored runs (target never reached
    by either arm) are EXCLUDED and counted, never imputed.
    """
    ds, cens = [], 0
    for m, r in zip(method_runs, reference_runs):
        bm = budget_to_target(m["sr"], m["cost"], target)
        br = budget_to_target(r["sr"], r["cost"], target)
        if not np.isfinite(bm) or not np.isfinite(br) or br <= 0:
            cens += 1
            continue
        ds.append(1.0 - bm / br)
    return (float(np.mean(ds)) if ds else float("nan")), len(ds), cens
