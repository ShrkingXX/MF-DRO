# Findings — MF-DRO incumbent-freeze on Hartmann 6D

## Current Understanding

**The freeze is DRO-family-specific, not initialization-driven.** Freeze rate
(distinct values in the regret curve == 1) across all prior Hartmann 6D runs:

| Method | Frozen | Median distinct regret values |
|---|---|---|
| SF-DRO | **12/12 (100%)** | 1 |
| MF-GP-UCB | **10/10 (100%)** | 1 |
| MF-DRO | **9/12 (75%)** | 1 |
| MF-MI-Greedy | **0/10 (0%)** | 6 |
| Greedy-MES | 0/12 (0%) | 4 |
| KO-MES / Additive-MES / SF-MES | 0/5 each | 5–9 |

## Patterns and Insights

1. **This refutes the pre-registered leading hypothesis.** If narrow-basin
   coverage at initialization were the cause, every method drawing the same
   LHS design would freeze. MF-MI-Greedy freezes 0/10 on the same benchmark.
   A bad init does not freeze MI-Greedy, so init coverage cannot be the
   primary mechanism.

2. **SF-DRO freezes 100%.** Single-fidelity DRO freezes *more* than MF-DRO
   (75%). The pathology therefore lives in the DRO/DT component, not in the
   multi-fidelity machinery.

3. **The MES family never freezes.** Every MES-based method is at 0%.

4. **Ackley 10D shows no freeze for MF-DRO** (incumbent-improvement 7/9/8 vs
   3–4 for baselines). Combined with (1)–(3): the pathology is DRO-on-narrow-
   basin, i.e. an interaction, not a property of either alone.

## Performance gap (context for feasibility)

Final simple regret on Hartmann 6D, mean over seeds:

- MF-MI-Greedy: **0.279** (ko_mes_paper) / 0.364 (stage2) — the hard target
- Greedy-MES: 0.36–0.45
- MF-DRO: **1.31** (stage2_v3) / 1.72 (stage2)
- MF-GP-UCB: 1.99–2.46

MF-DRO already beats MF-GP-UCB. The binding target is MF-MI-Greedy, ~4x lower
regret. MF-DRO's *best single seed* (0.576) is still worse than MI-Greedy's
*mean* (0.279).

## Lessons and Constraints

- Iteration counts differ widely across prior runs (MF-MI-Greedy median 53
  iters, MF-DRO 100, MF-GP-UCB 800). Prior cross-method comparisons are not
  obviously cost-matched. Any comparison must re-verify matched real cost.
- `REVISION_LOG.md`'s Hartmann initialization claim is verified wrong
  (6.2% not 12%; 86% gate-failure rate; seed=42 at 30th percentile). Do not
  build on it.

## Open Questions

- What *mechanism* freezes the DRO incumbent? Candidates: RTG/reward collapse,
  DT action collapse, GP posterior collapse on narrow basins, acquisition
  saturation. `_diag_dt_action_collapse.py` and `_diag_sf_*` target this.
- Why does MF-GP-UCB also freeze 100% while MI-Greedy never does?
- Is the 4x gap to MI-Greedy closable within the DRO frame at all?
