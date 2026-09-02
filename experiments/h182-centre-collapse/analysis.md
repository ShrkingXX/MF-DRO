# h182 — the failure is COLLAPSE TO THE BOX CENTRE, and it explains the benchmark asymmetry

**EXPLORATORY.** No new runs; re-analysis of saved traces. Not pre-registered.
Statistic: over each run's **last 20 HF queries**, mean distance (unit box) to the
box centre, and mean pairwise dispersion. Frozen rel% as recorded.

## Hartmann — graded, and distance-to-x* is ANTI-monotone

| arm | rel% | to x* | **to CENTRE** | dispersion |
|---|---|---|---|---|
| control | 7.99 | 0.814 | **0.610** | 0.343 |
| UCB-LOC | 10.58 | 0.852 | **0.597** | 0.350 |
| HEAD-MES | 25.16 | 0.619 | **0.546** | 0.289 |
| TAIL-MES | 46.45 | 0.600 | **0.149** | 0.216 |
| ORACLE | 52.23 | 0.572 | **0.142** | 0.195 |
| RANDOM | 65.14 | 0.582 | **0.086** | 0.120 |

**Distance to the centre is perfectly rank-monotone with performance across all
six arms.** Dispersion is too (one negligible control/UCB inversion, matching
their near-tied regret).

**Distance to x\* runs the wrong way** — the worst arms look "closest" to the
optimum. The geometry explains it: `||centre − x*|| = 0.5681`, so an arm collapsed
*at the centre* scores ≈0.57 while converging nowhere. RANDOM scores 0.582.

> **Diagnostic caution for the codebase.** `query_dist_to_xstar_per_iter` is
> anti-monotone with performance on these arms and must not be read alone. It
> cannot distinguish "converged near the optimum" from "collapsed at the centre,
> which happens to sit 0.57 from the optimum". (This does *not* retroactively
> overturn the earlier second-basin observation, which was made on different arms
> whose queries sat 0.97–1.10 from x* — that pattern is not centre-collapse.)

## Borehole — BIMODAL, with a clean gap and no overlap

| arm | rel% | to CENTRE | dispersion |
|---|---|---|---|
| L=1 | 13.69 | 0.864 | 0.181 |
| UCB-LOC | 15.13 | 0.847 | 0.116 |
| control | 15.82 | 0.852 | 0.125 |
| **HEAD-MES** | **16.96** | **0.812** | 0.126 |
| EXPLOIT-LOC | 19.07 | 0.814 | 0.108 |
| MES-FROZEN | 19.36 | 0.803 | 0.116 |
| TAIL-MES | 43.94 | 0.093 | 0.125 |
| DIVERSE-GOOD | 43.94 | 0.107 | 0.135 |
| ORACLE | 43.94 | 0.102 | 0.129 |
| RANDOM | 43.94 | 0.070 | 0.091 |

Working arms occupy **0.803–0.864**; failing arms **0.070–0.107**. Nothing lies
between. **10/10 arms fit, with a gap 0.70 wide.** Dispersion does *not* separate
them here (0.108–0.181 vs 0.091–0.135, overlapping) — on Borehole the sole
discriminator is escape from the centre.

## This answers the asymmetry that has been open longest

The front's answer has a WEAK form on both benchmarks and a STRONG form only on
Borehole, and why has been recorded as unexplained since h175. The two tables
give it:

- **Borehole: escaping the centre is necessary AND sufficient.** HEAD-MES escapes
  (0.812, inside the working band) and performs at control level (16.96 vs 15.82).
  One acquisition-chosen first step is enough to leave the centre in a usable
  direction — so only the first step matters. **The strong form.**
- **Hartmann: escaping the centre is necessary but NOT sufficient.** HEAD-MES
  escapes (0.546, on the working side of Hartmann's gap) and still loses 3×
  (25.16 vs 7.99). Escape is partial and performance is partial; there is no
  cliff. **The weak form only.**

Why the benchmarks differ this way is consistent with their geometry: Hartmann-6
is multimodal (a second basin sits 1.1027 away and is already documented here),
so leaving the centre does not determine *which* direction; Borehole is dominated
by a few monotone-sensitive dimensions whose optima are at the boundary (already
recorded), so leaving the centre in the acquisition-chosen direction is most of
the problem.

## What could RETRACT this

- **A Borehole arm that escapes the centre (>0.8) and still fails, or one that
  stays at the centre (<0.11) and succeeds.** Either breaks the necessary-and-
  sufficient claim. Currently 10/10 arms fit with no overlap.
- The geometric reading of *why* the benchmarks differ is an interpretation of
  two independently-recorded facts (Hartmann multimodality, Borehole boundary
  optima), not something this analysis measured. The measured claim is the
  necessary/sufficient split; the geometry is the proposed reason for it.
- Last-20-HF-queries is one window choice, made once and not tuned.
