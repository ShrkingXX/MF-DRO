# h182 — the failure is COLLAPSE TO THE BOX CENTRE

**EXPLORATORY.** No new runs; re-analysis of saved traces. Not pre-registered.
Statistic: over each run's **last 20 HF queries**, mean distance (unit box) to the
box centre. Frozen rel% throughout. Arms keyed by tag **and** experiment (never
merged across experiments — the same tag in two experiments is two configurations).

## CORRECTION to this file's first version

The first version reported Borehole as **bimodal** — working arms at 0.803–0.864,
failing arms at 0.070–0.107, "a gap 0.70 wide, 10/10 arms fit." That was measured
on **10 hand-picked teacher arms**. Re-run over **every** Borehole arm with ≥4
matched seeds, the gap fills in: MF-DRO [h75] at 0.752, POOL1000 [h78] at 0.778,
STALE-PATH at 0.798 land inside it, with intermediate regrets (24.25, 22.28,
19.53). **The bimodality was an artifact of which arms I sampled.** The real
structure is a continuum, and the corrected claim below is the stronger one.

## Borehole — monotone across 28 MF-DRO arms

| population | n | Spearman ρ(distance from centre, rel%) |
|---|---|---|
| **MF-DRO arms only** | **28** | **−0.967** |
| GP baselines | 4 | −0.800 |
| all arms pooled | 32 | −0.918 |

Across 28 MF-DRO arms — spanning teachers, ROI variants, rollout lengths, loss
functions and conditioning changes — the further the policy's queries sit from
the box centre, the better it does, near-monotonically. Largest departures from
the trend are small (+4.8, +3.8, −2.8 rel% pts).

*(ρ is a descriptive rank statistic over arms, each itself a 5-seed mean. No
p-value is attached; the n=5 rule concerns seeds within an arm.)*

Endpoints: RANDOM-POOL 0.070 → 43.94; MF-MES 1.093 → 6.40.

### The one counterexample is out of population

**MF-MI-Greedy** escapes the centre (0.904) and still performs poorly (28.44) —
the only arm badly off-trend. It is a **GP baseline with no Decision Transformer
at all**, so it cannot bear on a claim about what a DT emits. Naming it rather
than dropping it: escaping the centre is clearly not sufficient *in general*, only
within the DT-based arms this analysis is about.

## Hartmann — the same ordering, and distance-to-x* runs BACKWARDS

| arm | rel% | to x* | **to CENTRE** | dispersion |
|---|---|---|---|---|
| control | 7.99 | 0.814 | **0.610** | 0.343 |
| UCB-LOC | 10.58 | 0.852 | **0.597** | 0.350 |
| HEAD-MES | 25.16 | 0.619 | **0.546** | 0.289 |
| TAIL-MES | 46.45 | 0.600 | **0.149** | 0.216 |
| ORACLE | 52.23 | 0.572 | **0.142** | 0.195 |
| RANDOM | 65.14 | 0.582 | **0.086** | 0.120 |

Distance to the centre is perfectly rank-monotone with performance. **Distance to
x\* is anti-monotone** — the worst arms look "closest" to the optimum, because
`||centre − x*|| = 0.5681` and an arm collapsed *at* the centre scores ≈0.57 while
converging nowhere. RANDOM scores 0.582.

> **Diagnostic caution for the codebase.** `query_dist_to_xstar_per_iter` is
> anti-monotone with performance on these arms and must not be read alone: it
> cannot separate "converged near the optimum" from "collapsed at the centre".
> This does **not** overturn the earlier second-basin observation, which was made
> on different arms whose queries sat 0.97–1.10 from x* — not centre-collapse.

## What this does NOT explain: the benchmark asymmetry

I initially wrote that escape from the centre is "necessary and sufficient on
Borehole, necessary but not sufficient on Hartmann", and that this explained why
the front's answer has a strong form only on Borehole. **The numbers do not
support that.**

| | HEAD's escape ÷ control's | HEAD's regret ÷ control's |
|---|---|---|
| Borehole | 0.812 / 0.852 = **0.95** | 16.96 / 15.82 = **1.07×** |
| Hartmann | 0.546 / 0.610 = **0.90** | 25.16 / 7.99 = **3.15×** |

HEAD retains a **similar fraction** of the control's escape on both benchmarks —
95% and 90% — yet costs 1.07× on one and 3.15× on the other. Escape distance
therefore orders arms *within* a benchmark extremely well and does **not** by
itself explain the difference *between* them. **The asymmetry remains open**, as
it has been since h175.

## What could RETRACT this

- An MF-DRO arm far from the centre that fails, or one at the centre that
  succeeds. Currently 28 arms, ρ = −0.967, no serious outlier.
- The direction of causation is untested. Centre-collapse and poor regret could
  both follow from a third cause; nothing here intervenes on distance directly.
- Last-20-HF-queries is one window, chosen once and not tuned.
