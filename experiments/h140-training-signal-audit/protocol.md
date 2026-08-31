# h140 — WHY is the training signal bad on low-HF-budget benchmarks?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY, and it opens a new research direction.
STATISTIC: the per-iteration training diagnostics already stored in every run,
compared **within-run as first-third vs last-third fractional change**, then
contrasted **between benchmarks**, paired over seeds 42-46, control arm only.
Fractional-within-run because the diagnostics are not comparable across
benchmarks in absolute terms — the same scale error h134 avoided.

## The direction this opens

MF-DRO's measured deficits are: it loses to MF-MES on regret (8/10), on the
diagnosis's own query-quality metric (9/10), and on wall-clock (16-29x). Every
ROI result is Borehole-only. **h134 established the likely reason the ROI cannot
help elsewhere: on Hartmann the DT's `L_loc` nearly DOUBLES across a run (4/5
seeds) while Borehole's is flat.** The head is losing ground against its own
training target.

h134 could not separate two explanations, and said so before looking:
**(a) the head cannot learn**, or **(b) the target moves faster than the head
follows.** They imply different fixes — (a) wants capacity or optimisation
changes, (b) wants target stabilisation (replay buffers, EMA targets, confidence
weighting). Choosing a heuristic before separating them is guessing.

The direct instrument would be `actions_x_var_per_iter` (teacher-action variance,
h117's patch). **It is NOT serialised into any stored result** — the patch appends
it in memory only. Verified. So this audit uses what IS stored.

## Predictions (locked)

Hartmann vs Borehole, control arm, seeds 42-46, fractional within-run change:

**P1.** `grad_coherency_per_iter` degrades more on Hartmann than Borehole, between-
benchmark effect >= 1.0. FALSIFIED if < 1.0. (Partitioned; calibrated by h134's
own between-benchmark contrast of 1.10 on the same seeds.)

**P2.** `action_reward_corr_per_iter` — the correlation between the DT's actions
and the reward they earn — is lower on Hartmann, effect >= 1.0. This is the
sharpest available test of whether the *signal* is informative at all, as opposed
to the head being unable to use it.

**P3 (NO direction registered).** `rtg_gpbelief_corr`, `neg_rtg_frac`,
`p_pred_inference`, `rtg_frac_between_traj_var`, `L_fid` — reported whatever they
show. I have no grounds for a sign on any of them and will not manufacture one.

## What this could RETRACT

1. **"Hartmann's problem is a training-signal problem"** — my own recommendation,
   made to the user, that the direction for improving MF-DRO is the training
   signal rather than the region heuristic. If P1 and P2 both fail and no P3
   diagnostic separates, then h134's `L_loc` rise stands alone and unexplained,
   and I have recommended a direction on a single statistic.
2. **h134's disjunction as actionable.** If the diagnostics cannot distinguish (a)
   from (b) either, then the honest position is that we do not know why, and the
   next step is instrumentation — serialising `actions_x_var_per_iter` — not a
   heuristic.

## Stated before looking

Five diagnostics with no registered direction (P3) is a multiplicity risk: with
enough of them one will look interesting. **P3 findings are EXPLORATORY and cannot
support the direction on their own** — only P1 or P2, which have locked
directions, can. Recording this so a P3 hit cannot later be presented as if it had
been predicted.
