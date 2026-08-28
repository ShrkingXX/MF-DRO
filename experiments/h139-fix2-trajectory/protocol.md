# h139 — Which way does a FIXED beta move the acceptance rate? Three of my beliefs disagree.

STATUS: LOCKED before the data exists. The per-iteration `accept_frac` logging is
patched but ungated (h136 queued) and no run has produced the array.
TYPE: CONFIRMATORY.
STATISTIC: per-iteration realized acceptance from the tagged `roi_stats` records,
grouped by `n_real_iter`. Borehole ROI-FIX2. Compared as **last-third mean minus
first-third mean**, paired within seed.

## Why: I hold three beliefs about FIX2 and they cannot all be true

1. **Analytic.** The ROI is `{x | mu+beta*sigma >= max(mu-beta*sigma)}`. Early,
   sigma is large, UCB is high everywhere and max LCB is low, so the set is
   **wide**. As sigma shrinks the set contracts toward `{x | mu >= mu*}`. **A
   fixed beta should therefore LOOSEN early and TIGHTEN late.**
2. **h133 (measured).** FIX2 does **not** stall late — it recovers 24.34% of
   midpoint-available regret, indistinguishable from the loose arms and far above
   q=0.10's 10.51%. If FIX2 were tight late it should stall like q=0.10.
3. **Wall-clock (measured).** FIX2 costs 137.4 min, **more** than q=0.10's 117.4,
   which the peer and I both read as cost tracking its tightest moments.

(1) and (3) agree that FIX2 gets tight somewhere. (2) says it is not tight late.
The only reconciliation is that FIX2 is tight **early** and loose **late** — the
opposite of the analytic argument, and it would make FIX2 a **de facto widening
schedule**, i.e. the paper's `beta_t` direction already implemented by accident.

## Prediction (locked)

**P1.** FIX2's acceptance **declines** across the run: last-third mean minus
first-third mean is negative, effect >= 1.0, >= 4/5 seeds. **FALSIFIED if effect
< 1.0** (partitioned; verified with `check_gate.py`). If the sign is positive and
separable, that is falsification in the strongest form.

I predict P1 **holds**, on the analytic argument — which means I am predicting the
outcome that leaves belief (2) unexplained.

## What this RETRACTS, whichever way it goes

- **P1 holds (declines).** Then belief (2) is unexplained: an arm that tightens
  late does not stall, which contradicts h133's account that late tightness causes
  the stall. **I would have to withdraw "the stall is caused by late
  over-restriction"** — the motivation I handed the peer for h123 and the entire
  premise of my h132.
- **P1 fails (rises).** Then the analytic argument is wrong, FIX2 is an implicit
  widening schedule, and **h133's placement of FIX2 as a "rung at q=0.21" is void**
  — I already flagged that reading as an error and this would confirm it. It would
  also mean the best arm on record is already a schedule, which makes h123's
  comparator question live again on evidence rather than on my say-so.

**Either way one of these dies.** Registering that now, because after the fact
either outcome could be narrated as consistent with the rest.

## Note

This does not require h136 to pass — h136 gates whether the logging perturbs a
run, which must be settled before any *result* from the logged arm is used. But
the trajectory question needs a fresh FIX2 run either way, which is not scheduled
and is not competing for slots today.
