# h218 — Is it coverage, or is it spread? Two arms that separate the last standing accounts

Registered BEFORE any h218 run. CONFIRMATORY. One new knob (`action_jitter`, default 0.0 =
bit-identical).

## The two questions

### (1) Why does all-random fail — coverage, or the conditional mean?

The user's hypothesis: too few random rollouts to cover the high-return region.
The competing account (`why-full-random-failed.md`): under MSE the DT learns
**E[action | state]**, and a teacher drawing actions *independently of state* has a
state-independent conditional mean — the pool's centroid — **no matter how many rollouts
are drawn**. More data converges faster to the same constant. h149's traces show exactly
that: query cloud 0.024 from the box centre, dispersion 0.090, *identical on two
benchmarks*.

These make opposite dose predictions, so a dose settles it.

**R-ROI-120**: all-random from the ROI pool, **120** rollouts/member, against h214's
**R-ROI** at 20/member.
- **coverage** ⇒ R-ROI-120 clearly better than R-ROI
- **conditional mean** ⇒ the two are the same, both emitting ≈ the ROI centroid

What makes the ROI version interesting at all: the h214 smoke measured the ROI pool's
random actions at **|mean − 0.5| = 0.47** — the ROI centroid is far from the box centre and
moves with the model's belief, unlike h149's uniform pool.

### (2) Is MIXR's Borehole gain just spread in the regression target?

Ruled out so far: data volume (h209 NIR120), boundary geometry (h210 L1 + MIXR-P72
dims-on-face), fidelity composition (h210 MIXR-P72), RTG conditioning (audit: the channel
is a constant input), the window (h211 P-B). What has never been isolated is the plainest
possibility — that mixing in a spread-out half simply **widens the distribution of action
targets** and acts as a regularizer.

**MES+jitter** tests it with nothing else attached. Take the ordinary MES rollouts and add
Gaussian noise to their **recorded action targets** only: states, fidelities, outcomes,
RTG and rollout count all unchanged, no random rollouts at all. Jitter is drawn from a
**dedicated generator**, so the global RNG stream is untouched (h213 measured how easily
one extra draw desynchronises two arms).

## Arms — 15 runs, Borehole 42–46, CTRL-K1 base (K=1, `mes_entropy`, ROI-Q10)

| arm | config |
|---|---|
| **RROI120** | `rollout_mix=[('random',120,None)]` — all-random, 6x h214's R-ROI |
| **JIT10** | plain CTRL-K1 + `action_jitter=0.10` (unit coords) |
| **JIT25** | plain CTRL-K1 + `action_jitter=0.25` |

References, same seeds: CTRL-K1 **11.59**, MIXR-K1 **7.17**, R-ROI (h214, pending),
h149 all-random-uniform **43.94** (saturation floor).

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-COV:** |R-ROI-120 − R-ROI| ≤ 1.26 — the dose does nothing, because the conditional
  mean does not depend on sample count. **If R-ROI-120 beats R-ROI by > 1.26, coverage is
  real and the conditional-mean account in `why-full-random-failed.md` is RETRACTED.**
- **P-SPREAD:** at least one jitter level reaches within the band of MIXR-K1 (7.17) and
  beats CTRL-K1 (11.59) by > 1.26 on ≥ 4/5. **Then MIXR's mechanism is target spread** —
  the first positive identification after five eliminations — and the method simplifies to
  "add noise to the teacher's action labels", needing no random rollouts at all.
- **If both jitter levels stay within the band of CTRL-K1**, spread is not the mechanism
  either, and what the random half contributes is something only *new rollouts* can supply
  (different states visited, different outcomes) — which points the next arm at state
  coverage rather than action spread.
- **Ordering, so it can be wrong:** JIT25 < JIT10 < CTRL-K1, and R-ROI-120 ≈ R-ROI.

## Diagnostic recorded at readout

Action-target dispersion of the training batch per arm, so the jitter's match to MIXR's
mixture is **measured, not assumed** — the h214 smoke gives the random half's tau=0 action
spread at 0.68–0.69 for comparison. If neither jitter level reaches MIXR's mixture
dispersion, P-SPREAD is under-dosed and that is reported rather than read as a null.

## What each outcome RETRACTS

- **P-COV fails** → the conditional-mean account of h149's collapse, and with it the claim
  that arm R was dropped for the wrong reason.
- **P-SPREAD succeeds** → retracts nothing, but supersedes MIXR: the simpler intervention
  wins and the random half is incidental.
- **P-SPREAD fails at both doses** → target spread joins the eliminated list.

## Compute

15 runs, cap 15, queued behind h214 and h217. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
