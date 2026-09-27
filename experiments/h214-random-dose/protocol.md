# h214 — All-random with the ROI pool, and a dose on the number of random rollouts

Registered BEFORE any h214 run. CONFIRMATORY. No core change: every arm is a
`rollout_mix` setting on the existing CTRL-K1 base.

## Two things this settles

**(1) The all-random schema, with the pool it should have had.** h149 ran
`rollout_policy="random"` but left `use_roi` at its **False** default, so the random
teacher drew **uniformly over the whole box**. Under THE_ANSWER (the MSE location head
emits the conditional *mean* of its training actions), that mean is the box centre — and
the traces show exactly that: query cloud **0.024** from the box centre, dispersion
**0.090**, *identical on two benchmarks*, never improving on the initial design (43.94 =
saturation floor). See `experiments/h211-mixr-on-k1/why-full-random-failed.md`.

With `use_roi=True` the random teacher draws from the ROI-filtered pool, so its action mean
is the **ROI centroid** — a point that tracks the model's own belief and tightens with it.
MIXR's random half has always used this pool; what has never been run is the pool **without
the MES half**, which isolates the random half's whole contribution.

**(2) Does more random data help?** Random rollouts are measurably cheaper:
h207 SC7 gives **1.94 s per MES rollout vs 0.96 s per random rollout**, so random buys ~2x
the trajectories per second. The hypothesis under test is that performance improves with
more of them.

## Arms — 15 runs, Borehole 42–46, all on the CTRL-K1 base (K=1, pos-emb ON, `mes_entropy`, ROI-Q10)

| arm | `rollout_mix` per member | rollouts/member | rel. rollout cost |
|---|---|---|---|
| **R-ROI** | `[('random', 20, None)]` | 20 random | **0.5x** |
| **D60** | `[('mes',20,None),('random',60,None)]` | 20 + 60 | 2.5x |
| **D120** | `[('mes',20,None),('random',120,None)]` | 20 + 120 | 4.0x |

In hand, same seeds, same base: **CTRL-K1 11.59** (0 random) and **MIXR-K1 7.17**
(20 random). Together with D60 and D120 that is a four-point dose curve
0 → 20 → 60 → 120, plus the pure-random point.

**Confound, stated:** the dose arms raise the training **volume** and the **random
fraction** together. h209's NIR120 (40 MES/member, 120 traj/iter) showed extra *MES* volume
does nothing (+1.42 vs NIR, inside band), so volume alone is an unlikely driver — but
random-data volume has never been isolated. R-ROI is the volume-matched pure-random point
(20 rollouts, same count as CTRL-K1).

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-DOSE (the user's hypothesis):** monotone improvement — D60 < MIXR-K1 (7.17) − 1.26,
  and D120 < D60 − 1.26.
- **My lean, registered against it: NON-monotone.** I expect the curve to improve from 0
  to 20 and then flatten or reverse, because raising the random fraction pulls the learned
  action mean further toward the ROI centroid, and in the limit the policy *is* the
  centroid. Concretely I expect D60 ≈ MIXR-K1 and D120 ≥ MIXR-K1. **If the dose is
  monotone through D120 my action-mean account is wrong and "more cheap trajectories
  help" is the finding.**
- **P-RROI (the crux):** R-ROI lands between CTRL-K1 (11.59) and the saturation floor
  (43.94), i.e. it does **not** collapse the way h149's uniform version did, because the
  ROI centroid is a moving, belief-tracking point rather than a fixed box centre. Lean:
  ~12–20. **If R-ROI ≈ MIXR-K1 (7.17), the random half's entire contribution is the ROI
  centroid and MIXR reduces to a blend of two known points. If R-ROI ≈ 43.94, the ROI pool
  does not rescue the all-random schema and the mixture itself is doing the work** —
  which retracts the ROI-centroid reading in `why-full-random-failed.md`.
- **Ordering, so it can be wrong:** CTRL-K1 > R-ROI > D120 ≈ D60 ≈ MIXR-K1.

## Diagnostics recorded at readout (not gates)

Per arm, on real post-init queries: **dispersion** ‖per-dim std‖, **distance of the query
cloud's centre from the box centre**, and **mean distance from each query to that
iteration's ROI centroid**. The action-mean account predicts distance-to-ROI-centroid falls
monotonically as the random fraction rises, reaching ~0 for R-ROI. That is a direct test of
the mechanism, independent of the regret.

## What each outcome RETRACTS

- **Monotone dose** → my non-monotone lean, and the action-mean account of the random half.
- **R-ROI collapse to the floor** → the ROI-centroid reading of why-full-random-failed.md;
  the pool is not what distinguishes the two all-random arms.
- **R-ROI ≈ MIXR-K1** → MIXR's MES half is doing nothing, which would make the method
  "train on ROI-random rollouts" and much simpler than anything claimed so far.

## Compute

15 runs, cap 15, one wave, queued behind h212's launch (the launcher waits for slots).
Borehole at MIXR-K1 (20+20) ran ~150 min/seed; D120 is ~4x the rollout cost, so expect
5–7 h/seed. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
