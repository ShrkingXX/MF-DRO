# h221 — Make the RTG label encode the first action: CRN + advantage baseline

Registered BEFORE any h221 run. CONFIRMATORY. Two new knobs (`fantasy_crn`,
`rtg_advantage`), both default off.

## The defect being repaired

Measured (`rtg-carries-no-action-information.md`, same directory): the adjusted R² of
`rtg[0]` on the **first action** `(x_0, ell_0)`, within an ensemble member, n=100 rollouts:

| rollouts | adj R² | τ=0 action spread |
|---|---|---|
| MES | **+0.007** | 0.35 |
| random | **−0.045** | 0.70 |

Both zero. The action varies (20 distinct of 20) and the return varies (sd 0.10–0.16); they
do not vary together. `rtg[0]` is dominated by the other seven steps and by
fantasy-sampling luck, so the label says *"this rollout got lucky"*, not *"this action was
better"*.

This is upstream of the head (h220), the target schema (h212) and RNG parity (h213):
none can extract selection from a label that does not encode it. Combined with h208's
measurement that positions 1–7 are inert as supervision (L8 ≈ L8-TRUNC, 10 seeds), it is
why rollout steps τ>0 cannot reach the real query **at all**.

## The two interventions

**CRN** (`fantasy_crn`): one seed per (member, BO iteration), shared by every rollout of
that member, applied as `manual_seed(crn_seed + 1000003·τ)` around the fantasy draw with
the RNG state saved and restored. Two rollouts of a member then see the **same** noise at
the same step index, so outcome differences are attributable to action differences rather
than to luck. The same device `regret_lookahead_teacher` already uses.

**Advantage baseline** (`rtg_advantage`): subtract the member's mean `rtg[0]` from each of
its rollouts' whole RTG vector. Removes the component of the return that is shared across
rollouts starting from the same posterior, leaving the part that differs between them.
Applied per member, before the batch-level normalisation (a single scale, so it cannot
reorder).

Neither needs extra rollouts. They compose.

## Arms — 15 runs, Borehole 42–46, CTRL-K1 base (K=1, `mes_entropy`, ROI-Q10)

| arm | `fantasy_crn` | `rtg_advantage` |
|---|---|---|
| **CRN** | on | off |
| **ADV** | off | on |
| **BOTH** | on | on |

Reference at the same seeds: **CTRL-K1 11.59**. Also in hand: MIXR-K1 7.17.

## Stage 0 — the gate is the R² itself

The arms are pointless if the interventions do not repair the measured defect, so Stage 0
**re-measures adjusted R² of `rtg[0]` on `(x_0, ell_0)`** under each setting, n=100
rollouts/member, exactly as the defect was measured.

- **SC1 (GATE):** CRN raises adjusted R² above **0.10** on ≥2 of 3 members (from +0.007).
- **SC2:** the advantage baseline leaves the *within-member ordering* of `rtg[0]`
  unchanged (it is a constant shift) — a check that it is doing what it claims and not
  destroying signal.
- **SC3:** identity gate, 122.29066752728207, with both flags off.

**If SC1 fails, the arms do not launch** and the finding is that CRN does not recover the
action signal — which would point at the credit horizon (fix 2, `rollout_length=1`) rather
than at the noise.

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-CRN:** CRN − CTRL-K1 < −1.26 on ≥4/5. Making the label encode the action is what the
  whole RTG design needs, so it should show up in regret.
- **P-ADV:** |ADV − CTRL-K1| ≤ 1.26. A constant shift per member cannot change what the
  label *ranks*; it only recentres it, so on its own I expect little. It is included
  because it is free and because it interacts with the target schema (recentred RTG makes
  `max(batch_max, alpha·running_max)` behave differently).
- **P-BOTH:** BOTH ≤ CRN. Lean: the two compose, with CRN doing the work.
- **Ordering, so it can be wrong:** BOTH ≈ CRN < ADV ≈ CTRL-K1.

**My record on this thread:** three of my last four registered leans were wrong (label
inert — h213; non-monotone dose — h214; harm on the held-out benchmarks — h217). Weight
the argument, not the lean.

## What each outcome RETRACTS

- **P-CRN fails while SC1 passes** → the label now encodes the action and the DT *still*
  does not use it. That moves the blame decisively back to the head/conditioning (h220's
  territory) and would be the cleanest possible evidence for it.
- **SC1 fails** → fantasy noise is not what decouples the label from the action; the credit
  horizon is, and `rollout_length=1` becomes the arm.
- **P-CRN succeeds** → the first repair of the RTG channel in this project, and the
  multi-teacher / low-quality-contrast lines become worth revisiting on top of it.

## Compute

15 runs, cap 15, queued behind the running fleet. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
