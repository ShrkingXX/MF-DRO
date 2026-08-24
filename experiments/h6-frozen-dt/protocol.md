# Protocol — H6: is the DT contributing anything after training?

**Locked before running.** Results commit must be separate.

## Motivation

H4 and H5 both refuted, and together they license a strong claim:

- swapping `h` for a different state's hidden vector: argmax unchanged 12/12
- with **no GP features at all** (coords only), so that any non-arbitrary
  ranking MUST flow through `h`: argmax *still* unchanged 12/12
- batch-mean / shuffled `h`: ~8% argmax change
- state perturbation at 1x batch std: argmax unchanged, score corr 0.9997
- RTG sweep 0.1x-10x: argmax essentially pinned in every configuration

Within a single trained model, the proposal is very nearly independent of the
conditioning. Yet `x_t_trace` std is 0.166-0.213, so queries do move across a
run. The reconciliation: the DT is **retrained every BO iteration**, so it
changes because its *weights* are re-fit, not because it conditions.

If that is right, MF-DRO is functioning as a per-iteration acquisition function
parameterised by a transformer — not as a return-conditioned policy.

## Hypothesis

**H6**: the DT's *learned conditioning* contributes ~nothing to run behaviour;
the per-iteration re-fitting does the work.

## Design

Hartmann 6D, seeds 42-51, matched cost 200, identical init — the frozen
evaluation, unchanged. One variable:

- **Arm LIVE** (control): current behaviour — retrain the DT every iteration.
  This arm already exists: it is `h1-leak-fix-validation`'s MF-DRO arm. Reuse
  it; do not re-run.
- **Arm FROZEN**: train normally through iteration `k=5`, then **stop training
  the DT entirely** for the rest of the run (weights frozen; GP ensemble still
  refits, rollouts still generated for the RTG/BTG targets). Everything else
  identical.

`k=5` is fixed in advance, not tuned.

## Locked predictions (confirmatory)

1. **Primary**: FROZEN mean final simple regret is within **1 SE** of LIVE's.
   (i.e. freezing the policy after 5 iterations costs approximately nothing)
2. FROZEN's incumbent-improvement count is within 1 of LIVE's.

## What each outcome means

- Predictions hold -> **H6 supported**: continued DT training contributes
  ~nothing measurable. The "learned return-conditioned policy" framing is not
  doing work in this method, and the honest write-up is a negative result about
  the DRO frame on this problem. PROTOCOL.md explicitly permits this.
- FROZEN is clearly *worse* -> H6 refuted: retraining does matter, so the DT is
  learning something run-relevant even though single-model probes cannot detect
  it through `h`. That would mean my probes measure the wrong thing, and the
  next step is to find what re-fitting changes that conditioning does not.
- FROZEN is clearly *better* -> continued training is actively harmful
  (overfitting to fresh rollouts), which is itself a concrete, actionable
  finding about `num_epochs` / retraining cadence.

## Why this is worth the compute

Every outcome is informative, and it is the first experiment that tests the
*premise of the method* rather than one of its components. It also directly
addresses PROTOCOL.md's actual research question — whether a fix exists **within
the DRO frame** — because if the DT contributes nothing, the frame itself is
what is in question.

## Compute

10 runs, `num_workers=10 x threads_per_worker=1 = 10 <= 15`. Reuses the LIVE arm
from h1, so only the FROZEN arm is new. Must not launch while the h1 grid still
holds the pool.
