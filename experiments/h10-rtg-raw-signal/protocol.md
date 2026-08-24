# Protocol — H10: is RTG inert, or starved? (the decisive test)

**Locked before running.** Results commit must be separate.

## The question, and why it is the last one standing

Every RTG result is ambiguous between:

1. **the network ignores RTG** (architecture/training failure), and
2. **RTG carries too little variation to be usable** (schema failure).

A write-up claiming (1) — which is what "MF-DRO re-fits rather than conditions"
amounts to — would overstate the evidence while (2) is live.

## Why the obvious fix failed, and what the data says the real lever is

H9 tried widening `alpha_rtg` (0.5 -> 0.1). **Void**: the floor binds 46.3% of
iterations at 0.5 and **0.0%** at 0.1, so the knob simply disappears and both
arms gave the identical band [0.5672, 1.0000].

Crucially, H9 also shows that with the floor inactive the target *is* `batch_max`
— and `batch_max`'s own range is still only [0.5672, 1.0]. So the narrowness is
**not** the floor. It is the normalisation:

    rtg <- rtg / running_max_ever      (my own RTG-cap fix)
    batch_max = (this batch's best) / (best ever seen)

The best of ~200 fresh rollouts is nearly always close to the best ever, so this
ratio is pinned near 1 by construction. **The signal is compressed by the very
fix that removed the earlier "2 distinct values across 200 trajectories"
pathology.**

## Hypothesis

**H10**: RTG-insensitivity belongs to the network, not the signal. Conditioning
on a genuinely varying target will still not move the decision.

## Design — one variable

New flag `rtg_target_mode`:

- **`normalized`** (control, current default): target = batch-max-normalised, band ~[0.57, 1.0]
- **`raw`**: skip the batch-level normalisation entirely; the target is the
  **raw** improvement of the batch's best rollout. Early in a run improvements
  are large, late they approach zero, so this varies over orders of magnitude
  *by construction*.

Both arms **train from scratch with their own target**, because the network can
only learn to use a signal it was trained on — the H4 lesson (AdaLN-Zero starts
at identity and needs training to depart from it).

Each arm is then swept **on its own realised band** (6 points, 12 pools), the
in-distribution discipline H8 established.

## Locked predictions

1. **Manipulation check**: `raw`'s realised band is >= 5x wider (in max/min
   ratio) than `normalized`'s ~1.76x. If it is not, the experiment is VOID and
   prediction 2 must not be interpreted — same guard that correctly killed H9.
2. **Primary**: `raw` argmax movement stays **< 20%** of sweeps.

Prediction 2 again bets on my prior conclusion holding.

## What each outcome means

- Manipulation passes, movement < 20% -> **H10 supported**: the insensitivity is
  the network's. The confound closes and "MF-DRO re-fits rather than conditions"
  becomes defensible as written. This is the result that unblocks a write-up.
- Manipulation passes, movement high -> **H10 refuted, and far more interesting**:
  RTG was *starved by my own fix*, not ignored. Every conditioning-side
  refutation (H4, H5, H8) must be re-read as "tested under a compressed signal",
  and the remedy is a schema change, not an architecture change.
- Manipulation fails -> VOID; report and do not interpret.

## Compute

2 arms x 1 training run + probes, single process each. Machine idle;
2 workers x 1 thread = 2 <= 15.
