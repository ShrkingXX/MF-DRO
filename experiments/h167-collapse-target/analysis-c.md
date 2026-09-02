# h167c — **P2 fires.** Fitting the teacher well is anti-correlated with working.

EXPLORATORY (extension proposed after seeing h167b). Raw `L_loc` is not
comparable across arms — a more spread-out target set has a higher floor. The
comparable quantity is the **ratio**: how much better the network does than the
best possible constant for its own target distribution.

| arm | best-constant MSE | observed L_loc | **ratio** |
|---|---|---|---|
| **control (works)** | 0.0532 | 0.040 | **1.33** |
| RANDOM-POOL (fails) | 0.0834 | 0.020 | **4.17** |
| ORACLE (fails) | 0.0533 | 0.020 | **2.67** |
| DIVERSE-GOOD (fails) | 0.0544 | 0.020 | **2.72** |

**The working arm fits its teacher WORST.** The control's network beats a
constant by only 1.33× — it explains roughly a quarter of the target variance.
Every failing arm's network beats its constant by 2.7–4.2×.

## What this does to the L_loc puzzle

P1 (the puzzle dissolves into a units artefact) does **not** hold. It survives
and inverts: it was already odd that the failing arms had lower raw loss; on the
comparable measure they are not merely lower but **substantially better fits**.

So the statement is no longer "the DT fits the failing teachers better and still
emits worse points". It is stronger:

> **Fitting the teacher is not what makes the method work. Across four arms,
> fitting it better goes with performing worse.**

That is consistent with everything h167 established — the failure is not
training-side — and it independently reinforces h150's retraction of "the DT
distils the teacher's policy": a network that explains a quarter of its
teacher's action variance is not distilling it.

## The threat to this conclusion, stated plainly

The four baselines are NOT equally reliable, and the asymmetry runs against the
surprising result.

- RANDOM, ORACLE and DIVERSE-GOOD have **state-independent** target
  distributions (uniform draws; interpolation to x*; interpolation to a
  true-objective argmax). Their baselines are exact up to sampling.
- The **control's targets are state-dependent** — MES argmax over the current
  model. Its baseline was generated on **6 states from seeds 42/43**, while its
  observed `L_loc` averages over ~60 real iterations. If the real MES action
  distribution spreads out later in a run, the true baseline is higher and the
  control's ratio is an **underestimate**.

So the one arm carrying the surprise has the weakest baseline. The direction of
the bias is knowable but its size is not, and 1.33 vs 2.67 is not so large a gap
that this can be waved away.

**What would settle it:** serialising `actions_x` per rollout, which this
project has failed to do five times and which would make every one of these
baselines exact rather than reconstructed. Registered as the fix, not attempted
here.

## Standing

Nothing already claimed is retracted — h167's relocation of the failure to
inference rests on the failing arms' own numbers, which are the reliable ones.
This sharpens the L_loc puzzle rather than removing it, and it closes off
training-side explanations further rather than opening one.
