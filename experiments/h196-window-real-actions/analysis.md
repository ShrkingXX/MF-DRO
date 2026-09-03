# h196 — **P1. The audit's defect was real and worth 2.61 rel% points.**

**CONFIRMATORY**, 5/5 runs, readout committed before any finished (with a completeness
guard added after the first version was seen reporting numbers from 22%-complete runs).

## Gate: does feeding the REAL past actions matter?

| | frozen rel% |
|---|---|
| h194 WINDOW — **zeroed** action tokens | **16.58** |
| **h196 WINDOW — real actions fed** | **13.96** |

Paired **−2.61** (se 0.93), better on **5/5 seeds**, threshold 1.26 → **P1: the fix
matters.**

Per-seed: −1.61, −3.35, −1.91, −0.37, −5.83.

## It recovers half the deficit, and the window still hurts

| | deficit vs CTRL-K1 (11.59) |
|---|---|
| zeroed actions (h194) | **+4.99** |
| real actions (h196) | **+2.37** |

**52% of the window's deficit was an implementation defect.** The remaining +2.37 is
still P3 on h196's own comparison against CTRL-K1 — worse on 4/5 seeds.

## What this settles

**h194's Stage 1a conclusion was, as flagged, about a defective implementation.** The
caveat attached to it ("P3 means the window *as implemented* hurts, not that windows
hurt") was load-bearing and correct: half the effect was the defect.

**But the window is still harmful when fed correctly.** So the corrected reading is
narrower, not reversed: with real past actions, MSE loss, and target-valued history
RTGs, an 8-step window costs **+2.37 rel% points** against no window.

## What remains untested, and why h197 exists

h196 fixes only h195's defect #1. It does **not** implement the human's full
specification — its loss is MSE, and its historical RTGs are still the *target*, not
information gain computed from the real GP. h197 adds both. So h196 bounds the
action-feeding effect alone (−2.61) and leaves the rest to h197.

## Credit where due

The defect was found only because the human challenged my claim that the sliding-window
question was "already tested — h27", and asked for the implementation to be audited
against the DT paper. Neither the identity gate nor code review would have caught it; it
took reading Algorithm 1 against our inference path.
