# H46 — h44's responsiveness is mostly architecture, not learning

**Status: GENERIC (locked verdict 1).** Confirmatory.

## Result

Hartmann 6D, seed 42, 20 iterations, regression head. Arm U is the trained DT's
**own initial weights**, deep-copied out of the policy before training, so the
two arms differ by training and nothing else. Same recorded states, same run
spread (0.244892), same sweeps. The trained arm reproduced h44 to the digit.

| channel | arm | max ‖Δx‖ | mean ‖Δx‖ | max/spread | mean/spread |
|---|---|---|---|---|---|
| state | T trained | 0.122096 | 0.083635 | 49.9% | 34.2% |
| state | **U untrained** | 0.088937 | 0.047837 | **36.3%** | 19.5% |
| RTG | T trained | 0.092748 | 0.060920 | 37.9% | 24.9% |
| RTG | **U untrained** | 0.037524 | 0.016719 | **15.3%** | 6.8% |
| BTG | T trained | 0.002042 | 0.001145 | 0.8% | 0.5% |
| BTG | U untrained | 0.000875 | 0.000482 | 0.4% | 0.2% |

`R_U / R_T` on the state channel = **0.728**, which clears the locked GENERIC
threshold of 0.7.

## Verdict

**h44's 49.9% is withdrawn as evidence of learned conditioning.** An untrained
network of the same architecture, never having seen a gradient, is 73% as
responsive to the state as the trained one. The number measures the sensitivity
of an MLP to its inputs, which is a property every non-degenerate MLP has.

This is the same error class as the h5 AUDIT's h-swap that compared a state
against itself: a quantity that reads as evidence and is not.

## Two things not to over-read

1. **0.728 clears 0.7 by 0.028, on one seed.** The verdict is the one that was
   locked, and it stands, but it is a boundary call and a second seed could move
   it into AMBIGUOUS. It should not be quoted as a decisive margin.

2. **The RTG channel does not behave like the state channel.** `R_U/R_T` there
   is 0.404 — inside the AMBIGUOUS band — and training raised RTG
   responsiveness 2.5× (15.3% -> 37.9%). The protocol named the state channel as
   the deciding one, so this does not change the verdict, and it is reported as
   a secondary observation on n = 1. It is the only place in this project where
   training has been seen to *increase* an input channel's influence, and it is
   worth one confirmatory run before anything is built on it.

## Consequence

The scope qualifier added to `FOR_ADVISOR.md` after h44 asserted that "the state
reaches the proposal" under the regression head. That is now unsupported and has
been rewritten. The correct statement is that the regression head's output moves
with its inputs about as much before training as after, so h44 distinguishes the
two heads' *architectures*, not their conditioning.

Whether the regression head is any *better* remains open and is h45's and h47's
question, neither of which depends on h44.
