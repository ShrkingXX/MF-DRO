# h167c -- the control's constant baseline, which h167b excluded rather than guessed

STATUS: protocol locked, nothing run. TYPE: EXPLORATORY (a direct extension of
h167b, proposed after seeing its result).

## The gap h167b left open

h167b compared each failing teacher's observed `L_loc` against the best constant
predictor for its own action distribution:

  RANDOM        best constant 0.0834   observed L_loc 0.018-0.022   ratio ~4.2
  ORACLE        best constant 0.0533   observed L_loc 0.018-0.022   ratio ~2.7
  DIVERSE-GOOD  best constant 0.0544   observed L_loc 0.018-0.022   ratio ~2.8

The control was **excluded** because its targets are model-dependent and no
analytic constant exists. That was the right call at the time, but the number is
computable empirically: h163 already generates MES teacher actions on real
states, and its best constant follows directly.

## Why it matters

The L_loc puzzle is that the control's loss (0.040) is HIGHER than the failing
arms' (0.018-0.022) even though the control is the one that works. Raw loss is
not comparable across arms with different target distributions -- a spread-out
target set has a higher floor. The **ratio** (best-constant MSE / observed loss)
is comparable: it measures how much structure each network captured beyond
guessing.

## Predictions (no strong prior; both directions are informative)

P1 The control's ratio EXCEEDS the failing arms' 2.7-4.2 -> the working network
   captured more structure, the raw-loss comparison was misleading, and the
   L_loc puzzle dissolves into a units artefact.
P2 The control's ratio is COMPARABLE or LOWER -> the failing networks captured
   as much or more structure than the working one. The puzzle survives in a
   sharper form, and it would mean fitting the teacher well is simply not what
   distinguishes the arms -- consistent with h167's relocation of the failure to
   inference, and further evidence against any training-side account.

## What this can RETRACT

Nothing already claimed -- h167's relocation rests on the failing arms' own
numbers, which stand either way. What it can do is REMOVE the L_loc puzzle from
the open-questions list (under P1) or SHARPEN it (under P2). Recording that
neither outcome rescues a training-side explanation, so this is not a route back
to one.

## Caveat

The MES teacher action distribution is generated on real Borehole states via
h163's code, not read from serialised `actions_x` (never stored). Same
approximation h167b made for ORACLE and DIVERSE-GOOD, applied consistently.

## Compute

1 worker, minutes. 13 running + 1 = 14 <= 15.
