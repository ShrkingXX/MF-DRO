# h162 -- LEARNABILITY: is the teacher's action a function of the observed state?

STATUS: protocol locked, nothing run. TYPE: **EXPLORATORY** (post-hoc, on data
already collected). No new runs; no compute beyond reading JSONs.

## A sharper statement of the open hypothesis

findings.md carries "the working arms query model-selected locations" as a
post-hoc hypothesis. The sharper version, which makes testable predictions:

**The teacher's action must be a function of the state the DT can observe.**

  control MES  action = argmax MES(state's model)          -> a function of the state
  h155 UCB     action = argmax UCB(state's model)           -> a function of the state
  h153 frozen  action = argmax MES(an EARLIER state's model)-> a function of a state
  ORACLE       action = interpolate toward x*               -> x* is NOT in the state
  DIVERSE-GOOD action = interpolate toward argmax of true-f draws -> NOT in the state
  RANDOM-POOL  action = uniform draw                        -> NOT in the state

An action that is not a function of the observable state cannot be fitted from
it, however much data is supplied.

## The apparent contradiction this must resolve

`L_loc` is LOWER for the forced teachers (0.018-0.022) than for the control
(0.040). If their actions were unlearnable, the fit loss should be HIGHER. This
has been recorded as an unexplained puzzle for many ticks.

The learnability framing resolves it: **low loss on clustered targets means the
network is predicting their mean, not learning a mapping.** The oracle paths all
converge on x*, so their action targets have low variance, and a constant
predictor scores well. Low L_loc is then evidence of collapse, not of fit.

## The prediction, testable on serialised data

If the DT is reduced to predicting a near-constant for the unlearnable teachers,
its OWN real queries should be far more CLUSTERED than the control's.

M1 dispersion: mean pairwise distance between post-init real queries
   (domain-normalised). Prediction: control and h153 HIGH; ORACLE, DIVERSE-GOOD,
   RANDOM-POOL LOW.
M2 effective support: fraction of the query set needed to cover it -- measured
   as the mean distance from each query to its nearest other query, normalised
   by M1. Prediction: lower for the collapsed arms.

h154 measured mean STEP SIZE (consecutive distance) and found no such split
(control 0.127, ORACLE 0.146, DIVERSE 0.144, RANDOM 0.090). Step size is not
dispersion: a policy can take large steps inside a small region. M1/M2 measure
the region, which is the quantity the collapse story is about.

## What this can RETRACT

R1 No dispersion split -> the "predicting the mean" resolution of the L_loc
   puzzle is WRONG, and the learnability framing loses its only currently
   testable prediction. The puzzle returns unexplained and findings.md must say
   so rather than carrying the framing.
R2 Dispersion splits as predicted -> the framing survives one test. It is still
   EXPLORATORY and post-hoc; a single consistent correlate is not a mechanism,
   and this will not be reported as establishing one.
R3 Split in the OPPOSITE direction (collapsed arms MORE dispersed) -> actively
   contradicts the framing; the founding diagnosis did report MF-DRO's proposals
   as 3x more dispersed than MF-MES, so this outcome is live and is named.

## Standing caveat

h150 retracted a published finding built on query-level statistics at n=5.
h154's M2 direction was also refuted. This measure is weak evidence about what a
network represents, and no outcome here will be reported as establishing the
mechanism.
