# h163 -- the INVERSION: does a more dispersed teacher produce a more collapsed student?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY (the prediction is
non-obvious and is stated before the teacher-side numbers are computed).

## The prediction

h162 measured the STUDENT side: the DT's own real-query dispersion splits
working arms (0.246-0.289) from failing ones (0.112-0.189), with RANDOM-POOL
lowest at **0.1115**.

The learnability framing says the DT falls back to predicting the MEAN of an
unlearnable target. That makes a sharp and counter-intuitive prediction about
the TEACHER side, which has never been measured:

**RANDOM-POOL's teacher actions are uniform draws over the candidate pool -- the
MOST dispersed teacher of all six. And it produces the LEAST dispersed student.**

If the framing is right, teacher dispersion and student dispersion should be
roughly INVERSELY ordered: the more spread out and less state-predictable the
target, the harder the network falls back on a constant.

If instead the DT simply imitates its teacher's spread, they should be
POSITIVELY ordered. That is the natural null and it is what "the model
inherits its teacher's behaviour" would predict.

## The measurement

Generate teacher actions for each of the six arms' rules on real Borehole states
(offline harness, same states as h156/h158), and measure mean pairwise distance
in normalised domain -- the same M1 as h162, applied to the TEACHER rather than
the student. Then rank-correlate teacher M1 against h162's student M1.

Teacher rules, taken from the arms' own generators:
  MES argmax (control), UCB beta=2 (h155), UCB beta=0 (h159),
  ORACLE interpolation to x*, DIVERSE-GOOD interpolation to a POOL=256 argmax,
  RANDOM uniform draws.
h153 (frozen MES) shares the control's rule and is not separately generated.

## What this can RETRACT

R1 POSITIVE correlation -> the DT largely inherits its teacher's spread, the
   collapse story is wrong, and h162's dispersion split needs a different
   explanation. The learnability framing loses the prediction that currently
   distinguishes it from "the model copies its teacher".
R2 NEGATIVE / inverted -> the framing survives a second and much less obvious
   test. Still EXPLORATORY in aggregate; it remains a correlate, not a
   demonstrated cause.
R3 No clear ordering -> uninformative; reported as such rather than as weak
   support.

## Caveat, unchanged

Query-level statistics at n=5 are the evidence class that produced h150
(retracted) and h154's refuted M2 direction. The INVERSION is worth testing
because it is non-obvious and the natural null predicts the opposite sign, not
because this class of evidence has become stronger.

## Compute

1 worker, offline, minutes. h161 (5) + 1 = 6 <= 15.
