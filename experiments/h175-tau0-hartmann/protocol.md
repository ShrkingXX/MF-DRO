# h175 -- is the tau=0 mechanism WEAKER on Hartmann?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.
Zero new runs: reconstructs the tau=0 teacher action offline, as h170 did.

## The question the h173 correction raises

h173 at n=5 showed the STRONG form ("only the first step matters") holds on
Borehole and fails on Hartmann: HEAD matched the control there (16.96 vs 15.82)
but is worse on 5 of 5 Hartmann seeds (25.16 vs 7.99). So on Hartmann the later
steps do real work.

The natural explanation is that the tau=0 mechanism is itself weaker there -- the
DT's query is less determined by its teacher's first-step mean, leaving room for
the later steps to matter. **h170 tested that only on Borehole**, where the query
sat 3.3x closer to the tau=0 mean than to the box centre, with a ~5 SE residual.

## The measurement

h170's harness, unchanged, on Hartmann arms: for each of ~40 states, reconstruct
the tau=0 teacher action distribution (120 draws), take its mean, and compare the
DT's actual query against it, the box centre, and a random pool point.

Arms: Hartmann control and h165 UCB-LOC (working, where the two targets differ --
the discriminating case), plus RANDOM-POOL and ORACLE (failing, calibration).

## Predictions

P1 The Hartmann working arms' d(q, tau=0 mean) / d(q, centre) RATIO is
   materially LARGER than Borehole's 0.306 and 0.309 (i.e. the query is
   relatively less determined by the tau=0 mean), consistent with the later steps
   mattering there.
P2 The failing arms still calibrate (their reconstructed tau=0 mean lands on the
   box centre, within the estimator floor).

## What this can RETRACT

R1 P1 FAILS -- the ratio is the same on both benchmarks -> the tau=0 mechanism is
   equally strong on Hartmann, and the strong form's failure there needs a
   DIFFERENT explanation. That would be the seventh account to be proposed and
   the mechanism's scope claim would be left unexplained rather than merely
   incomplete.
R2 P1 holds -> the two benchmarks differ in how tightly tau=0 determines the
   query, which is a quantitative account of why the strong form is
   benchmark-specific.
R3 P2 fails -> the reconstruction is miscalibrated on Hartmann and nothing can be
   read either way. Checked FIRST, before P1 is looked at, because h170's initial
   run failed exactly this way (12 draws put the estimator floor above the
   threshold).

## Estimator floor, computed in advance

Hartmann is 6D. The mean of 120 uniform draws sits an expected
sqrt(6)/(sqrt(12)*sqrt(120)) ~= 0.065 from the centre. P2's threshold is set at
0.15, comfortably above that -- the mistake h170 made was setting a threshold
BELOW the floor, and it is not repeated here.
