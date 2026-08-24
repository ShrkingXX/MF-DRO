# Protocol — H8: does RTG move the decision WITHIN its realised band?

**Locked before running.** Results commit must be separate.

## Why

Every RTG sweep in this project used multipliers 0.1x-10x (a 100x span). I then
measured that the realised `rtg_target` is **structurally clamped to
[0.500, 1.000]** — a 2x band — by `max(batch_max, 0.5*running_max)` acting on a
batch-max-normalised rtg. So those sweeps were ~50x wider than anything the model
has ever seen, and largely probed out-of-distribution conditioning values.

"The policy does not respond to values it was never trained on" is close to
vacuous. The claim has to be re-tested inside the band.

## Hypothesis

**H8**: within the realised band [0.5, 1.0], varying the RTG target does not
change the proposed candidate.

## Design

Reuse the existing probe harness on a trained model (no retraining, no new BO
run). Two sweeps on the SAME model, state and candidate pool:

- **IN-BAND**: rtg_target in {0.50, 0.60, 0.70, 0.80, 0.90, 1.00} — the actual
  realised support.
- **OOD** (the old design, for direct comparison): base x {0.1, 0.5, 1, 2, 5, 10}.

12 resampled candidate pools each. Record for both: fraction of sweeps where the
argmax moves, distinct argmaxes per sweep, and pairwise score correlation.

## Locked predictions

1. **Primary**: IN-BAND argmax movement is **< 20%** of sweeps — i.e. the
   original conclusion survives once the OOD confound is removed.
2. IN-BAND score-vector correlations are **higher** (more similar) than OOD,
   since the inputs differ less.

Prediction 1 is a prediction that my own earlier finding **holds**. If it fails
— if the argmax moves freely in-band — then the entire "RTG is inert" line was
an artefact of testing OOD values, and several earlier conclusions (H4's
refutation especially) need revisiting.

## What each outcome means

- IN-BAND movement < 20% -> "RTG does not drive decisions" survives, now stated
  correctly as a claim about the realised support. H4/H5 conclusions stand.
- IN-BAND movement high -> the RTG findings were an OOD artefact. H4's AdaLN
  refutation becomes unsafe, and the schema (alpha_rtg, normalisation) becomes
  the prime suspect rather than the network.
- Both near zero AND OOD near zero -> consistent with the network being
  insensitive across the board; the band width is then not the explanation.

## Compute

Single process, thread-capped. 12 active extension workers + 1 = 13 <= 15.
