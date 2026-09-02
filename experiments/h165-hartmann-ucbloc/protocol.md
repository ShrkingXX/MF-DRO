# h165 -- does "a different learnable teacher works" generalise to Hartmann?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Why

The strongest positive result of the whole front is h155: a UCB teacher matches
the control (15.13 vs 15.82, both 5/5), which retracted "the MES rule
specifically". **It is Borehole-only.** So is the entire 2x2, h158's dose, and
h162/h163. Hartmann currently has only FAILING non-control arms (ORACLE,
RANDOM-POOL) plus the control. There is no Hartmann evidence that any
substitute teacher can work.

## The arm

h155's worker, unchanged, on Hartmann_6D seeds 42-46. `rollout_policy="ucb_loc"`,
ucb_loc_beta=2.0 (default), fidelity channel held fixed by construction.
mf_dro.py is not modified; h155's bit-identity gate stands.

## Prediction

Near the Hartmann control, improving on a comparable share of seeds, and clearly
better than the Hartmann failing arms.

Hartmann references (frozen metric, rel% @cost 200): control and RANDOM-POOL to
be read from the pre-committed readout at analysis time, not quoted here from
memory.

## Named confound, checked before the number is read

Realised HF fraction against the Hartmann control's. Hartmann's control makes
far fewer HF queries than Borehole's (fewer than 12 on three of five seeds, per
h164), so a fidelity shift is more likely to matter here. A collapse voids the
arm regardless of regret.

## What this can RETRACT

R1 UCB-LOC fails on Hartmann -> h155's retraction of "the MES rule specifically"
   is Borehole-specific, and findings.md must scope it to one benchmark.
R2 UCB-LOC works -> the strongest positive result generalises.
R3 Intermediate -> inconclusive at n=5, reported as such.

## Compute

5 workers alongside h161's 5 = 10 <= 15.
