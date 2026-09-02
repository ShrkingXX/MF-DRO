# h166 -- complete the 2x2 on HARTMANN

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## The gap

The 2x2 is the front's central structural result -- three of four cells work,
and only the doubly-changed cell fails -- and it is **Borehole-only**. Hartmann
currently has:

  MES / closed-loop      control                     present
  non-MES / closed-loop  h165 UCB-LOC                running
  non-MES / open-loop    RANDOM-POOL                 present (ORACLE confounded)
  **MES / open-loop      MES-FROZEN                  MISSING**

h153 (the Borehole MES/open cell) is the arm that refuted the target-collapse
account, by carrying the failing arms' conditioning target with the control's
performance. That refutation rests on one benchmark.

## The arm

h153's worker, unchanged, on Hartmann_6D seeds 42-46. Two passes: an ordinary
MES rollout supplies the path, then the same rollout replays it frozen. The
NORMALISED -> RAW conversion, the fidelity rule and SC1-SC3 instrumentation are
all h153's; mf_dro.py is not modified.

Hartmann is ~4x cheaper per seed than Borehole (h165 seed 43 finished in 19
minutes), so a two-pass arm here costs roughly what a one-pass Borehole arm does.

## Predictions

Like Borehole's h153: **works** (near the Hartmann control, far from
RANDOM-POOL's failure) while its **rtg_target collapses** into the failing band.

The split matters more than either half: it is the split -- good performance
with a collapsed target -- that refuted the target-collapse account.

## Sanity checks, read before the number

SC1 pass-2 path reproduces pass-1 exactly (Borehole gave 0.0 on every seed)
SC2 open-loop penalty > 0 (Borehole: 0.30-0.40, vs 0.16 measured offline)
SC3 fidelity flip fraction between passes (Borehole: 0.15-0.18)
SC4 realised HF fraction against the Hartmann control's -- Hartmann's control
    makes far fewer HF queries than Borehole's, so a shift matters more here

## What this can RETRACT

R1 Hartmann MES-FROZEN FAILS (~RANDOM-POOL's level) -> the 2x2's key asymmetry
   is Borehole-specific, and h153's refutation of the target-collapse account
   must be scoped to one benchmark. findings.md and the published report both
   state that refutation without a benchmark qualifier.
R2 It works AND its target collapses -> the refutation generalises and the 2x2
   becomes a two-benchmark result.
R3 It works but its target does NOT collapse -> the split does not reproduce,
   so Hartmann cannot corroborate the refutation either way. Named because it is
   the outcome that looks like success but carries no evidence.

## Compute

5 workers alongside h161 (5) and h165 (4 remaining) = 14 <= 15.
