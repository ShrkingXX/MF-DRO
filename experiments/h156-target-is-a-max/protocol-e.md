# h156e -- is the misfit my harness's MISSING ENSEMBLE VARIANCE?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## The pattern

Every condition is UNDER-predicted, never over:

  C1 control        −7.2, −1.5, −11.1, −7.9 %
  C3 random         −7.6, −13.5, −4.6, −1.1 %
  C4 oracle         −2.9, −23.6, −31.2, −29.8 %
  C5 diverse-good   −19.3, −24.7 %

A one-sided error across all five conditions and both benchmarks is a
systematic bias in the instrument, not five separate facts about the arms.

## Two candidates, one already eliminated

RULED OUT -- the running-max FLOOR. `target = max(batch_max, 0.5*running_max)`,
so a binding floor would pin the target to a constant once running_max settles.
ORACLE seed42's series has **125 distinct values in 136 iterations** (min
0.2434, max 0.4917), so batch_max is the binding term almost always. The floor
is not the explanation.

UNDER TEST -- the ENSEMBLE. The real batch is
`for ko in self.ko_ensemble: for _ in range(rollouts_per_model)` with
gp_num_models=10, rollouts_per_iter=200. Those 10 KO-GPs are fit on IDENTICAL
data (mf_dro.py:_update_ko_ensemble) and differ only through fitting
randomness -- different hyperparameters, therefore different posteriors and
different b. **My harness uses ONE GP and N=100.** It is missing an entire
variance component that the real batch's MAX feeds on, which is exactly a
one-sided under-prediction.

It should hurt C4/C5 MOST in relative terms: their within-condition spread is
smallest (sd 0.10-0.11 against C1's 0.22), so the missing between-model spread
is a larger share of their total.

## The change

Build a 10-member ensemble per state (10 KennedyOHaganGP fits on the same data,
different torch seeds, matching gp_num_models=10), draw 20 trajectories per
member per condition = 200 per batch (matching rollouts_per_iter=200), and take
the MAX over the full 200. Conditions themselves are unchanged.

## REPLICATE, per the guard adopted this session

h156's precision claim collapsed because I never measured the harness's own
reproducibility. This protocol therefore runs **two independent replicates at
different seeds**, and no agreement figure will be quoted from a single run.
Prior noise floor to beat: 6.1% mean, 10.9% worst.

## GATE (pre-stated)

PASS  the systematic under-prediction largely closes: mean |error| across the
      four observable conditions falls below ~10% AND no condition exceeds ~15%,
      in BOTH replicates.
FAIL  C4/C5 stay 20%+ under. The misfit is then not the ensemble either, and
      the caveat in findings.md hardens into a permanent limitation.
PARTIAL  errors shrink materially but C4/C5 remain outside 15%. Reported as
      partial; the ensemble would be A cause, not THE cause.

Named now: an improvement that only moves numbers inside the 6% noise floor is
NOT evidence of anything and will be reported as no effect.

## What this can RETRACT

- PASS retracts h156d's conclusion that the interpolating-condition misfit is a
  fact about the arms; it becomes a fixed bias in my instrument, and h156c/d's
  analyses need correcting rather than appending.
- FAIL/PARTIAL retracts nothing already claimed. The scale separation (3-4x,
  far outside noise) is untouched either way, as is the h153 forecast, which
  depends on the C2/C1 RATIO -- a within-run quantity where a shared missing
  variance component largely cancels.

## Compute

2 workers (two replicates, Borehole). h153 (5) + h155 (5) + 2 = 12 <= 15.
