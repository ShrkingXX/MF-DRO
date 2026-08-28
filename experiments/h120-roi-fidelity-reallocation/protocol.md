# H120 — Does the ROI act on the FIDELITY MIX rather than on spatial search?

STATUS: LOCKED before any h84 statistic was computed.
TYPE: CONFIRMATORY. Tests the hypothesis h119's screen generated, on data that
      played no part in generating it.
COMPUTE: zero new runs.
DATA: h84-roi-strategy ONLY — Borehole_8D, arms ROI-OFF and ROI-Q10, seeds
      42-46. Disjoint from h119's h90 seeds 47-51. No cross-experiment pairing.

## Provenance checks done BEFORE locking (all pass)

- Arm configs are byte-identical to h90's:
  ROI-OFF  = `dict(use_roi=False)`                                  [= h90 NO-ROI]
  ROI-Q10  = `dict(use_roi=True, roi_beta_mode='quantile', roi_target_accept=0.10)`
- BUDGET=200.0 and Borehole SPEC n_hf=10, n_lf=20 in both experiments.
- Every h84 Borehole run post-dates `950fdd6` ("ROI candidate pool lost
  resolution -- every ROI A/B was confounded"), so these arms are NOT affected
  by that confound. Verified per-run with `git merge-base --is-ancestor`.
- h84's arms ran on DIFFERENT commits (ROI-OFF on 2c1b1fe/be7109f, ROI-Q10 on
  af5ec31). `git diff` between all three over `src/ dro_runner.py benchmarks.py`
  is EMPTY: they differ only in docs and result files. Run-relevant code is
  identical across the arms.
- All runs record `dirty=False`.

## Hypothesis (from h119's screen, stated before computing on h84)

The ROI does not change WHERE MF-DRO searches. It changes WHAT IT BUYS: fewer
high-fidelity queries, more low-fidelity ones, earlier convergence, and better
individual HF queries.

## Predictions (locked)

P1 (PRIMARY). Fidelity reallocation. ROI-Q10 makes FEWER non-init HF queries
   and MORE non-init LF queries than ROI-OFF, in >= 4/5 seeds, with paired
   |mean|/sd >= 1.0 on the HF count.

P2. Earlier convergence. Cost consumed before the final best HF y is first
   reached, as a fraction of budget, is LOWER for ROI-Q10 in >= 4/5 seeds.

P3 (COUNT-MATCHED, the sharp one). Mean y over the FIRST K non-init HF queries,
   where K = min(count_ROI-OFF, count_ROI-Q10) within each seed, is HIGHER for
   ROI-Q10 in >= 4/5 seeds with |mean|/sd >= 1.0.
   Count-matching is required because h119's C5 averaged over 9% fewer queries
   in a run that also converged earlier, and a more converged run has a better
   average by construction. P3 removes that confound. The uncorrected version
   (h119's C5) is also reported, labelled as the confounded one.

P4 (NEGATIVE CONTROL, pre-registered as such). C6, the founding diagnosis's own
   statistic — fraction of HF queries worse than the best initial-design HF
   point — did NOT separate in the screen (0.62). It is predicted NOT to
   separate here either. If it DOES separate, the screen's account is wrong or
   incomplete and that must be reported as prominently as a pass.

## Falsification

If P1 fails, the fidelity-reallocation account is refuted and the h119 screen
produced a seed artefact. If P1 passes but P3 fails, the ROI reallocates budget
without improving per-query quality, which is a materially different and weaker
claim than the screen suggested — and it would have to be reported as such
rather than as partial support.

## Limitations

- n=5 seeds, one benchmark, no p-values.
- Confirms on Borehole only, where the ROI effect is known to live. Says nothing
  about generality; h111 already showed the regret effect fails on Hartmann and
  Ackley at two tightness settings.
- Two arms of a four-arm experiment are used. ROI-ANN and ROI-FIX2 exist in h84
  and are deliberately NOT examined here: they were not part of the hypothesis,
  and screening them would reintroduce exactly the multiplicity this protocol
  exists to escape.
- This tests the CHANNEL, not whether the channel explains the regret benefit.
  Even a full pass leaves "does reallocating fidelity cause the 3.5-4.2 pt gain"
  untested, because no arm here manipulates the fidelity mix directly.
