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

---

## AMENDMENT 1 (2026-08-28) — the confirmatory data does not exist. Filed on
## discovering it, before computing any h120 statistic.

I locked this protocol having verified h84's arm configs, budget, spec, commit
provenance and dirty flags — and did NOT verify that its ROI-OFF arm has five
seeds. It does not. h84 Borehole holds:

  ROI-Q10   seeds 42, 43, 44, 45, 46   (5)
  ROI-FIX2  seeds 42, 43, 44, 45, 46   (5)
  ROI-ANN   seeds 42, 43, 44, 45, 46   (5)
  ROI-OFF   seeds 42, 43               (2)   <-- the control

Both the ckpt and the finals directories agree, so this is a genuine shortfall
in h84 and not a checkpoint artefact. Reported as required by the project's
"report every run, including failures and gate misses" rule: h84's no-ROI
control was never completed past seed 43.

CONSEQUENCE: P1-P4 all require >= 4 of 5 seeds. **They cannot be evaluated. H120
is UNRUNNABLE as locked.** It is not a fail and not a pass; the data is absent.

SURVEY OF ALTERNATIVES (done before choosing): the only experiments pairing a
no-ROI control against an ROI arm within one experiment are h84 (n=2) and h90
(n=5, seeds 47-51). h90 is the set that GENERATED the hypothesis, so it cannot
confirm it. h86 has ROI-Q10 at 42-46 but no control arm. There is therefore NO
independent 5-seed set anywhere in this repository, and no combination of
existing experiments produces one without cross-experiment pairing, which is
banned here.

WHAT I WILL DO:
1. Compute P1-P4 on the two available seeds and report them as **n=2,
   DESCRIPTIVE, NOT CONFIRMATION**. Two seeds cannot support any of the locked
   criteria and no gate verdict is issued. Their only use is to say whether the
   direction is even consistent with h119 before compute is spent.
2. Register the compute: ROI-OFF at seeds 44, 45, 46 under h84's exact config,
   which would complete the control arm and make this protocol evaluable at
   n=5 as locked. NOT launched now: the machine is at 15/15 workers (10 on the
   peer's h113, 5 on h117). Queued for when slots free.
3. The locked predictions P1-P4 stand UNCHANGED and are evaluated only once the
   five seeds exist. The n=2 numbers below do not modify them and must not be
   cited as partial support.

---

## AMENDMENT 2 (2026-08-28) — the provenance condition was wrong. Withdrawn.

Amendment 1 said the three control runs "must run on code that is empty-diff
against h84's af5ec31b1 over src/ dro_runner.py benchmarks.py" and referred to
"my working tree". Both are corrected, on the peer session's challenge:

1. **There is ONE working tree.** Both sessions share this repository;
   `git status` returns the same two modified files for either of us. There is
   no my-tree/your-tree distinction and I should not have written one.
2. **The empty-diff condition is unmeetable and unnecessary.** Meeting it would
   require reverting the h94/h102 patches, which would break the peer's h113
   mid-flight. And the property it was proxying for has already been MEASURED,
   twice, on this very tree:
     - h105: use_roi=FALSE path byte-identical across contributing commits
       (md5 ff70f008c0ac). ROI-OFF *is* the use_roi=False path.
     - h109: use_roi=TRUE path bit-identical to h84's stored traces, 2/2 seeds,
       115 and 103 queries, |dregret| = 0.
   Verified independently here rather than taken on assertion: h109's stored
   `code.dirty` is **True** with `dirty_files` = exactly
   `['M src/model/decisionTransformer.py', ' M src/policy/mf_dro.py']` — the
   same two files modified now. So the PATCHED tree has already reproduced h84's
   own traces bit-for-bit on the harder path.

   Requiring an empty diff on top of two measured reproductions is stricter than
   any other result in this project has been held to.

### The residual gap, stated rather than glossed

h109's `dirty_files` matches the current modification by FILENAME, not by
CONTENT. The h94 patch has been edited since (the `len(self.iteration_log)` fix
after its NameError), so "the same two files were dirty" is not proof that the
same bytes were dirty.

That exact gap is what h117's GATE G0 measures: Ackley_10D MF-DRO seed42 on the
CURRENT tree against h83's stored trace. It is still running (35 min elapsed
against a 32 min median for that arm).

**Dependency stated up front: if G0 fails, it voids these three ROI-OFF runs as
well as h117.** They are launched now anyway because the two prior reproductions
make failure unlikely, the slots are free only while h113 drains, and a failure
costs three runs rather than a wrong conclusion. This is a compute bet, declared
before the fact, not a provenance claim.

Predictions P1-P4 remain UNCHANGED.

### AMENDMENT 2 DEPENDENCY DISCHARGED (2026-08-28)

h117's GATE G0 **PASSED**: 83 queries, 0 differing. The current tree reproduces
a stored MF-DRO trace bit-identically, so the residual "same filenames is not
same bytes" gap flagged above is now closed by measurement on the current
content. The three ROI-OFF runs are not void. The compute bet declared above
was not called.
