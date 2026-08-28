# H90 — clean confirmation of the Borehole ROI gain at fresh seeds

LOCKED BEFORE ANY RUN. Written while compute is at 15/15 with h85; not launched.

## Why

After h87 withdrew the Hartmann flip, **Borehole is the only positive regret
result the ROI has left**, and it has never been tested at seeds it was not
developed on. h87's lesson was not "Hartmann was unlucky" -- it was that a clean
re-test at unused seeds is the only thing that settles a claim.

Per h85's Amendment 2, the Borehole gain is currently PENDING CONFIRMATION, not
a finding. This experiment is what converts it into one, or withdraws it.

## The claim under test

  Borehole_8D, ROI-Q10 vs no ROI, seeds 42-46, paired:
    -3.78, -2.49, -1.56, -5.71, -7.57   mean -4.22, sd 2.43, 5/5 wins
    drop-one-seed: mean stays -3.39 to -4.89, always 4/4

Note this is a claim about the ROI IMPROVING MF-DRO, not about beating a
baseline. MF-MES is at 6.40 on Borehole against MF-DRO's 11.59; the ROI closes
27% of that gap and does not come close to eliminating it.

## Design (the h87 template)

| | |
|---|---|
| benchmark | Borehole_8D only |
| arms | MF-DRO + ROI-Q10, and MF-DRO with `use_roi=False` |
| seeds | **47, 48, 49, 50, 51** -- never used for Borehole |
| config | q=0.10 FIXED IN ADVANCE. No other ROI setting will be run on these seeds. |
| everything else | identical to h83/h84: M=3, pool 600, refinement off, budget 200 post-init, regression head |
| runs | 10 |

Both arms are re-run at the fresh seeds. The no-ROI arm is NOT reused from
anywhere -- there are no h83 Borehole runs at seeds 47-51, and reusing a
different seed set is what the whole experiment exists to avoid.

ONE ROI ARM ONLY. If q=0.10 underperforms, that is the result.

## Metric (frozen)

h83's metric via h83's own sr_curve/grid: SR grid-interpolated at exactly cost
200, as relative regret. The analysis script will be committed BEFORE the
treatment arm finishes, as h87's was.

## Predictions (pre-registered)

**P1 (PRIMARY). The paired difference (ROI minus no-ROI) is negative on >= 4/5
fresh seeds AND the paired mean is negative.**

I expect this to HOLD, and I am on record getting exactly this call wrong once.
For h87 I registered the same expectation for Hartmann and it failed 2/5. What
is different, stated so it can be judged rather than trusted:

  - Hartmann's gain came from ONE setting of three; the other two were neutral
    or harmful. Borehole's gain appears at ALL THREE settings tested (-4.81
    fixed beta, -4.22 q=0.10, -1.31 q~0.49), so it is a property of applying an
    ROI rather than of the setting selected.
  - Hartmann's 4/5 rested on a paired sd of 0.45 that turned out to be
    seed-set-specific (7.45 at fresh seeds). Borehole's sd is 2.43 with every
    seed improving and a drop-one-seed range of -3.39 to -4.89 -- the
    robustness does not depend on the spread being small.
  - Hartmann's margin was 0.68 pts against a comparator. Borehole's is 4.22 pts
    against itself.

**P2. The margin shrinks relative to -4.22 pts.** Even without setting-selection,
seeds 42-46 are where the configuration was developed.

**P3 (NEGATIVE). MF-DRO + ROI-Q10 still does NOT beat MF-MES on Borehole.**
Registered as negative: the gap is 11.59 vs 6.40 and a 4-point gain does not
close it. If this is refuted, something is wrong with the comparison.

## Falsifier

If P1 fails, the Borehole gain is WITHDRAWN as prominently as it was reported,
and the ROI has **no surviving regret result on any benchmark**. What would
remain is the controllability argument alone -- that fixed beta cannot set ROI
tightness (12.6%-100% across benchmarks, 250x within a run, 6.9x across seeds)
-- which is a claim about the method's parameterisation, not its performance.

That outcome must be stated that plainly if it occurs.

## Gate

Launch when compute frees. Requires 10 slots. A concurrent session currently
owns the launcher and is running h85; whichever session launches this must check
`pgrep -f 'code/worker.py' | wc -l` first and stay within 15 total.

## Renumbered from H88 (concurrent-session collision)

This experiment was written as H88. A concurrently running session independently
created `experiments/h88-surrogate-vs-data` under the same number and committed
results to it. Two distinct hypotheses cannot share an ID in a project whose
history is its pre-registration record, so this one -- which had no results yet
-- was renumbered to H89. The other retains H88.

Recorded rather than silently renamed: anyone reading the git log will see two
commits titled "H88" for different experiments, and this note explains why.

## Amendment 2 — renumbered again (H89 -> H90) and a third arm added

RENUMBER. A concurrent session created `experiments/h89-hffloor-confirm` an hour
after this session published a claims block reserving `h89-borehole-confirm`.
Reserving SLUGS does not prevent NUMBER collisions. This experiment moves to H90
(it has no results; renumbering is free) and the claims block now reserves
number RANGES instead: the other session holds h85-h89, this session holds h90+.
Second collision of the day; the first was H88.

THIRD ARM, added BEFORE launch. h85 finished while this was queued and produced
the largest intervention effect measured anywhere in this investigation:
**teacher refinement on Borehole, -5.85 pts, better on 5/5 seeds**, with its
mechanism bar P2 also met (near-bound fraction 8.93% -> 16.32%). Neither
confirmation experiment written by either session covered it -- this one tested
the ROI, the other tests the HF floor. Confirming the second-best result while
leaving the best one unconfirmed would be indefensible.

  arms: NO-ROI (control), ROI-Q10, REFINE-100
  seeds: 47-51, never used for Borehole
  15 runs, exactly the worker cap

Both treatment configurations are FIXED IN ADVANCE and are the ones h85/h84 ran.
No variant of either will be run on these seeds.

### Additional prediction for the new arm

**P4. REFINE-100's paired difference against NO-ROI is negative on >= 4/5 fresh
seeds.** h85 measured -5.85 on 5/5 at seeds 42-46. I expect this to hold, and
note that I made the same call for the Hartmann ROI flip in h87 and was wrong.
What differs: refinement's effect is 1.4x the ROI's, its mechanism bar was
independently met, and its per-seed effects (-6.12, -6.97, -5.79, -4.75, -5.63)
sit in a 2.2-point band with no seed carrying it -- where the Hartmann flip's
0.36-point band turned out to be seed-set-specific.

## Amendment 3 — overlap with the concurrent session's H89 is COMPLEMENTARY. Do not trim.

Both sessions independently added a Borehole teacher-refinement confirmation arm
within minutes of each other, having each concluded it was the highest-value
unconfirmed result. That convergence is a signal the priority was right, but it
looks like duplication and someone may be tempted to cut one. They should not.

  H89 (other session), seeds 52-56:
    Hartmann CONTROL, Hartmann HF-FLOOR, Borehole CONTROL, Borehole REFINE-100
  H90 (this session), seeds 47-51:
    Borehole NO-ROI, Borehole ROI-Q10, Borehole REFINE-100

UNIQUE TO H90: **Borehole ROI-Q10.** H89 does not test the ROI at all. The ROI is
the intervention this entire investigation was commissioned to evaluate, and its
Borehole gain (-4.22 pts, 5/5, robust to dropping any seed) is the only ROI
result still standing after the Hartmann withdrawal. Without H90 it never gets a
clean test.

THE OVERLAP IS AT DIFFERENT SEEDS, and that is worth having rather than
tolerating. h87's central lesson was that a paired difference measured on one
seed set does not transfer: Hartmann's paired sd was 0.45 on seeds 42-46 and
7.45 on 47-51, because the two methods failed on different instances. Refinement
is now the largest claimed effect in the project (-5.85 pts, 5/5). Testing it on
TWO independent fresh seed sets (10 seeds total) is a direct response to that
lesson, not redundancy.

If compute is scarce, the arm to cut is H90's REFINE-100 -- not its ROI-Q10, and
not H89's anything.

---

## ADDENDUM, registered with ZERO h90 results on disk

The analysis script read only ROI-Q10 vs NO-ROI and silently ignored the
REFINE-100 arm -- 5 of the 15 runs, and the third seed set for the only
intervention still standing. Registering its bar now, while no h90 result exists,
rather than choosing one after seeing the numbers.

### Teacher refinement's record, and the decay in it

  seeds 42-46   mean -5.85   5/5
  seeds 52-56   mean -2.11   4/5     36% of the original effect
  seeds 47-51   this run

The effect did not merely survive its first fresh-seed test, it SHRANK by 64%.
Two points do not establish a trend, but they do define two readings that this
third seed set can distinguish:

  STABLE   the fresh-seed value is the true one -> expect approx -2.1, 4/5
  DECAYING each new seed set costs the claim -> expect approx -1.0, 3/5 or worse

**P4.** REFINE-100 beats NO-ROI on >=3/5 with a negative paired mean.
Registered POSITIVE, but deliberately at a LOWER bar than its 5/5 and 4/5 record
would justify, because the decay above is the more honest prior. If the mean
lands weaker than -1.0 or wins <=2/5, the STABLE reading is dead and refinement
is a seed-set-dependent effect like the three already withdrawn.

**P5.** REFINE-100 does not make MF-DRO competitive: its mean stays above the
strongest Borehole baseline. Registered POSITIVE. This has now held at two seed
sets and I expect it to hold again; it is registered because it is cheap and
because a refutation would be the most important result of the run.

### Falsifier

If P4 fails, teacher refinement joins the ROI flip and the HF floor as withdrawn,
and NO intervention tried this session survives fresh seeds. The session's answer
to its primary question would then be uniformly negative, and the only durable
results would be the controllability argument and the seed-dependence finding.

## Amendment 4 — the two arms are made independent (registered with 0/5 REFINE results)

The analysis script exited as soon as the ROI arm had a missing pair, which
meant the REFINE-100 bars -- P4 and P5, registered in the addendum as separate
claims -- could be skipped entirely by an unrelated failure in the ROI arm. One
crashed ROI run would have silently cost the refinement arm its verdict, and the
skip printed nothing to say so.

Fixed: each arm now evaluates independently, and an incomplete arm prints
"CANNOT EVALUATE" instead of falling through to a verdict or to the falsifier.
This mirrors the P8 lookup-path defect in h89, where a verdict script printed
CANNOT EVALUATE because it read the wrong directory -- the same class of error,
caught here before it could fire.

Registered with **zero REFINE-100 results on disk** (all 5 still running) and 1
of 5 ROI pairs complete. No bar, threshold, or seed set is changed -- this is a
control-flow fix to the evaluator, not a change to what is being tested.
