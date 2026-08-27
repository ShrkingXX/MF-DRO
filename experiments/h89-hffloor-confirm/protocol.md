# H89 — clean confirmation of the HF floor at fresh seeds

LOCKED BEFORE ANY RUN. Required by h85 Amendment 2 and by the h87 precedent.

## Why

h85 measured the real-query HF floor (`real_hf_every=4`) on Hartmann at 5/5:

  mean 7.99% -> 6.62%  (paired -1.37 pts)
  sd   5.85  -> 2.00   (across-seed spread cut 66%)

That is provisional, not a finding. h84's Hartmann result also looked good at
4/5 seeds, carried four correct caveats, was announced, and did not survive h87
(2/5 at fresh seeds). The h85 result is in a WEAKER position than h84's was: its
mean improvement rests almost entirely on ONE seed (42: 16.41% -> 6.91%), and on
three of five seeds the floor is slightly worse.

## Design

| | |
|---|---|
| benchmark | Hartmann_6D only |
| arms | MF-DRO + `real_hf_every=4`, and MF-DRO control (no floor) |
| seeds | **52, 53, 54, 55, 56** -- never used in h83/h84/h85/h86/h87 |
| config | `real_hf_every=4` FIXED IN ADVANCE. No other value will be run on these seeds. |
| everything else | identical to h83: M=3, pool 600, no ROI, regression head, budget 200 post-init |

The control must be re-run at the fresh seeds; h83 only covers 42-46. It runs
FIRST (Lesson 21) so the comparator is available before the treatment lands.

## Predictions (pre-registered, in priority order)

- **P1 (PRIMARY — VARIANCE).** The floor reduces across-seed spread of relative
  regret: sd(floor) < sd(control) at seeds 52-56. This is the registered
  mechanism -- a worst-case rescue -- and the effect h85 showed most strongly
  (5.85 -> 2.00). Chosen as PRIMARY over the mean deliberately: h87 demonstrated
  that a mean at n=5 can be carried entirely by one seed, and h85's own mean
  already is.
- **P2 (MEAN).** The paired mean difference is negative. Reported but NOT
  primary, for the reason above.
- **P3 (MECHANISM — the one that can invalidate everything).** The CONTROL at
  fresh seeds exhibits fidelity collapse: consecutive low-fidelity streaks
  longer than 3 on >= 3/5 seeds. h85's controls showed streaks of 131, 73, 75
  and 50. **If collapse does not recur, the floor has nothing to rescue and any
  effect measured here is not the registered mechanism**, regardless of what the
  regret numbers say.
- **P4 (HONESTY BAR).** The improvement is concentrated in seeds whose control
  collapses. Specifically: the floor's per-seed gain correlates with the
  control's longest LF streak. If the floor helps uniformly, including on seeds
  with no collapse, the mechanism story is wrong even if the method works.

No p-values at n=5. The analysis script will be written and committed BEFORE the
treatment arm finishes, will refuse verdicts below 5/5, and will print the
falsifier itself (h87 template).

## Falsifier

If P1 fails, the variance claim is withdrawn from findings.md and any report, as
prominently as it was made. If P3 fails, the mechanism claim is withdrawn even
if P1 and P2 pass -- a working intervention with a wrong explanation is still a
wrong explanation, and this project has already shipped one.

## Attribution

Forcing periodic high-fidelity queries was the human's proposal. I argued
against it from h83's seed table and registered a negative prediction (h85 P6),
which h85 is refuting. That prior is why this confirmation is being run rather
than the result being announced.

## Amendment 1 — add a Borehole REFINE-100 confirmation arm (before any h89 run)

h85's Borehole teacher-refinement arm completed at 5/5 while h89 was still
unstarted: **-5.85 pts, better on 5/5 seeds** (15.82% -> 9.96%). That is the
largest effect any intervention has produced in this project, larger than the
calibrated ROI's -4.22, and it too is PROVISIONAL under Amendment 2.

Rather than register a separate experiment, h89 gains a second, independent
confirmation arm. The two share nothing except the fresh-seed discipline; each
is judged on its own bars.

| | HF floor (original) | Teacher refinement (added) |
|---|---|---|
| benchmark | Hartmann_6D | Borehole_8D |
| treatment | `real_hf_every=4` | `teacher_refine_samples=100` |
| control | MF-DRO, no floor | MF-DRO, no refinement |
| seeds | 52-56 | 52-56 |

Both controls run FIRST (Lesson 21). Total 20 runs.

### Predictions for the refinement arm (pre-registered)

- **P5 (PRIMARY).** The paired difference is negative on >= 4/5 fresh seeds.
  Unlike the floor arm, the MEAN is an acceptable primary here because h85's
  Borehole effect is NOT carried by one seed -- all five improve, by 3.6 to 7.0
  pts. A seed-count bar is therefore the right test and it is set high.
- **P6.** The effect size is at least half of h85's, i.e. paired mean <= -2.9
  pts. Registered because h87 showed a margin can survive in sign and collapse
  in magnitude, and magnitude is what matters for a method claim.
- **P7 (COST).** Wall-clock stays under 2.5x the control. h85 measured
  1.99-2.02x in flight. P4 of h85 set 2x and refinement is at the boundary; 2.5x
  is the outer limit at which this remains a usable default rather than a
  curiosity.
- **P8 (NEGATIVE).** Refinement does NOT close the gap to MF-MES on Borehole
  (6.40%). h85 reached 9.96%, still 3.56 pts behind. Registered negative so that
  a genuine narrowing is visible as a surprise rather than assumed.

### Falsifier

If P5 fails, the refinement result is withdrawn as prominently as it was
recorded. If P5 passes but P6 fails, it is reported as a real but much smaller
effect than h85 suggested, and h85's figure is corrected.

## Amendment 2 — run MF-MES at seeds 52-56 so P8 is evaluable (before the treatment arm finishes)

P8 was registered as "refinement does NOT close the gap to MF-MES on Borehole
(6.40%)". That 6.40% is measured at seeds 42-46. h87 showed fresh seeds can be
3.27 pts harder for MF-MES, so comparing a seeds-52-56 result against it would
be a cross-seed-set comparison of exactly the kind that produced the
final_regret-vs-grid-at-200 error earlier today. The analysis script already
refuses that comparison and prints CANNOT EVALUATE.

The fix is to run the comparator at the matched seeds. MF-MES on Borehole costs
~5 min/run against MF-DRO's ~100, and 10 worker slots are free while the
REFINE-100 arm finishes.

THIS DOES NOT CHANGE P8. The bar is unchanged; this makes it evaluable as
written. No treatment run is added, no configuration is altered, and the
Borehole REFINE-100 arm is untouched and still unfinished, so its verdict
remains written-blind.
