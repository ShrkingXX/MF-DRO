# h182 — the failure is COLLAPSE TO THE BOX CENTRE

**EXPLORATORY.** No new runs; re-analysis of saved traces. Not pre-registered.
Statistic: over each run's **last 20 HF queries**, mean distance (unit box) to the
box centre. Frozen rel% throughout. Arms keyed by tag **and** experiment (never
merged across experiments — the same tag in two experiments is two configurations).

## CORRECTION to this file's first version

The first version reported Borehole as **bimodal** — working arms at 0.803–0.864,
failing arms at 0.070–0.107, "a gap 0.70 wide, 10/10 arms fit." That was measured
on **10 hand-picked teacher arms**. Re-run over **every** Borehole arm with ≥4
matched seeds, the gap fills in: MF-DRO [h75] at 0.752, POOL1000 [h78] at 0.778,
STALE-PATH at 0.798 land inside it, with intermediate regrets (24.25, 22.28,
19.53). **The bimodality was an artifact of which arms I sampled.** The real
structure is a continuum, and the corrected claim below is the stronger one.

## Borehole — monotone across 28 MF-DRO arms

| population | n | Spearman ρ(distance from centre, rel%) |
|---|---|---|
| **MF-DRO arms only** | **28** | **−0.967** |
| GP baselines | 4 | −0.800 |
| all arms pooled | 32 | −0.918 |

Across 28 MF-DRO arms — spanning teachers, ROI variants, rollout lengths, loss
functions and conditioning changes — the further the policy's queries sit from
the box centre, the better it does, near-monotonically. Largest departures from
the trend are small (+4.8, +3.8, −2.8 rel% pts).

*(ρ is a descriptive rank statistic over arms, each itself a 5-seed mean. No
p-value is attached; the n=5 rule concerns seeds within an arm.)*

Endpoints: RANDOM-POOL 0.070 → 43.94; MF-MES 1.093 → 6.40.

### The one counterexample is out of population

**MF-MI-Greedy** escapes the centre (0.904) and still performs poorly (28.44) —
the only arm badly off-trend. It is a **GP baseline with no Decision Transformer
at all**, so it cannot bear on a claim about what a DT emits. Naming it rather
than dropping it: escaping the centre is clearly not sufficient *in general*, only
within the DT-based arms this analysis is about.

## Hartmann — the same ordering, and distance-to-x* runs BACKWARDS

| arm | rel% | to x* | **to CENTRE** | dispersion |
|---|---|---|---|---|
| control | 7.99 | 0.814 | **0.610** | 0.343 |
| UCB-LOC | 10.58 | 0.852 | **0.597** | 0.350 |
| HEAD-MES | 25.16 | 0.619 | **0.546** | 0.289 |
| TAIL-MES | 46.45 | 0.600 | **0.149** | 0.216 |
| ORACLE | 52.23 | 0.572 | **0.142** | 0.195 |
| RANDOM | 65.14 | 0.582 | **0.086** | 0.120 |

Distance to the centre is perfectly rank-monotone with performance. **Distance to
x\* is anti-monotone** — the worst arms look "closest" to the optimum, because
`||centre − x*|| = 0.5681` and an arm collapsed *at* the centre scores ≈0.57 while
converging nowhere. RANDOM scores 0.582.

> **Diagnostic caution for the codebase.** `query_dist_to_xstar_per_iter` is
> anti-monotone with performance on these arms and must not be read alone: it
> cannot separate "converged near the optimum" from "collapsed at the centre".
> This does **not** overturn the earlier second-basin observation, which was made
> on different arms whose queries sat 0.97–1.10 from x* — not centre-collapse.

## What this does NOT explain: the benchmark asymmetry

I initially wrote that escape from the centre is "necessary and sufficient on
Borehole, necessary but not sufficient on Hartmann", and that this explained why
the front's answer has a strong form only on Borehole. **The numbers do not
support that.**

| | HEAD's escape ÷ control's | HEAD's regret ÷ control's |
|---|---|---|
| Borehole | 0.812 / 0.852 = **0.95** | 16.96 / 15.82 = **1.07×** |
| Hartmann | 0.546 / 0.610 = **0.90** | 25.16 / 7.99 = **3.15×** |

HEAD retains a **similar fraction** of the control's escape on both benchmarks —
95% and 90% — yet costs 1.07× on one and 3.15× on the other. Escape distance
therefore orders arms *within* a benchmark extremely well and does **not** by
itself explain the difference *between* them. **The asymmetry remains open**, as
it has been since h175.

## What could RETRACT this

- An MF-DRO arm far from the centre that fails, or one at the centre that
  succeeds. Currently 28 arms, ρ = −0.967, no serious outlier.
- The direction of causation is untested. Centre-collapse and poor regret could
  both follow from a third cause; nothing here intervenes on distance directly.
- Last-20-HF-queries is one window, chosen once and not tuned.

---

## The signature is DYNAMIC: failing arms contract, working arms expand

Splitting each run's HF queries into six equal windows and taking the ratio of the
last window's mean centre-distance to the first's:

| arm | rel% | w1 | w6 | **w6/w1** |
|---|---|---|---|---|
| TAIL-MES | 43.94 | 0.290 | 0.089 | **0.308** |
| RANDOM-POOL | 43.94 | 0.195 | 0.064 | **0.328** |
| ORACLE-EXPERT | 43.94 | 0.280 | 0.100 | **0.357** |
| DIVERSE-GOOD | 43.94 | 0.242 | 0.103 | **0.424** |
| MES-FROZEN | 19.36 | 0.666 | 0.799 | **1.200** |
| HEAD-MES | 16.96 | 0.645 | 0.819 | **1.270** |
| control | 15.82 | 0.672 | 0.852 | **1.268** |
| ROLLOUT1 | 13.69 | 0.709 | 0.864 | **1.218** |
| ROI-L1 | 9.81 | 0.782 | 0.978 | **1.250** |

**Run over all 28 MF-DRO Borehole arms** — not the hand-picked subset that misled
the first pass — the separation is complete:

| group | n | w6/w1 range |
|---|---|---|
| failing (rel% > 30) | 4 | **0.308 – 0.424** |
| working (rel% ≤ 30) | 24 | **1.200 – 1.390** |

**Gap 0.777 wide, no overlap, 28/28 arms.** Every arm that works moves *away* from
the centre over its run; every arm that fails moves *into* it. These are opposite
directions, not degrees of one thing.

### Why this matters for the front's answer

The first emitted query predicts final regret only moderately (ρ = −0.707), while
the last-20 centre-distance predicts it strongly (ρ = −0.967), and the two
correlate at only +0.676. So **the first query is not destiny.** The τ=0 mechanism
explains what the DT emits at a *given* training state; across a run the DT is
retrained repeatedly, and the arms diverge — one group compounding its way out of
the centre, the other compounding its way in.

### Honest limits

- **Only 4 failing arms, and all four sit at the identical 43.94 saturation
  floor.** The separation is 24-vs-4 with a degenerate failing group; a fifth
  failing arm at a distinct regret would test it far better than a sixth working one.
- **Nothing lies between 0.424 and 1.200**, so the location of the boundary — and
  whether 1.0 is special — is untested.
- **Correlational.** No arm here intervenes on centre-distance directly, so
  "collapse causes failure" and "both follow from a third cause" are not separated.

---

## The geometric explanation of the asymmetry is REFUSED (measured, not argued)

Last section proposed that Borehole's optimum being at the boundary makes
"distance from the centre" a proxy for quality there, so one good first step
suffices — while Hartmann's multimodality breaks that. **Tested directly by
sampling the objectives** (20 000 uniform draws each, HF objective):

| benchmark | ρ(distance from centre, objective value) | \|ρ\| | top-1% mean dist | all-sample mean |
|---|---|---|---|---|
| **Borehole_8D** | **−0.027** | **0.027** | 0.948 | 0.804 |
| **Hartmann_6D** | **−0.374** | **0.374** | 0.667 | 0.695 |
| Currin_2D | −0.179 | 0.179 | 0.530 | 0.382 |

**The prediction required Borehole's \|ρ\| to be the large one. It is the small
one — 14× smaller than Hartmann's.** The geometric explanation is refused. This
is the **second** explanation of the benchmark asymmetry to be refused by its own
numbers (the escape-fraction account fell last tick), and the asymmetry stays open.

### But the refutation exposes something better: a dissociation

On Borehole, distance from the centre predicts **arm performance** at ρ = −0.967
while predicting **objective value** at ρ = −0.027. Those cannot both be about the
same thing.

So **centre-distance is not a proxy for "somewhere good"** — random far-from-centre
points on Borehole are no better than central ones. It is a **marker of policy
behaviour**: a policy that has stopped discriminating emits an average, and an
average of a diffuse distribution lands at the centre. The centre is where a
collapsed policy *ends up*, not a place that is bad to be.

This bears directly on the causal question h182 left open. It makes
"collapse causes failure" the *less* likely reading: the centre is not a bad
region, so being there cannot itself be what costs regret. The better-supported
reading is that both the collapse and the poor regret are downstream of the same
thing — the policy no longer selecting.

*(One caveat against over-reading ρ = −0.027: the relationship is non-monotone
rather than absent. Borehole's top 1% of samples do sit further out than average,
0.948 vs 0.804. Rank correlation near zero means distance alone does not order
value — not that the two are unrelated.)*

---

## CORRECTION — "the centre is not a bad place" was WRONG

The section above concluded, from ρ(distance, objective value) = −0.027 on
Borehole, that the centre is not a bad region and that collapse is therefore a
*marker* rather than a cause. **That inference was wrong**, and the caveat I
attached to it — "near-zero rank correlation means distance does not *order*
value, not that they are unrelated" — was the reason it was wrong. I flagged it
and then drew the conclusion anyway.

A rank correlation over the whole box says nothing about the value *achievable in
a small ball at the centre*. Sampled directly:

| benchmark | best f in the whole box | best f within 0.10 of the centre |
|---|---|---|
| Borehole_8D | 273.00 | **85.76** |
| Hartmann_6D | 3.107 | **0.948** |

The neighbourhood the failing arms occupy is severely value-limited on both
benchmarks. **The centre is a bad place to be.**

### The consequence, measured scale-free

Comparing each run's best HF value against its **own initial design's** best:

| arm | seeds improving on their own initial design | init best | final best |
|---|---|---|---|
| RANDOM-POOL | **0/5** | 173.5441 | **173.5441** |
| TAIL-MES | **0/5** | 173.5441 | **173.5441** |
| ORACLE-EXPERT | **0/5** | 173.5441 | **173.5441** |
| DIVERSE-GOOD | **0/5** | 173.5441 | **173.5441** |
| MF-DRO (control) | 5/5 | 173.5441 | 260.6162 |
| HEAD-MES | 5/5 | 173.5441 | 257.0583 |
| ROI-L1 | 5/5 | 173.5441 | 279.1945 |

**Across 20 runs (4 failing arms × 5 seeds) the policy never once improved on its
own initial design.** Final best equals initial best exactly, every seed. This
also explains the long-noted "saturation floor" at 43.94 rel%: that number *is*
the initial design, and the arms coincide there because none of them contributes
anything at all.

### Corrected causal reading

Collapse to the centre is **directly harmful**, not merely diagnostic: it confines
every query to a region that cannot beat the initial design, and the 0/5 result
shows exactly that outcome. The earlier "marker, not a cause" framing is retracted.

What remains untested is the *direction* of the collapse itself — nothing here
intervenes to stop a policy collapsing and observe the result. "Collapsing costs
regret" is now well-supported; "what makes a policy collapse" is not settled by
this.

---

## HARTMANN SWEEP — the relationship replicates, and it is DT-SPECIFIC

h182's headline (ρ = −0.967 over 28 Borehole arms) had only been checked on **6**
hand-picked Hartmann arms. Run over **every** Hartmann arm with ≥4 matched seeds:

| population | n | Spearman ρ(distance from centre, rel%) |
|---|---|---|
| **MF-DRO arms** | **17** | **−0.841** |
| non-DT arms (GP baselines + teacher-only) | 5 | **+0.600** |
| pooled | 22 | −0.373 |

**The relationship replicates for MF-DRO arms** — −0.841 against Borehole's −0.967,
weaker but the same direction and strong over 17 arms.

### The non-DT arms do not collapse at all, and that is the informative part

| arm | centre dist | rel% |
|---|---|---|
| SF-DRO | 0.565 | 10.33 |
| teacher-only (NODT) | 0.675 | 15.58 |
| MF-MES | 0.710 | **6.69** |
| MF-GP-UCB | 0.813 | 56.67 |
| MF-MI-Greedy | 0.893 | 41.34 |

They all sit **far** from the centre (0.565–0.893 — no collapse anywhere), and their
ordering does not follow distance: the **best** of them (MF-MES, 6.69) sits at 0.710
while the **worst** (MF-GP-UCB, 56.67) sits further out at 0.813.

**Centre-collapse is a property of the DT-based policy, not of the optimisation
problem.** That is what the mechanism predicts: the DT emits a *constant* (its teacher's
τ=0 mean), and a bad constant is the box centre. A GP method argmaxes an acquisition
function afresh every step and has no constant to collapse onto. The pooled ρ (−0.373)
is meaningless precisely because it mixes two populations with different behaviour.

### Two honest caveats

- **SF-DRO's classification is arguable.** It is the single-fidelity DRO variant and is
  DT-based, so grouping it with the GP baselines is questionable. It sits at 0.565 /
  10.33, which fits the *MF-DRO* pattern (far out, performs well), so moving it would
  strengthen rather than weaken the split — it is left in the conservative group.
- **An apparent gap on Hartmann should not be trusted yet.** The 17 MF-DRO arms split
  into three collapsed (0.086–0.149, rel% 46–76) and fourteen not (0.523–0.698, rel%
  5.9–13.2), with nothing between. **Borehole showed exactly such a gap on 10 arms and
  it dissolved when all 28 were included.** This one is recorded as an observation, not
  a claim.
