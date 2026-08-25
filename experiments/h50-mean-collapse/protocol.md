# H50 — does the regression head's output sit BETWEEN the teacher's modes?

**CONFIRMATORY. Protocol committed before any run.**

## The claim being tested

h45 (10/10 seeds, regression head) found a failure mode that is NOT the
incumbent freeze: seeds 49 (regret 1.3640) and 50 (1.1818) had **zero
improvements** while proposing a **distinct point every iteration**
(distinct/iters = 144/144 and 74/74). Fresh proposals forever, none ever
beating the incumbent.

The proposed mechanism (Change 1a's argument, restated by the peer session):
MSE regression onto a **multimodal** teacher argmax distribution minimises to
the conditional **mean**, which lands *between* the modes — a point no mode
endorses, hence distinct-but-useless queries.

This is currently a hypothesis. The only data touching it is h45's `x_spread`
column, where the 2 failing seeds explore **0.72x** as widely as the 8
succeeding ones (0.1385 vs 0.1912) — directionally consistent with collapsing
onto a region, but n=2, and `x_spread` is computed over only the first 40
queries. H50 measures the thing itself.

## Design

Four seeds, chosen as the two extremes of h45's bimodal outcome:

| group | seeds | h45 regret | improvements |
|---|---|---|---|
| FAIL | 49, 50 | 1.3640, 1.1818 | 0, 0 |
| PASS | 42, 44 | 0.2051, 0.1157 | 2, 9 |

Configuration is **h45's verbatim** (regression head,
`use_candidate_scoring=False`, `rollout_reward="mes_entropy"`,
`cost_budget=200` post-init, `initial_hf=36 / initial_lf=60`,
`dkl_threshold=9999`, `num_epochs=10`, `rollout_length=8`, Hartmann 6D).
Seeds are the same, so each run reproduces its h45 counterpart.

**No `mf_dro.py` edit.** Instrumentation wraps `mf.dt.propose_mf` from the
worker, the same pattern h31/h47 used. The wrapper calls the original, then
measures the teacher at that exact GP state. It cannot change the trajectory:
it returns the original's output unmodified.

## What is measured, per real BO iteration

**Teacher argmax distribution.** On the SAME 200-point uniform pool the real
proposal uses, call `compute_joint_mf_mes` once per (ensemble member m, draw r)
for m in 0..9, r in 0..4 -> **50 argmax samples**. Each call redraws its own
Thompson y*, so the 50 samples span both ensemble disagreement and y*
uncertainty — the two sources of spread the DT's training data actually has.
Pool size 200 matches the teacher the DT distilled, so modes are genuine pool
points rather than an artefact of a denser grid.

**Mode structure.** Cluster the 50 samples in normalised [0,1]^6 with average-
linkage hierarchical clustering at distance 0.15. Record `n_modes`, mode
weights, centroids, and `mode_sep` (weighted mean pairwise centroid distance).

**Where the DT sits.**
- `d_nearest_mode` = min_i ||x_dt - c_i||
- `d_teacher_mean` = ||x_dt - sum_i w_i c_i||   (the conditional mean)
- `between_ratio` = `d_nearest_mode` / `mode_sep`

**Also recorded per iteration** (already produced by `run()`, no new code):
inference regret (`inference_regret_curve`), simple regret
(`hf_regret_curve`), the real queried location (`x_t_trace`), its value
(`y_t_trace`), and the fidelity (`fidelity_trace`). `use_gp_refinement=False`,
so `x_t_trace` IS the DT's raw proposal — no refinement stands between them.

## Prediction, and what would falsify it

**Mean-collapse is SUPPORTED only if all three hold, FAIL vs PASS:**
1. `n_modes` is higher on FAIL (the teacher is genuinely more multimodal), AND
2. `d_teacher_mean` is **lower** on FAIL (the DT sits at the conditional mean), AND
3. `between_ratio` is **higher** on FAIL (and > ~0.5, i.e. the DT is far from
   every individual mode relative to how far apart the modes are).

**Falsified if** the DT sits *on* a mode in both groups (`between_ratio` small
everywhere), or if FAIL and PASS are indistinguishable on all three, or if
`n_modes` ~ 1 throughout — in which case there are no modes to fall between and
the mechanism cannot be operating, whatever else is wrong.

A distinct possibility worth naming now: the DT may sit near the teacher mean in
**both** groups, with the groups differing only in how multimodal the teacher
is. That would make mean-collapse a permanent property of the regression head
rather than the cause of these two failures, and would predict the failure is
about the *landscape*, not the head.

## Statistics — stated in advance

**n = 2 vs 2 seeds. No significance test will be run or reported.** Iterations
within a seed are not independent, so per-iteration counts must not be treated
as sample size. I will report per-seed medians and the full per-iteration
distributions, and describe the comparison as descriptive. Any claim beyond
"these 4 runs look like X" would need more seeds.

---
## PRE-RESULT ADDENDUM (written with 0 result files on disk, 4 workers ~3 min in)

Prompted by push-back from the peer session. Recorded before any data exists,
which is the only point at which it is legitimate.

### What I was about to do wrong

Having realised criterion #2 (`d_teacher_mean` lower on FAIL) is close to
tautological — MSE onto repeated tau=0 states fits the conditional mean **by
construction**, so the DT should sit near the teacher mean in BOTH groups — I
said that if #1 and #3 separated while #2 did not, I would "report that as the
mode-geometry result rather than as a failed criterion."

That is criterion drift. Stating it in advance makes it honest, not correct:
the registered test would still have been quietly replaced by the one the data
supported. **This project has already retracted a headline for exactly this
shape** — p=0.0371 reported as significant, which did not survive Bonferroni
over the metrics actually examined. The error there was not fabrication; it was
letting the reported criterion drift toward what the data would bear. Credit to
the peer session for catching the same move here.

### The registered test is UNCHANGED: 3 of 3

1. `n_modes` higher on FAIL, **and**
2. `d_teacher_mean` **lower** on FAIL, **and**
3. `between_ratio` higher on FAIL (and > ~0.5)

**If #2 does not discriminate, the pre-registered test FAILED.** It will be
reported in those words. Any mode-geometry finding will be reported
**separately and labelled EXPLORATORY**.

### I considered demoting #2 to a diagnostic, and decided NOT to

The peer suggested that a criterion which cannot fail informatively is not
doing pre-registration work, and could legitimately be demoted now. That is a
fair argument and demoting it before results would be defensible. I am
declining it for a specific reason:

**#2 failing is itself a real finding.** Mean-collapse as originally posed is
the claim that the DT sits at the conditional mean *on the failing seeds* —
i.e. that this is what distinguishes them. If `d_teacher_mean` is equally low
on PASS, that claim is **refuted as an explanation of h45's bimodality**, even
though the collapse itself is confirmed. Demoting #2 to a diagnostic would
erase that negative result and leave only the flattering half. So #2 stays a
criterion precisely because I expect it to fail.

### Timestamped prediction, before data

I predict **#2 will not discriminate** — `d_teacher_mean` will be comparably
low in both groups — because the loss makes it so regardless of seed. I expect
the groups to separate, if at all, on **mode geometry** (`n_modes`, `mode_sep`,
mode weights, and the stability of mode locations across iterations). Recording
this now so that if it holds, the exploratory finding carries a real prior
rather than a retrofitted one — and so that if it fails, that is visible too.

### Consequence for the write-up

The headline cannot be "mean-collapse confirmed via mode geometry". If it comes
out as predicted, the accurate sentence is: **"the pre-registered mean-collapse
test failed; what separates the groups is mode geometry, measured
exploratorily."** The reframing that the interesting question is not *whether*
the DT collapses to the mean but *whether the teacher's mode structure makes
collapsing fatal here and benign there* is the peer session's, and is credited.

n = 2 vs 2 seeds throughout. No significance test, confirmatory or exploratory.

### Resolution of the #2 question (still 0 results on disk)

The peer session withdrew the demotion argument after checking it. Agreed
resolution, no disagreement outstanding:

- **#2 stays a criterion.** It tests the *discriminative* claim — that collapse
  is what separates failing seeds from healthy ones — not the tautological one
  that MSE fits the conditional mean.
- **Both sessions predict, in advance, that #2 will fail.**
- That failure is a **reportable negative result about the hypothesis as posed**,
  not a criterion that underperformed. It refutes mean-collapse as an
  *explanation* of h45's bimodality while confirming the collapse itself.
- General form both sessions now accept: **a criterion you expect to FAIL does
  more pre-registration work than one you expect to pass.**

**For the analysis write-up:** the prediction that #2 will fail and that
separation, if any, comes from mode geometry is dated at commit `58081bb`,
which precedes every result file in this directory. If mode geometry does
separate, the finding stays labelled EXPLORATORY, but the prior is genuine and
independently checkable — a reviewer can verify the commit predates the data.
State this in the analysis with the hash; prose alone does not carry it,
because the commit is the evidence.

---
## VALIDITY GATE — H50 must reproduce h45 (registered with 0 results on disk)

Prompted by the peer session announcing an `mf_dro.py` edit. Checking their
claim surfaced a risk to H50 that neither of us had flagged.

**H50 and h45 ran different code.** h45's workers imported `mf_dro.py` at
**01:43**; H50's imported at **04:27**, after four intervening commits:

| commit | time | why it should be inert for H50 |
|---|---|---|
| 6c7989b | 03:14 | `use_candidate_scoring` default flip — H50 sets it explicitly |
| 3231fba | 03:21 | `natural_decision_lengthscale`, defaults False (verified: flag off reproduces member0=0.1839) |
| 7c72939 | 03:54 | `use_roi=False` default; branch does `torch.rand(_N_POOL=200,…)`, RNG-identical to the old literal |
| f123335 | 04:01 | all changes inside `if roi_mode == 'mes'`, unreachable when `use_roi=False` |

H50's whole design rests on each run **reproducing its h45 counterpart** — that
is what licenses calling 49/50 FAIL and 42/44 PASS. Seeds and config are
identical, so if the executed paths are genuinely identical the trajectories
must match **exactly**.

### The gate

For each seed, compare H50's `sr_curve` against h45's `hf_regret_curve`
element-wise, and H50's `n_iters` / `n_improvements` against h45's.

- **Match** -> code equivalence is demonstrated empirically, not argued from
  reading diffs. The FAIL/PASS grouping stands and the analysis proceeds.
- **Diverge** -> the grouping is VOID. Seeds 49/50 cannot be called the failing
  ones under code they were not measured under. In that case H50 is reported as
  an uninterpretable run, the divergence is characterised, and the experiment is
  re-run with h45's code pinned — not salvaged by re-labelling whichever seeds
  happen to fail this time.

This gate is checked and reported **before** any mode-geometry number, and its
outcome is stated in the analysis regardless of which way it goes. Registered
now, with 0 result files on disk, precisely so it cannot become optional if the
grouping turns out inconvenient.

Note the running processes are immune to further edits — Python loaded the
module at 04:27 — so the peer's pending fixes cannot affect H50 mid-flight.
The risk is only to a restart, and H50 will not be restarted without redoing
this check.

---
## HALTED at user instruction — 0/4 results. Status of each registered item.

Stopped ~45 min in, before any seed completed. Cores handed to h57. The states
below are distinguished deliberately: "registered but not executed" is not
"failed", and neither is "found unnecessary".

### The pre-registered 3-of-3 test: NOT RUN
Criteria #1/#2/#3 require the trained DT's proposal across a trajectory. No
trajectory completed. **No confirmatory claim of any kind may be drawn from
this directory.**

### The validity gate (6f5ece2): REGISTERED, NEVER EXECUTED
It was to compare H50's `sr_curve` elementwise against h45's for a shared seed,
empirically testing whether the four `mf_dro.py` commits between 01:43 and
04:27 (6c7989b, 3231fba, 7c72939, f123335) are inert on the
regression-head + `mes_entropy` + `use_roi=False` path.

**It did not fail, and it was not found unnecessary. It never ran.** Nobody may
cite this directory as evidence that H50 reproduced h45, or that those commits
are behaviour-preserving. Reading the diffs suggests they are — explicit
`use_candidate_scoring`, a flag defaulting False, an RNG-identical
`torch.rand(200,…)`, and code unreachable outside `roi_mode=='mes'` — but that
is argument, not measurement, which is exactly why the gate existed.

**When it becomes worth running again:** if anyone relates h57's MF-DRO arm
back to h45/h17/h31 numbers. h57 is self-contained — all 36 jobs at one pinned
commit with the hash recorded per result — so its internal comparison does not
need this. Cross-experiment comparison does. Cost: one seed, ~40 min. The gate
stays registered rather than deleted so it can be picked up unchanged.

### EXPLORATORY OBSERVATION (a smoke test, not a result)

Teacher argmax structure at **iteration 0 only**, all four seeds, from a
standalone probe (`/tmp/iter0_modes.py`) that never touched the runs. 50
argmaxes = 10 members x 5 y* draws on the same 200-point pool:

| seed | group | unique | modes | top-2 weights | mode_sep | spread |
|---|---|---|---|---|---|---|
| 49 | FAIL | 3 | 3 | [0.50, 0.46] | 1.1578 | 0.6342 |
| 50 | FAIL | 2 | 2 | [0.90, 0.10] | 0.4722 | 0.0867 |
| 42 | PASS | 4 | 4 | [0.36, 0.32] | 0.8170 | 0.5869 |
| 44 | PASS | 2 | 2 | [0.58, 0.42] | 0.4665 | 0.2319 |
| | FAIL mean | | **2.5** | | 0.8150 | |
| | PASS mean | | **3.0** | | 0.6417 | |

Two things, both against the mean-collapse hypothesis:

1. **Criterion #1 points the wrong way** — FAIL is *less* multimodal than PASS.
2. **The two failing seeds are qualitatively opposite.** Seed 49 is the textbook
   setup (two near-equal attractors, separation 1.158). Seed 50 has **90% of the
   teacher's mass on one mode**, spread 0.0867 — effectively unimodal, with
   nothing to fall between — and failed just as hard (1.1818, zero
   improvements). Seed 42, which succeeded, is *more* multimodal than seed 50,
   which failed.

**Retraction.** I earlier called the smoke test "encouraging for mean-collapse".
That was seeds 49 and 42 only — the pair that happens to look supportive. The
four-seed version does not support it. This is the same selection effect this
project has now been bitten by twice (the retracted p=0.0371 headline, and the
criterion-drift the peer session caught at 58081bb), arrived at a third way.
**Carry the four-seed version, not the two-seed one.**

Status of this observation: iteration 0 of a trajectory that was never run,
n=2 vs 2, EXPLORATORY. It is suggestive against the hypothesis and is not
evidence for any alternative. Testing mean-collapse properly still requires the
trajectory measurement H50 was built for.
