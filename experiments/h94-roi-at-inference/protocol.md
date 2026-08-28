# H94 — apply the ROI where the DRO paper applies it: to the QUERY

LOCKED BEFORE ANY RUN AND BEFORE THE IMPLEMENTING CODE IS WRITTEN.
Compute is at 14/15 (session B owns h90 + h93). NOT LAUNCHED. See Gate.

## Why

The code audit (findings.md, "the ROI has never been applied to a real query")
established that `use_roi=True` restricts only the TEACHER's rollout pool. The
real query is `action_head(h).clamp(0,1)` and is never told the ROI exists.

DRO Sec 4.2 defines X_hat_{m,t} = {x | UCB_{m,t}(x) >= max_x' LCB_{m,t}(x')} as
a constraint on WHERE YOU QUERY. This project has only ever implemented it as a
constraint on where the teacher DEMONSTRATES. H94 tests the paper's version.

This is the direct answer to the commissioned question -- "find an ROI strategy
that stops MF-DRO wasting HF budget on low-value regions" -- because a
constraint on the query is the only version of the ROI that can actually stop a
query from landing somewhere.

## The line this must not cross, and how the design polices it

Enforcing an ROI at inference is one bad step away from pool+argmax, which is
excluded because it erases the contribution. The rule adopted here:

  **The ROI may only declare which points are ADMISSIBLE. It may never rank
  admissible points. The DT alone chooses among them.**

Concretely: the DT emits x. If x is admissible, it is queried UNCHANGED. If it
is not, x is replaced by the nearest admissible point BY EUCLIDEAN DISTANCE TO
THE DT'S OWN OUTPUT. No acquisition function is evaluated at any point in this
operation. The DT remains the decision-maker; the ROI is a feasible set.

That argument is not self-certifying, so the design includes a control that
detects the failure empirically rather than trusting the reasoning -- see arm D.

## Arms (Borehole_8D, seeds 47-51)

| arm | training ROI | inference | new runs |
|---|---|---|---|
| A NO-ROI | off | none | 0 -- **reuse h90** |
| B ROI-Q10 | q=0.10 | none | 0 -- **reuse h90** |
| C ROI-PROJECT | q=0.10 | snap to nearest **in-ROI** candidate | 5 |
| D SNAP-CONTROL | q=0.10 | snap to nearest candidate of an **unfiltered** pool | 5 |

10 new runs. A and B are reused from h90 at the SAME seeds with byte-identical
config; reuse is legitimate for a CONTROL (the treatment C/D is a new mechanism
never tuned on any seed, so the selection bias reuse would introduce has no
channel here). If h90's A or B arm fails to complete 5/5, H94 reports on
whatever pairs exist and says so.

Pool size 600 and q=0.10 are FIXED IN ADVANCE, matching ROI-Q10. No variant of
C or D will be run on these seeds.

### Arm D is the whole point of the experiment

D snaps to a discrete uniform pool with NO ROI test. It therefore isolates
"quantizing the DT's output onto a finite pool" from "restricting it to the
ROI". Snapping alone can plausibly help -- it is a crude form of the pool
mechanism the user excluded -- and without D, a gain from C would be
uninterpretable.

**If D explains C, the honest conclusion is that the excluded pool mechanism
resurfaced, NOT that the ROI works.** That outcome will be reported in exactly
those words.

## Metric (frozen)

h83's `sr_curve`/`grid`, SR interpolated at cost 200, relative regret. Identical
to h84/h86/h87/h90. The evaluation is not touched -- only the method.

## Predictions (pre-registered, before the code exists)

**P1 (PRIMARY, the discriminator). C beats D, paired, on >= 4/5 seeds with a
negative mean.** This is the only comparison that isolates the ROI from
snapping. Registered as GENUINELY UNCERTAIN -- I am not predicting a direction
with confidence, and the reason is my own record: eight mechanism claims this
session, seven refuted, and every intervention that survived one seed set shrank
or died at the next. A confident positive here would be unearned.

**P2. C beats A (NO-ROI) on >= 4/5 with a negative mean.** Registered POSITIVE
but weakly. Note P2 can pass while P1 fails -- that is the D-explains-C outcome.

**P3. |C - A| > |B - A|**: enforcing the ROI at the query moves regret MORE than
the imitation channel does. This is the audit's central implication stated as a
number. Registered POSITIVE: if the audit is right, a hard constraint should
dominate a lossy imitation of that constraint.

**P4 (NEGATIVE). C still does not beat MF-MES (6.40 on Borehole).** Registered
negative; a 4-5 point gain does not close a 5-point gap from 11.59. A refutation
would be the most important result of the run and would need re-verification
before being believed.

**P5 (MECHANISM). In arm C, the fraction of real queries that required snapping
is > 0.5 at some point in the run.** If the DT's raw output is almost always
already in-ROI, then C is nearly identical to B by construction and P1/P3 are
uninformative regardless of how they come out. This is the "did the intervention
actually intervene" check that Lesson 23 exists to enforce; it is logged per
iteration, not inferred.

## Falsifier

If P1 fails AND P3 fails, the audit's implication is dead: applying the ROI to
the query rather than to the demonstrations does not help, and the ROI has no
surviving performance result anywhere. What remains is the controllability
finding alone (a constant beta cannot set ROI tightness), which is a claim about
parameterisation, not performance. That must be stated as plainly as it is here.

If P5 fails, C and D are reported as INCONCLUSIVE rather than negative, because
the manipulation did not take.

## Gate

**DO NOT LAUNCH while session B's h90/h93 occupy 14 workers.** H94 needs 10
slots and the cap is 15. Launch only when a corrected count
(`sh src/analysis/worker_count.sh`, which excludes launcher processes) shows
<= 5 workers. h90 must finish first regardless: it supplies arms A and B.

## Predecessor numbering

h91-h93 belong to the concurrent session. This session reserved h90+; to avoid a
third ID collision this experiment takes h94 and the reservation is narrowed
here: **this session holds h94-h99.**

## Amendment 1 — implementation written, held OUT of src/ as a patch, and UNVERIFIED

The inference-time ROI is implemented (`code/roi_at_inference.patch`, applies to
`src/policy/mf_dro.py`): a `_roi_snap` method plus a two-line hook immediately
after the [0,1]^d -> raw rescale. `roi_inference_mode` defaults to None, so the
hook is a no-op for every existing configuration.

### Why it is NOT in src/

A concurrent session has 15 workers running (h90, h93) and MORE JOBS STILL TO
LAUNCH. A newly launched worker imports `src/policy/mf_dro.py` at process start,
so an edit sitting in the working tree would be picked up by runs belonging to
somebody else's experiment. Already-running processes are unaffected (the module
is loaded once), but the not-yet-launched ones are not.

The edit is a no-op by inspection -- `getattr(config,'roi_inference_mode',None)`
returns None, the branch is skipped, no RNG is consumed. **That reasoning is
exactly the kind this session has been repeatedly wrong about, and the cost of
reverting is zero.** src/ is restored to HEAD; the change lives as a patch that
has been verified to re-apply byte-exactly.

### TWO GATES, both required before any H94 run

**G1 -- BIT-IDENTITY, not yet performed.** A `use_roi=False` run on identical
seed must produce an identical x/fidelity/y trace with and without the patch.
The check was written and launched; it exceeded a 2-minute limit because a
MF-DRO run trains for bo_iterations=4000 before its first real query, and it was
abandoned rather than left competing for cores at the cap. **No H94 result may
be reported before G1 passes.** If G1 fails, the patch is wrong and H94 does not
run at all.

**G2 -- COMPUTE.** `sh src/analysis/worker_count.sh` <= 5, and h90 complete
(it supplies arms A and B).

### Recorded as a deviation from my own plan

The protocol above says the ROI test at inference reuses the rollout's
quantile-calibrated beta. It does, with one difference worth stating: the
rollout resolves beta ONCE on its first draw and reuses it across draws within a
step; `_roi_snap` draws a single pool and calibrates on it. Same estimator, one
draw instead of several -- a slightly noisier acceptance rate at the same target
q. Noted here rather than discovered later in a diff.

## Amendment 2 — a zero-compute diagnostic, registered BEFORE it is computed

The audit calls the ROI's route to a real query a "lossy imitation channel".
That is an assertion about a quantity the runs ALREADY LOG: `L_loc_per_iter`,
the MSE between the head's emitted location and the teacher's action. Lesson 23
says measure the quantity the mechanism operates on; this one costs nothing.

EXPLORATORY (the data exists; nothing is re-run). Registered before computing:

**D1.** L_loc is substantially non-zero throughout the run -- the head does NOT
converge onto the teacher's actions. If instead L_loc goes to ~0, the channel is
NOT lossy, the audit's central adjective is wrong, and h94's premise weakens
sharply: a faithful imitator of an in-ROI teacher would already be querying
in-ROI, and applying the ROI at inference would change little.

**D2.** L_loc is comparable between ROI-on and ROI-off. The ROI moves WHERE the
teacher points, not how well the head tracks it. If ROI-on has a much LARGER
L_loc, the ROI is making the teacher harder to imitate -- which would be a
mechanism for the ROI actively HARMING some benchmarks (Currin, +0.11).

**D3.** Interpretation limit, stated in advance: L_loc is a TRAINING loss over
rollout actions, not a measure of the gap at the real query. A small L_loc would
not prove the real query lands in-ROI. So D1 can weaken h94's premise but cannot
establish it, and no verdict on h94 will be drawn from this diagnostic.

## Amendment 3 — G1 attempted a second time, killed, and the reason recorded

Second attempt at the bit-identity gate. Run from two COPIES of the tree in
scratchpad (HEAD vs patched) so the live `src/` was never touched, at a cheap
config (bo_iterations=60, num_epochs=1, budget 25) on the reasoning that the
risk G1 tests -- RNG or state perturbation by a no-op hook -- shows up at any
training length.

KILLED after ~7 minutes, incomplete. **Not because it failed: because I was the
16th process on a 15-core cap.** The concurrent session had gone from 14 workers
to 15 while the check was queued, and my diagnostic made 16. It was niced to 19,
which is a mitigation and not a compliance argument.

I withdrew a false accusation about this exact rule earlier today. Holding
myself to it when it costs me the gate I want is the same rule.

G1 REMAINS UNMET. h94 is blocked on it, and blocked on G2 (workers <= 5)
regardless. The next attempt goes when h90 finishes and frees slots -- its
REFINE-100 arm has been running 2h17m and is the long pole.

### An observation recorded WITHOUT a conclusion

System load average read 38-39 during this window against 16-17 earlier. The
reliable measurement -- 16 processes at ~90% CPU, 53% memory free -- does not
support a thrashing diagnosis, and Darwin's load average counts states that
inflate it relative to the Linux reading I would instinctively apply. Recorded
because it is unusual, NOT flagged as a problem: I raised one alarm today by
over-reading a load-adjacent number and will not do it twice.

## Amendment 4 — G1 PASSED

Third attempt, run from two scratchpad copies of the tree (HEAD vs patched) so
live `src/` was never touched, under a wrapper that aborts if the worker count
exceeds 14. Compute was at 14, so one diagnostic slot sat exactly at the cap.

    use_roi=False, Borehole seed 47, identical config
    HEAD:    13 real iterations
    patched: 13 real iterations
    fidelity/x/y traces: BIT-IDENTICAL (exact float repr comparison)

The patch tested is the CURRENT one, including the `Var(actions_x)` logging
added after D2 -- not the earlier version. **G1 is MET.**

G2 is NOT met. h90's arms A and B (NO-ROI, ROI-Q10) are complete 5/5, which is
what h94 consumes, but 13 workers are running and h94 needs 10 slots. Its
REFINE-100 arm (h90's own P4, unrelated to h94) is the remaining long pole.

## Amendment 5 — GATE DEVIATION, recorded BEFORE launching

G2 as written required `workers <= 5`. Six are running. **I am launching 8 of 10
jobs at 6 workers: 6 + 8 = 14, inside the cap of 15.**

The gate's purpose was the compute cap, not the number 5. Recorded as a
deviation rather than silently reinterpreted, and recorded BEFORE the launch.

### Both preconditions the gate actually protects are verified, not assumed

**h90 is COMPLETE, 15/15.** It supplies arms A (NO-ROI) and B (ROI-Q10), both
5/5. Its REFINE-100 arm also finished: -3.29, 5/5, P4 and P5 MET.

**No launcher can spawn a worker into a patched src/.** This was the hazard that
kept the patch out of the tree for four hours. h93's launcher process is still
alive, so I checked its accounting rather than trusting the report that it had
nothing left: 14 completed + 6 in flight = 20 of 20 jobs. Nothing is queued, so
no new process will import `mf_dro.py`. Already-running workers loaded the module
at start and are unaffected.

**G1 passed** on this exact patch, including the Var(actions_x) logging.

### Why 8 and not 10

h94's runner interleaves by seed so PAIRS complete together (a paired comparison
needs both arms of a seed; five half-pairs are useless). 8 jobs = seeds 47-50 at
both arms = four complete pairs. Seed 51's two jobs are SKIPPED at launch and
will go when slots free, giving the fifth pair. One slot is left as margin.

Skip arguments are passed as separate literal arguments, not via an unquoted
variable expansion -- the concurrent session's 26-worker over-launch came from
zsh not word-splitting an unquoted expansion, so 16 skip-triples arrived as ONE
argv element and every job launched.

## Amendment 6 — G3 added AFTER an 8-job failure it would have prevented

Chronology stated plainly: the 8-job launch under Amendment 5 crashed with a
NameError on every job (see findings.md). This gate is added AFTER that failure,
not before it, and is not backdated.

**G3. THE ON PATH MUST BE EXECUTED BEFORE LAUNCH.** For any change whose
treatment is gated by a new config flag, the flag must be switched ON and the
code path run end-to-end at minimal scale, with the intended side effect
OBSERVED -- not merely the absence of a crash. For h94 that means: run both
`project` and `snap_control` at a tiny budget and confirm `roi_inf_log` is
non-empty and records snaps.

Rationale: G1 (bit-identity, OFF path) passed and could not catch an ON-path
error by construction. P5 (manipulation check) only reports at the end of a full
run and reports nothing at all when the run crashes. The two gates in place both
had blind spots on the same path.

G3 must pass before h94 relaunches. Its result will be recorded here whichever
way it goes.

## Amendment 7 — G3 PASSED, and it changes how h94 must be read. Recorded with 0 results.

    MODE project       11 real queries   snapped 11/11   accept 0.099   snap dist 0.7317
    MODE snap_control  13 real queries   snapped 13/13   accept 1.000   snap dist 0.5400
    Var(actions_x) logged in both (11 and 13 entries).

Both code paths execute, the quantile calibration hits its 0.10 target (0.099),
the unfiltered control accepts everything (1.000) as designed, and no ROI came
back empty. **G3 is MET.** h94 may launch.

### THE DT'S RAW PROPOSAL IS NEVER ADMISSIBLE. 11 of 11.

P5 asked whether >50% of queries required snapping. The answer is 100%, and
that is not the comfortable outcome. It means the intervention is never
"the DT proposes, the ROI occasionally corrects". **Every single query is
replaced**, and by a mean normalized distance of 0.73 in a cube of diameter
2.828 -- 26% of the domain, and MORE than the student-teacher gap L_loc implies
(0.55).

So what h94 actually tests is: *the nearest in-ROI pool member to the DT's
output*. The DT supplies a direction; the pool supplies the point. That is a
much heavier mediation of the DT's role than "declare which points are
admissible" suggested when I wrote the rule, and it must be stated in the
result whichever way the numbers fall.

**It is still not pool+argmax** -- no acquisition function ranks the candidates,
and which candidate wins depends entirely on the DT's output. But the honest
description is no longer "the DT decides and the ROI constrains"; it is "the DT
picks a point in candidate space by proximity". If h94 shows a gain, that
distinction is the first thing a reader will press on, and it should be
volunteered rather than defended.

### A confound in the C-vs-D comparison, named before results

    ROI-PROJECT   snap distance 0.73   (pool filtered to ~10%)
    SNAP-CONTROL  snap distance 0.54   (pool unfiltered)

D snaps LESS FAR, necessarily: an unfiltered pool has more candidates near any
given point. So C and D differ in ROI-membership AND in how far the query moves.
D remains a valid control for "does quantizing onto a finite pool help at all",
which is what it was built for and what the excluded-mechanism check needs. It
is NOT a distance-matched control, and P1 therefore cannot separate "the ROI
helps" from "moving further helps". Stating this now so it is not discovered in
the numbers later.

A distance-matched control is constructible (snap to the nearest member of an
unfiltered pool subsampled to the same size as the surviving ROI set) and is the
obvious follow-up if P1 comes out positive. Not run now: it would be a third arm
chosen after seeing G3, and h94's arms were fixed in advance.

## Amendment 8 — a THIRD confound, recorded with 0 results written

Observed from live checkpoints, before any h94 result file exists and therefore
before any regret number can be read. HF share of post-init queries:

    arm              47    48    49    50    51    mean
    NO-ROI (h90)   0.93  0.98  0.63  0.88  0.98   0.88
    ROI-Q10 (h90)  0.78  0.92  0.53  0.57  0.94   0.75
    ROI-PROJECT    0.76  0.89  0.54  0.74  0.95   0.78
    SNAP-CONTROL   0.59  0.77  0.59  0.60  0.83   0.68

**The arms differ in FIDELITY MIX, not only in where they query.** SNAP-CONTROL
spends materially less of a matched cost budget on high fidelity than
ROI-PROJECT does (0.68 vs 0.78), and both differ from the no-ROI control.

So C and D now differ in THREE ways, only the first intended:

    1. ROI membership of the admissible set   -- the intended contrast
    2. snap distance   0.73 (C) vs 0.54 (D)   -- Amendment 7
    3. HF share        0.78 (C) vs 0.68 (D)   -- this amendment

**If C beats D, any of the three could be responsible, and P1 cannot separate
them.** Fewer HF queries at matched cost is a plausible independent cause of
worse regret on its own, quite apart from the ROI.

### Why this was not foreseeable and is not an excuse

The snap changes only x, never the fidelity. The mix shift is INDIRECT: the
snapped location changes the state the DT conditions on, which changes what its
fidelity head emits on subsequent steps. Nothing in the design touched fidelity
and I did not anticipate that it would move.

That is an explanation, not a defence. The intervention was checked for whether
it fired (G3, P5) and not for what else it perturbed, and "what else did this
move" is a question that should be asked of any intervention before its verdict
rather than after.

### Consequence for how h94 is reported

P1's verdict stands as registered and will be read as registered. But the
INTERPRETATION attached to any positive P1 is now: "C beats D, and C differs
from D in ROI membership, snap distance and fidelity allocation." Attributing
the difference to the ROI specifically would require an arm that matches D to C
on distance and HF share, which does not exist and is not being added after the
fact.

A negative or null P1 is comparatively clean: if the ROI applied to the query
does NOT beat unfiltered snapping despite also getting more HF queries, that is
a stronger negative than the design was built to deliver.
