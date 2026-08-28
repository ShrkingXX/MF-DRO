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
