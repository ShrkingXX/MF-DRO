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
