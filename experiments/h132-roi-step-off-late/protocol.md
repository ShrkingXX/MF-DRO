# h132 — If the late STALL is over-restriction, switching the ROI OFF should recover it

STATUS: LOCKED before any run. NOT YET LAUNCHED.
TYPE: CONFIRMATORY, and it tests **my own post-hoc hypothesis** — see the
      weakness statement below, which is the most important section here.
COMPUTE: queued BEHIND the peer's h123. Does not launch until h123 is fully
      dispatched. h127 seed 51 also outranks it.
READ POINT: post-init cost, h83 frozen `sr_curve` + `grid`, reported at 25, 50,
      100, 200 — not 200 alone, since a schedule mixes regimes.

## The finding this follows from

h131 (exploratory) measured that the ROI's Borehole advantage erodes +3.588% of
optimum between cost 100 and 200, effect 1.40, 5/5 seeds, and that the mechanism
is a **stall**: over that span the control's regret falls 5.193% of optimum while
the ROI's falls 1.605%. The ROI stops improving; the control does not.

## The arm

`ROI-STEP` = q=0.10 until post-init cost 100, then `use_roi=False` for the
remainder. A step, not a ramp. The peer's h123 tests the paper's smooth `beta_t`;
this isolates the mechanism with maximum contrast, because a step confounds
nothing with the schedule's shape.

## Predictions (locked)

**P1.** ROI-STEP beats constant q=0.10 at cost 200, >= 4/5 seeds, effect >= 1.0.

**P2.** ROI-STEP is **indistinguishable** from constant q=0.10 at cost 100 —
effect < 0.5 — because the two arms are byte-identical before the switch. This is
a **manipulation check, not a finding**: if it fails, the implementation is wrong
and P1 is void regardless of which way P1 landed.

**P3.** ROI-STEP does not fully close to the control's late progress: it recovers
part, not all, of the 3.588%. Registered so that "recovers everything" would be
as surprising as "recovers nothing", and neither can be narrated as success.

**Falsified if** ROI-STEP is worse than constant q=0.10 at cost 200 with effect
>= 1.0. That would mean the late restriction is doing useful work and the stall
has another cause.

## THE WEAKNESS, stated first

**This design is tuned to my own hypothesis, which was post-hoc.** I ran the
erosion analysis after seeing the curve that suggested it, then built the arm most
likely to confirm it. That is a weaker evidential position than the peer's h123,
which registered no direction and was not designed around my idea.

**So h123 adjudicates my prediction and h132 does not.** If both land the same
way, h123 carries the weight and h132 explains the mechanism. If they disagree,
h123 governs. Recording this before either runs, because afterwards it would be
indistinguishable from an excuse.

## A read-point hazard this creates, recorded now

h131 showed Borehole's benefit PEAKS at −7.813% (cost 100) and DECLINES to
−4.224% (cost 200). The peak is the more flattering number and it sits exactly at
this experiment's switch point.

**The frozen metric is cost 200. −4.224% remains the headline.** The peak is a
mechanistic fact, not an alternative headline, and quoting it as the ROI's benefit
would be precisely the read-point flexibility this project has spent the day
guarding against. Registered here so that neither session drifts into it.

---

## AMENDMENT 1 — P4/P5, a ZERO-COMPUTE limb testable now (registered before computing)

The primary question states the causal path: *the DT regression head emits x
directly and is trained on rollout-teacher actions drawn from `roi_candidates`,
so the ROI is the lever that shapes the training distribution.* If the late stall
is real, it should be visible in the DT's own learning signals, and those are
already logged per iteration (`L_loc_per_iter`, 115 entries per run, alongside
`cost_curve` so iterations can be mapped to the post-init cost axis).

**Mechanism under test.** Late in the run the ROI's accepted set has contracted
around a region the DT has already learned. Its targets stop moving, so the head
fits them well while the proposals stop exploring — low loss, flat regret.

**P4 (locked direction).** Over iterations with post-init cost > 100 — the span
where the advantage erodes — the ROI arm's mean `L_loc` is **lower** than the
control's, paired over seeds 42-46, effect >= 0.5 and >= 4/5 seeds.

**Falsified if** the ROI's `L_loc` is *higher* with effect >= 0.5. That would mean
the ROI makes the DT's targets harder to fit rather than easier, and the stall
needs a different account.

**P5 (NO direction registered, reported either way).** `grad_coherency_per_iter`
and `action_reward_corr_per_iter` over the same span. I have no grounds for a
sign on either, and grad_coherency collapses from 0.91 to 0.01 within a single
control run, so its late values may be uninformative in both arms. Reported as
descriptive, and **a null here is not evidence for P4.**

**Scope limit, stated first.** P4 is a *correlational* test on existing runs. Low
late-run `L_loc` alongside flat regret is consistent with the training
distribution having collapsed, and equally consistent with the DT having simply
converged for reasons unrelated to the ROI. **Only h132's step arm can separate
those**, because only it removes the restriction while holding everything else
fixed. P4 therefore cannot confirm the mechanism — it can only fail to find the
signature the mechanism requires, which would weaken h132's premise before its
runs are spent.

---

## AMENDMENT 2 — what this result could RETRACT (added before launch)

Every protocol I wrote today looked only forward: it predicted outcomes and never
named a standing claim the result could cost me. That omission is common to all
four of today's failures, and the one time a protocol *did* name such a claim
(h135's caveat on "0.11 from additive") was the only instance where I caught my
own uncomputed spread rather than the peer catching it.

So, before h132 runs, the claims **I currently hold that its result could
force me to withdraw**:

1. **"The ROI's late stall is caused by over-restriction."** If ROI-STEP does not
   beat constant q=0.10 at cost 200, the stall exists (h133's extreme contrast is
   solid) but my causal account of it is wrong. I would have to withdraw the
   over-restriction reading and, with it, the motivation I handed the peer for
   h123's widening direction.

2. **"h132 P4's `L_loc` signature supports the training-distribution account."**
   That result is correlational by my own registration. If the step arm shows no
   regret recovery, the `L_loc` finding becomes a description of the DT
   converging, not of a distribution collapsing — and I should say so rather than
   keeping it as ambiguous support.

3. **"Escaping q~0.10 is what matters" (h133).** h133 is correlational across
   arms whose tightness ran the whole time. If the step arm fails while h123's
   ramp succeeds, then *late* tightness was never the operative variable and
   h133's step-not-gradient reading was about whole-run tightness only.

4. **My prediction to the peer** that a widening schedule should recover part of
   the 3.588% erosion. h123 adjudicates that, not h132 — but a null here is
   evidence against it and I should not quietly let h123 carry the verdict alone
   if h132 goes the wrong way first.

**None of these are hedges.** Each names a specific sentence in findings.md that
would need striking, so that a null produces a retraction rather than a
reinterpretation.
