# h168 -- is the collapse caused by the INFERENCE CONDITIONING?

STATUS: protocol locked, nothing run (queued: compute is at 14/15).
TYPE: CONFIRMATORY.

## Why this, and why it is not proxy-shopping

h167 established two things that together relocate the failure:
  - the failing arms' DT fits its teacher **2.5-4x better than any constant**
    (L_loc 0.018-0.022 vs best-constant 0.053-0.083), so the actions ARE learnable;
  - yet its real queries land within **0.04 of the box centre** in 8 dimensions,
    while every working arm sits 0.74-0.78 away.

Training succeeds; inference collapses. The one thing that differs between them
is the **conditioning**: at inference the DT is asked for an action at
`rtg_target = max(batch_max, 0.5*running_max)`, the extreme upper tail of the
training returns.

**On the h148 stopping rule.** h148 registered: "if P1 fails, I stop trying to
explain h145 with RCSL theory." Its recorded outcome was **"P1 not evaluable;
P2 decisive"** -- P1 was never evaluated, because the per-record RTG statistics
it needed were not serialised (one of the five registration-against-unserialised-
data failures this project has logged). So the rule's trigger never fired, and
h148's central question was never answered.

That is not a licence to resume proxy-shopping, and this is not a proxy. h167
arrived at the conditioning by **direct measurement** from an unrelated
direction (where the queries land, and how well the network fits), not by
generating another RCSL-flavoured statistic until one fit. An independent line
converging on a suspect is evidence; a third proxy would not be. Stated here so
the distinction is on the record before the run, not argued afterwards.

## The measurement

Re-run ONE failing arm with a probe added at the inference call site
(mf_dro.py:3224, `self.dt.propose_mf(state, rtg_tgt, btg_now, ...)`). At every
real iteration, after the DT is trained, query it at the SAME state with a
sweep of RTG values and record each emitted x:

  the real `rtg_target` (~0.30 for a failing arm)
  the batch mean rtg[0]                     (in-support, ~0.01)
  a few intermediate values

The proposal is READ ONLY -- the probe's outputs are recorded and discarded, and
the arm's own actual query is the unmodified one, so the run stays comparable to
the existing arm.

## Predictions

P1 At the real `rtg_target`, the emitted x sits near the box centre
   (d < ~0.15 in normalised space, matching h167's 0.024-0.041).
P2 At an in-support RTG, the emitted x is materially FURTHER from the centre.
   Operationalised: d(centre) at the in-support value exceeds d(centre) at
   `rtg_target` by at least 2x.
P3 A working arm probed the same way shows NO such gap -- its emitted x is far
   from the centre at every RTG value.

## What this can RETRACT

R1 P2 fails -- the emitted x is at the box centre at EVERY conditioning value ->
   the conditioning is NOT the cause. The collapse would be unconditional, and
   the suspect becomes the trained network itself (initialisation, output
   parameterisation, or the state features being uninformative in these arms).
   That would close the conditioning line for good, this time evaluably.
R2 P2 holds -> the conditioning is implicated, and the natural intervention
   (condition on an in-support quantile rather than the running max) becomes
   testable. This would be the first ACTIONABLE finding on this front.
R3 P3 fails (working arms also collapse at high RTG) -> the effect is real but
   not what separates working from failing arms, so it does not explain the 2x2.

## Design

One failing arm (RANDOM-POOL) and one working arm (control) as P3's comparison,
on **Hartmann** -- ~4x cheaper per seed than Borehole (19 min vs 80). Seeds
42-46, n=5 each, 10 workers. Queued behind the three arms currently running;
compute rule (<=15) is why it is not launched now rather than any doubt about
priority.

## Smoke test — probe VERIFIED, and an early signal recorded before the arm ran

Tiny-budget run (5 real iterations), Borehole, RANDOM-POOL arm:

  probe iterations recorded   5          PASS (the probe swallows exceptions, so
                                         a scope error would look like a no-op)
  sweep length per iteration  9 of 9     PASS
  emitted x varies with RTG   True       max spread across the sweep 0.0276

**Early signal, recorded now so it is not reported as a discovery later.** At
5 iterations the emitted x sits ~0.38 from the box centre at EVERY conditioning
value, and moves only 0.028 across the entire sweep from rtg=0.00 to rtg=1.00:

    rtg=0.00  d(centre)=0.3797       rtg=0.75  d(centre)=0.3910
    rtg=0.02  d(centre)=0.3808       rtg=1.00  d(centre)=0.3938
    rtg=0.05  d(centre)=0.3825

P2 requires d(centre) at an in-support RTG to exceed d(centre) at `rtg_target`
by at least 2x. Here the ratio is ~0.96 — no effect at all.

**This is 5 iterations of a barely-trained network on a truncated budget, and
h167's collapse was measured over a full ~60-iteration run.** It is not a result
and no verdict is taken from it. But if the pattern holds at full length, **R1
fires**: the conditioning is exonerated, the collapse is unconditional, and the
suspect becomes the trained network itself. That would close the conditioning
line evaluably — which is exactly what h148 failed to do.
