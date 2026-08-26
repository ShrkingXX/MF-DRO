# H76 — Where in the run does SF-DRO's Hartmann advantage appear?

**CONFIRMATORY.** Locked before any h76 number exists.

## Why

h73/h74 settled *that* SF-DRO beats SF-MES on Hartmann (+12.71 pts, 10/10,
p=0.0020) and loses on Borehole and Currin. **Why Hartmann differs is
unexplained**, and it is the only positive result this project has.

This project's record on *proposing* mechanisms is poor — six proposed and
refuted (LF quality, local refinement, boundary aversion, fidelity allocation,
rho misspecification, acquisition class). So this measures rather than proposes:
it asks only *when* in the run the gap opens, which is a descriptive fact that
constrains any future mechanism without asserting one.

SF-DRO already records `regret_curve`. h72's SF-MES worker recorded only
`final_regret`, so the comparison needs SF-MES re-run with curves. SF-MES costs
**seconds** per run.

## Design

Re-run SF-MES, seeds 42-51, all three benchmarks, recording the full regret
curve. Identical optimizer, config and seeds to h72.

**Built-in control is real here, not vacuous.** Because the same seeds are re-run
with the same code, every `final_regret` must reproduce h72's **bit-for-bit**.
Unlike h73's control this cannot pass by reading the other experiment's files —
these are fresh runs of the same cells. Enforced in the verdict script.

Then, per benchmark, compare mean regret curves iteration by iteration.

## Locked predictions

1. **PRIMARY.** On Hartmann, SF-DRO's mean regret drops below SF-MES's **final**
   mean regret by iteration **12 of 25** (the first half). That is the
   "SF-DRO gets there early" signature.
2. **SECONDARY.** On Hartmann the gap between the mean curves is **monotonically
   non-decreasing over the second half** (iterations 13-25) — i.e. SF-DRO keeps
   pulling away rather than being caught.
3. **NULL.** The gap opens only in the last third (iterations >= 17). Then the
   advantage is late-stage refinement, not early search, and prediction 1 is
   wrong about the shape.
4. **CONTROL.** On Borehole and Currin, where SF-DRO loses, no such early
   crossing occurs.

## What this cannot settle

It localises *when*, not *why*. A curve shape is consistent with many mechanisms
and identifies none. It also cannot separate SF-DRO's DT policy from its 10-model
GP ensemble — SF-MES uses a single GP, so the two differ in more than the policy,
and this experiment does not isolate that.
