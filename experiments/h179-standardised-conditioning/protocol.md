# h179 -- does making the conditioning responsive actually help?

STATUS: protocol locked, NOT launched. Queued behind h178.
TYPE: CONFIRMATORY.

## The claim under test, and the one NOT under test

h177 found both conditioning inputs inert; the module-level cause is that raw
scalars into `Linear(1->H)+LayerNorm` saturate, and BTG's operating range (~28)
sits deep in the saturated regime. Z-scoring restores responsiveness by **336x**
(0.0056 -> 1.8817 relative embedding change).

**That the channel becomes responsive does NOT imply the method improves.** The
DT may be better off ignoring a signal it cannot use; nothing measured so far
says the conditioning *should* be active. This arm tests the improvement, which
is a separate question from the mechanism.

## The arm

Standardise `rtg_target` and `btg_now` by their running mean and sd before they
reach `reward_embedding` / `btg_embed`. Everything else the control's config.

## Predictions -- and this is genuinely two-sided

P1 Regret improves: a responsive conditioning channel lets the DT use the target
   it is given, which is what the architecture intends.
P2 Regret is unchanged: the conditioning carries nothing useful, and the tau=0
   account (the query is a function of the state alone) is unaffected by making
   an unused channel usable.
P3 Regret degrades: a responsive channel injects a signal the DT was previously
   protected from -- plausible given h153/h161/h166/h171/h172 all showed arms
   with COLLAPSED targets performing fine, i.e. the target carries little
   information about what a good query is.

**P3 is not a throwaway.** Six arms on this front have paired a collapsed
conditioning target with good performance. If the target is uninformative, wiring
it in properly could actively hurt.

## What this can RETRACT

R1 P3 holds -> the "fix" is not a fix, and the finding must be reported as "the
   conditioning is inert, and that is load-bearing" rather than as a defect.
   findings.md currently calls it a defect with a fix; that framing would go.
R2 P1 holds -> the first performance improvement this front has produced from a
   mechanism rather than from removing work.
R3 P2 holds -> the channel is genuinely irrelevant either way; the tau=0 account
   is strengthened and the "defect" framing is downgraded to a curiosity.

## Prerequisite

**h178 must report first.** If it refutes the saturation account (trained
btg_resp comparable to rtg_resp), this arm's premise is gone and it should not
run in this form.

## Design

Borehole_8D seeds 42-46 against the h83 control. Requires a gated code change to
the embedding path, a bit-identity gate, and a smoke test showing the
standardisation actually reaches the modules.
