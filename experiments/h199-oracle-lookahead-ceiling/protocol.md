# h199 — the CEILING of the lookahead schema under oracle access

**CONFIRMATORY**, and it measures a **CEILING**, not a method. **Human-requested.**
Locked before any code is run.

## The question

h198 gives the lookahead teacher only the GP. It therefore plans through a surrogate,
and its "optimal decision" is optimal *under a possibly-wrong model*. h199 removes that
limitation and asks what the schema is worth when the model is perfect.

**This decomposes a null in h198 into two very different diagnoses:**

| h198 (GP) | h199 (oracle) | what it means |
|---|---|---|
| fails | **works** | the SCHEMA is fine; the GP's fantasies are the bottleneck |
| fails | **fails** | the schema/DT is the bottleneck; surrogate quality is irrelevant |
| works | works | teacher design is a live lever (h198 already settles this) |

Without h199, a null in h198 is ambiguous in exactly the way h145's was.

## Design — one change, and only one

The decision rule is **byte-identical** to h198's. The single substitution is inside the
imagined future: wherever the teacher draws a fantasy

    y = ko.sample_fantasy(x, "LH"[ell])          # h198: GP posterior draw

it instead evaluates the truth

    y = f_H(x) if ell == 1 else f_L(x)           # h199: oracle

Everything else — the MES shortlist, the CRN pairing, the greedy base policy, the
step-bounded horizon, `argmax` of expected terminal best value, the GP conditioning —
is unchanged. So h199 - h198 isolates **fantasy quality** and nothing else.

Note this makes the imagined future *deterministic*, so M replications become identical
and M is set to 1. That is not a config choice being smuggled in: with a deterministic
oracle, M>1 would be M identical copies. CRN becomes a no-op for the same reason.

**This is NOT a method.** f_H is not available at run time in any real setting. It is a
ceiling, and will be labelled as one everywhere it is quoted.

## The trap h146 already identified, and the SC that guards it

h146 established that h145's oracle confounded **QUALITY** with **zero endpoint
DIVERSITY**: if every trajectory ends at the same point, the DT's target has a large
state-independent component and the head fits a degenerate map. "Better trajectories
hurt" and "degenerate targets hurt" predict the same outcome.

h199 should NOT collapse, because the candidate shortlist still comes from MES on the
**current GP state**, which differs across rollouts — but that is an argument, and h146
is precisely the lesson that such arguments need measuring.

> **SC-DIVERSITY (GATE, registered before running).** Measure the across-rollout SD of
> the teacher's tau=0 action and its endpoint dispersion, against the MES teacher's own
> values on the same states.
> - If oracle-lookahead's tau=0 SD is **< 25%** of the MES teacher's, the arm is
>   **degenerate** and its result cannot be attributed — report as a GATE MISS, not a
>   result, exactly as h146 requires.
> - Otherwise Stage 1 runs.

Two further SCs, both registered now:
1. At `n_c=1` it still reduces exactly to greedy MES (the h198 SC1 identity must survive
   the oracle substitution).
2. The oracle is actually being consulted — the chosen action must differ from h198's on
   a measurable fraction of starts. If it does not, the substitution is a silent no-op.

## Prediction, committed before running

The confirmed mechanism says the DT emits its teacher's tau=0 action mean, so this arm
helps iff the oracle improves that mean **without** collapsing its variance. Because the
oracle sees the truth, its tau=0 choice should be markedly better than MES's.

- **P1** — beats the ROI-Q10 control (11.59). The lookahead schema has real headroom and
  h198's job is to close the gap to it.
- **P2** — no effect despite SC-DIVERSITY passing. **This is the strong result**: even a
  teacher choosing optimally under the TRUE function cannot move the DT. That would say
  the bottleneck is not teacher quality at any level, and would close the front far more
  firmly than any GP-limited arm can.
- **P3** — worse than the control while SC-DIVERSITY passes. Would mean good decisions
  actively harm the DT for a reason not yet identified — and would NOT be explainable by
  the h146 degeneracy account, since the gate excludes it.

## What each outcome RETRACTS

- **P1** retracts the implicit reading that teacher-shaping is a dead lever, and demotes
  the "declined on measured premises" status of the teacher-design arms.
- **P2** retracts nothing, but it would make the mechanism's *sufficiency* claim
  unsalvageable: no teacher, however good, reaches the DT.
- **P3** would retract the h146 QUALITY-vs-DIVERSITY resolution as complete, since a
  non-degenerate high-quality teacher would still be harming.

## Cost

The oracle substitution makes the future deterministic, so M=1 instead of 4 — this arm
is roughly **4x CHEAPER** than h198 per decision. Borehole seeds 42-46, 5 workers,
against the control already in hand. Runs only when h197/h198 free slots; the cap is 15.
