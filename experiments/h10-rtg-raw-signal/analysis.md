# H10 analysis — VOID, and the void yields the actual structural answer

## Result

| arm | realised band | width | argmax moved |
|---|---|---|---|
| CONTROL (`normalized`) | [0.5672, 1.0000] | 1.76x | 0/12 |
| RAW (`raw`) | [0.5628, 1.4560] | **2.59x** | 0/12 |

**Manipulation check FAILED** (required >= 5x, got 2.59x). Per protocol the
experiment is **VOID** and prediction 2 must not be interpreted. Second
consecutive void on this question — and the reason is the same one both times.

## The structural fact I should have derived before running anything

    target = max(batch_max, alpha_rtg * running_max)
    batch_max <= running_max            (running_max is the max OVER batch_max values)
    => target in [alpha_rtg * running_max, running_max]
    => band ratio <= 1 / alpha_rtg      ALWAYS

At `alpha_rtg = 0.5` the band **cannot exceed 2x**, whatever the reward is,
whatever the normalisation is. Measured: 1.76x normalised, 2.59x raw — both at
or near that ceiling (raw slightly exceeds 2x only because `running_max` is
itself growing during the run, so the floor tracks a moving target).

**This is provable in two lines and needed no compute.** H9 and H10 were both
doomed by the same cap:

- **H9** shrank `alpha_rtg` to 0.1 — which raises the *ceiling* to 10x, but with
  the batch-max normalisation the floor then never binds and the target collapses
  to `batch_max`, whose own range is narrow. Void.
- **H10** replaced the normalised quantity with a raw one — but left
  `alpha_rtg = 0.5`, so the 2x cap still applied. Void.

Neither intervention could have worked alone. **Both are required**: a small
`alpha_rtg` (to lift the cap) *and* an intrinsically varying signal (to fill the
space the lifted cap allows).

## Honest assessment of my own experimental design

I ran two experiments, ~30 minutes of compute, that a two-line algebraic check
would have ruled out in advance. The pre-registered manipulation checks did their
job — both experiments declared themselves void rather than producing an
interpretable-looking number — but the checks caught the problem *after* the fact
when the algebra would have caught it *before*.

The generalisable lesson: **when an experiment manipulates a quantity computed by
a formula I control, derive the formula's reachable range first.** A
manipulation check is a safety net, not a substitute for reading the code.

## Status of the confound

Still open. The inert-vs-starved question is unresolved, and it now has a
precise, testable form:

> Set `alpha_rtg` small enough that the floor never binds AND use a signal with
> intrinsic across-iteration variation (`rtg_target_mode="raw"`). If the argmax
> still does not move, the insensitivity is the network's.

That combination has never been run. It is the single remaining experiment
standing between this project and a defensible claim about the DT's conditioning.
