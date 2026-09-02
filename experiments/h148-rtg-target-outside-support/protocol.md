# h148 — Is the inference RTG target OUTSIDE the oracle data's support?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY. **POST-HOC** — this is a second proxy proposed after h147's
first proxy failed, and it is labelled so at the top rather than in a footnote.

## The refined claim

Brandfonbrener's return coverage is `alpha_f = P_beta(g = f(s) | s)`: the chance
the behaviour data actually ACHIEVES the conditioned return at that state. h147
tested return *spread* and it went the wrong way. This tests return *location*.

h147 measured that **over half the oracle's returns are negative** (0.552 vs
0.258). Inference conditions on `rtg_target`, which is a positive quantity. If the
target sits in a region the oracle's own trajectories rarely reach, `alpha_f` is
small — coverage failure through displacement rather than through narrowness.

## Prediction (locked)

Borehole, seeds 42-46, paired. Using each run's logged `rtg_target` (the inference
conditioning value) and its per-iteration RTG statistics:

**P1.** The **gap between the inference target and the trajectories' achieved
returns is LARGER in the oracle runs**, effect >= 1.0. Operationalised as
`rtg_target` minus the run's mean realised `rtg[0]` proxy, per iteration, averaged.
FALSIFIED if effect < 1.0.

**P2 (no direction).** `rtg_target` itself, oracle vs control — reported. If the
target is identical across arms, any gap difference comes purely from what the
trajectories achieve, which is the cleaner reading.

## What this could RETRACT — and the stopping rule

**If P1 fails, I stop trying to explain h145 with RCSL theory.** Two proxies from
the same frame, both pre-registered, both failed, would mean the frame does not
reach this result — and continuing to generate proxies until one fits is precisely
the failure mode this project has documented five times.

**Stopping rule, registered now:** if P1 fails, the recorded conclusion becomes
"the oracle teacher degrades MF-DRO markedly and we do not know why", and the next
step is the h146 quality-vs-diversity contrast already running, not a third proxy.
