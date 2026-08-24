# H9 analysis — VOID by its own pre-registered manipulation check, and that is useful

## Result

| arm | realised band | width | CV | argmax moved |
|---|---|---|---|---|
| NARROW (`alpha_rtg=0.5`, control) | [0.5672, 1.0000] | 1.76x | 0.186 | 0/12 |
| WIDE (`alpha_rtg=0.1`) | **[0.5672, 1.0000]** | **1.76x** | 0.186 | 0/12 |

**Manipulation check FAILED: the bands are identical.** Per the protocol, the
experiment is **VOID** and prediction 1 must not be interpreted. The guard was
written in advance precisely for this, and it fired.

## Why `alpha_rtg` is a no-op — and why that matters more than the intended result

`update_and_get_rtg_target` returns `max(batch_max, alpha_rtg * running_max)`.
Because rtg is **batch-max normalised**, `rtg[0] in [0,1]` and `running_max -> 1`
almost immediately. So the floor binds only when `batch_max < alpha_rtg`.

Measured over 790 real iterations (10 seeds):

    alpha_rtg = 0.5 : floor binds on  46.3% of iterations
    alpha_rtg = 0.1 : floor binds on   0.0% of iterations

At 0.1 the floor **never binds**, so `target == batch_max` and `alpha_rtg`
disappears from the computation entirely. Turning the knob down cannot widen the
band; it can only stop the knob from doing anything.

**The band is therefore set by the distribution of `batch_max`, not by
`alpha_rtg`.** `batch_max` is itself the max over ~200 batch-max-normalised
trajectories, so it concentrates near 1 by construction — which is exactly why
the realised support is a narrow [0.567, 1.0] and why H8 found so little to
respond to.

## Consequences

1. **My stated "top remaining lead" was wrong.** I recorded, in findings.md and
   research-state.yaml, that the inert-vs-starved confound could be separated by
   widening `alpha_rtg`. It cannot. That lead is retracted.
2. **The real lever is the normalisation, not the floor.** The batch-max
   normalisation introduced by the RTG-cap fix is what compresses the signal:
   it maps every batch's best trajectory to ~1 regardless of how good that batch
   actually was, discarding exactly the across-iteration variation a return
   conditioning signal needs. A genuine test of "starved vs inert" requires
   conditioning on an **un-normalised or differently-normalised** quantity
   (e.g. raw improvement, or normalisation by a fixed run-level scale rather
   than a per-batch max), plus a retrain.
3. **This is a self-inflicted design coupling worth recording**: the RTG-cap fix
   (which correctly removed the "2 distinct values across 200 trajectories"
   pathology) simultaneously created the narrow-support pathology. Fixing one
   RTG defect manufactured another.

## Status of the confound

Still open. "The network ignores RTG" vs "RTG carries too little variation"
remains unresolved, and H9 did not resolve it. What H9 *did* establish is that
the obvious cheap intervention does not work and why — which is a real, if
unglamorous, contribution to the next attempt.
