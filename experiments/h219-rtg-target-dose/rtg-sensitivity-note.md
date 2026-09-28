# Is RTG actually insensitive? Measured — and the answer is no

EXPLORATORY. Re-analysis of saved H168 probe records (h207 arms). No new runs.

## Direct causal evidence that the RTG TARGET matters at inference

Two h212 results are one-factor tests of the inference RTG value, holding the label and
the data fixed:

| change to the target alone | effect |
|---|---|
| un-pin it from the constant 0.5 (P-FIX) | **−5.27 on 5/5** |
| move it from ≈batch-max to the 90th percentile (P-NEUTRAL) | **+3.52 on 0/5** |

A model that ignored RTG could not produce either number. **"RTG is inert" is wrong as a
blanket claim**, and I have been repeating it.

## A correction to my own shorthand

I have been quoting "RTG sensitivity 1.1–1.4x" as if it meant near-inert. It does not:
that was a **ratio between arms** (MIXR/NIR, MIXO/NIR), not an absolute. The absolute
figures, from the same probe records, are:

| arm | d(x)/dRTG | d(x)/dSTATE | state/RTG |
|---|---|---|---|
| NIR-B | 0.0369 | 0.0389 | 1.06 |
| MIXR-B | 0.0410 | 0.0117 | 0.29 |
| MIXO-B | 0.0515 | 0.0096 | 0.19 |

Against a real-query dispersion of ~0.24 (CTRL-K1 Borehole 0.237), **RTG moves the emitted
query by roughly 15–20% of its total spread**. That is a real channel, not a dead one.

## The user's hypothesis (b) — "state is too dominant" — is NOT supported, but NOT refuted either

In MIXR and MIXO, RTG moves the output **more** than state does (ratios 0.29, 0.19), which
is the opposite of state dominance. **But that comparison is not fair and I will not claim
it.** The probe's state axis compares the real state against τ=0 states *from the same
iteration*, and h207's own STATE-DIAG recorded `uniq_tau0_states=3` for a 120-trajectory
batch — those states are nearly degenerate. So the measured d(x)/dSTATE is "sensitivity to
states that barely differ" and **understates** the true state sensitivity across a run.

A fair test needs a state axis spanning states from *different* iterations. That is a probe
change, registered as a diagnostic in h219 rather than asserted here.

## The user's hypothesis (a) — OOD from a train/inference mismatch — is partly refuted

The pinned-0.5 target was literally out of distribution (above everything the model trained
on) and cost 5.27. That supports (a).

But the other direction cuts against it: moving the target *inward* — from ≈batch-max to
the 90th percentile, i.e. **more** in-distribution — cost **+3.52**. If tail-conditioning
were the problem, moving inward should have helped.

So the picture is non-monotone: asking for ≈the batch best is good; asking *above*
everything is bad; asking *below* the best is also bad. Only a dose over the target
percentile maps that, which is what h219 runs.
