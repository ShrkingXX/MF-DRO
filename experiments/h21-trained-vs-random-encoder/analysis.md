# H21 — the encoder's contraction is architectural; the head's is LEARNED

Measured on the 10 distinct ensemble-member τ=0 states (identical seed, identical
states, identical pools; the only difference is whether `_train_dt` ran).

| | relative spread `s` | `h` | `w` | s→h | h→w | argmax moved | min cos(w) |
|---|---|---|---|---|---|---|---|
| **T** trained | 0.013993 | 0.005455 | 0.001826 | **0.3898×** | **0.3348×** | 0/12 | 0.99999224 |
| **R** random init | 0.013993 | 0.006438 | 0.010326 | **0.4601×** | **1.6039×** | 0/12 | 0.99978395 |

**PRED 1 FAIL** — random's s→h is 0.4601× versus trained's 0.3898×, a ratio of
only 1.18×, far short of the pre-registered 1.5×.
**PRED 2 FAIL** — random still moves the argmax 0/12.
**PRED 3 NULL fires**, but with a nuance the protocol did not anticipate.

## The two stages behave completely differently

- **Encoder (s→h): architectural.** A randomly initialised transformer contracts
  state variation almost as hard as a trained one (0.4601× vs 0.3898×). Training
  is not what closes this stage; the shape of the network on these inputs is.
- **Head (h→w): learned, and dramatic.** A random head *amplifies* variation
  (**1.6039×**); the trained head *contracts* it (**0.3348×**). That is a
  **4.8× swing** attributable entirely to training. Training actively collapses
  the head.

The paper currently says the loss is "distributed, ~3× at the encoder and ~3×
again at the head, neither stage alone the culprit." That remains true of the
*trained* network but is now too coarse: one stage is a property of the
architecture and the other is a property of the fitted solution. They have
different remedies and should not be described as one phenomenon.

## Why the decision still does not move

Even at random initialisation, with **5.7× more coefficient variation**
(0.010326 vs 0.001826), the argmax moves 0/12 and the coefficient vectors still
sit at cosine 0.99978 (~1.2° apart). So the invariance is robust to a 5.7×
increase in the very quantity that drives it. That is a useful measure of how
far the system is from the threshold at which conditioning would begin to
matter: not marginally short, but far short.

## Interpretation guard (fixed in advance, and it binds)

Arm R is a **channel-capacity probe only**. An untrained network's choices are
meaningless for performance, and nothing here suggests training should be
skipped or that random initialisation is preferable. The result is about whether
the channel is open, not about which policy is better.

## Note on comparability

The state spread here (0.013993) is over **ensemble members at a fixed
iteration**; the earlier localisation figure (0.2155) is over **real iterations**
of an actual run — about 15× more varied, as expected. The two attenuation
estimates (0.346×/0.294× real-iteration; 0.390×/0.335× ensemble-member) are
consistent in magnitude but are measured over different state sets and must not
be conflated.
