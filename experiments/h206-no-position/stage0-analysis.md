# h206 Stage 0 — PASS

Run before any arm, per protocol.md. Two short Borehole seed-42 runs (BUDGET=70),
one ablated and one control, plus the identity gate run separately.

```
  ctrl   seq_len=  8  labelled idx=  8  grad is None=False  max|dW|=1.672e-02
  nopos  seq_len=  8  labelled idx=  8  grad is None=True   max|dW|=0.000e+00
```

| SC | criterion | result |
|---|---|---|
| **SC1** | position table frozen under ablation, trained under CTRL | `max\|dW\|` 0.000e+00 vs 1.672e-02; grad `None` vs not — **PASS** |
| **SC2** | identity gate with flag off | 122.29066752728207, bit-identical — **PASS** |
| **SC3** | context still flows under the ablation | degenerate-history delta 4.632e-02 > 0 — **PASS** |
| **SC4** | CTRL is the arange-K8 arm h205 measured | seq_len 8, 8 labelled rows — **PASS** |

SC1 is the one that matters most: a silent no-op would read as P1, which is the
outcome I said I lean toward. `max|dW| == 0.000e+00` exactly, with `weight.grad`
never allocated, proves the table is untouched rather than merely unhelpful.

## A side-prediction I got wrong

I wrote into the SC3 output that the *reversed-history* delta would be ~0 under
the ablation, on the reasoning that order would then be carried only by the causal
mask. Measured:

| | reversed history | degenerate history |
|---|---|---|
| ctrl | 3.593e-03 | 1.183e-02 |
| nopos | **3.144e-03** | **4.632e-02** |

Not ~0 — essentially the same as CTRL's. The error is in the prediction, not the
reasoning: the causal mask **is** an order signal (token *i* attends to tokens ≤ *i*,
so permuting history changes what every token sees), so "carried only by the causal
mask" never implied "carried by nothing". The gate criterion did not depend on this,
so nothing about the PASS changes.

## EXPLORATORY, n=1 probe, not a finding

The ablated model responds to history CONTENT about **4x more strongly** than CTRL
(4.632e-02 vs 1.183e-02) at the one state probed. If that survives on real runs it
would suggest the positional embedding was partly *crowding out* the context rather
than complementing it — which would favour P3 over my registered lean of P1. One
probe, one seed, one state; recorded so it is on the record before the arms land,
not offered as evidence.

## Stage 1

Launched on the gate: 10/10 workers (N and P x seeds 42-46), 0 tracebacks,
10 x 1 thread <= 15.
