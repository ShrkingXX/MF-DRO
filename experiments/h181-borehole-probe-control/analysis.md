# h181 — the matched control. **P2. The 336× does not transfer, and h179's verdict is revised.**

**CONFIRMATORY** against the protocol committed (and corrected) before launch. 5/5 seeds.

## The registered statistic and gate

> statistic: mean over probed iterations of |x(rtg=0) − x(rtg=1)|, real inference
> state, pooled over seeds. Bands: **P1 < 0.051 | P2 [0.051, 0.152] | P3 > 0.152**

| arm | n | RTG responsiveness | per-seed | fidelity flips |
|---|---|---|---|---|
| h179 STDCOND (standardised) | 5 | 0.1010 | 0.096, 0.107, 0.128, 0.064, 0.110 | 23% |
| **h181 PROBECTL (control)** | 5 | **0.0788** | 0.083, 0.086, 0.098, 0.095, 0.033 | 18% |

**0.0788 → P2.** Standardisation raised RTG responsiveness by **1.28×**, with the
two arms' per-seed ranges overlapping heavily (0.064–0.128 vs 0.033–0.098).

## The confound h181 existed to remove was real

| | unstandardised RTG responsiveness |
|---|---|
| **Borehole** (h181) | **0.0788** |
| **Hartmann** (h177 / h178) | **0.0404** |

Borehole is **1.9×** more responsive than Hartmann *before any standardisation*. So
h179's 0.1010 was mostly the benchmark, not the intervention — exactly the
ambiguity that made h179 uninterpretable alone. (h177 and h178 return identical
per-seed values, an independent check that those probe arms are the same run.)

## On BTG — the channel the 336× was actually about

The registered gate was on RTG, but h178's headline was **BTG**: z-scoring restored
its embedding response **336×** at module level. In situ:

| | BTG responsiveness |
|---|---|
| Borehole unstandardised (h181) | **0.00685** |
| Borehole standardised (h179) | **0.02634** |
| Hartmann unstandardised (h177) | 0.00477 |
| **in-situ effect of standardisation** | **3.84×** |
| **fraction of the 336× that transfers** | **1.1%** |

**Confirmed, not retracted:** BTG is effectively inert at inference on *both*
benchmarks (0.005–0.007 unstandardised).

**Scoped down:** the **336× is a module-level number and stays one.** In the running
model standardisation buys 3.84× of a negligible base, leaving BTG at 0.026 —
against RTG's 0.101 and a seed-to-seed floor of ~0.82 on the emitted query. The
channel is still not meaningfully usable.

## h179's verdict is REVISED: R3 → uninformative

h179 was read as **P2 → R3**: "the channel is genuinely irrelevant either way; the
τ=0 account is strengthened and the 'defect' framing downgrades to a curiosity."

**That reading is withdrawn.** R3 requires that the channel was *made to work* and
still did not help. h181 shows it was not: the intervention moved RTG by 1.28× and
BTG to a level still an order of magnitude below RTG. h179 therefore tested a
change that barely changed its target, and its flat regret (16.66 vs 15.82) says
**nothing** about the counterfactual "what if the conditioning were usable".

That counterfactual remains **untested**. It is a genuine open item, not a settled
null — and it is the honest cost of h179 having varied two things at once.

## What this retracts, as named in the protocol

> R1: findings.md reports h178's z-scoring as restoring BTG responsiveness 336×
> with the scope "module-level upper bound". If the control shows the running model
> is equally responsive without standardisation, the in-situ half of that claim is
> unsupported.

Fired, in its milder form: the control is not *equally* responsive (3.84× is a real
increase), but only ~1% of the module-level effect reaches the running model, and
the absolute level stays negligible. The 336× must never be quoted without
"module-level".
