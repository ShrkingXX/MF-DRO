# RTG conditioning failure in Decision Transformers — literature

**Why we went looking.** We measured, repeatedly and across configurations, that
sweeping MF-DRO's RTG target 0.1x-10x leaves the proposed query's argmax
bit-stable. Score vectors *do* shift (corr 0.986-0.9999, never bit-identical),
so the pathway is connected but far too weak to change a decision. We had ruled
out a degenerate reward embedding but had no theory. It turns out this is a
known, named phenomenon.

---

## Return-Aligned Decision Transformer (RADT)
arXiv:2402.03923 — https://arxiv.org/pdf/2402.03923

**Diagnosis, verbatim in substance:** DT "struggles to align the actual return
with the target return due to the **under-allocation of attention scores to the
return-to-go tokens**."

This is a precise description of our measurement. RTG is one token competing for
attention against state and action tokens; the model learns to mostly ignore it,
so actual return decouples from target return.

**Their fix:** split the input into an RTG sequence and a state-action sequence,
and handle the RTG sequence with dedicated mechanisms so return information gets
sufficient computational weight — structural, not parametric.

**Their framing, which is the important part for us:** "return-to-go
conditioning requires *structural*, not just parametric, solutions." Tuning
(temperature, loss weights, LR) does not fix attention under-allocation.

---

## Decoupled Decision Transformer (DDT)
arXiv:2601.15953 — https://arxiv.org/html/2601.15953

**Mechanism — directly implementable for us.** Remove RTG from the transformer
input entirely; condition via Adaptive LayerNorm on the *last* RTG only.

    gamma(R), beta(R) = MLP(R)          # single Linear(1 -> hidden*2), no activation,
                                        # AdaLN-Zero init for training stability
    h_cond = gamma(R) * (h - mu(h)) / sqrt(var(h) + eps) + beta(R)

Input goes from `(R, o, a, R, o, ...)` (3k tokens) to `(o, a, ..., o)` (2k),
a 33% reduction, with quadratic-complexity savings.

**Evidence:** "significantly outperforms DT" on D4RL (hopper-medium 99.4 vs
68.3 normalized). Attention maps "align more closely with the ideal pattern,"
concentrating near the diagonal rather than dispersing over redundant RTG
tokens.

---

## Relevance to MF-DRO

Our architecture is the *worse* case for this failure: 4 tokens per step
`[rtg, btg, state, action]`, so **two** of four tokens are scalar conditioning
signals competing for attention, versus one of three in standard DT. If
under-allocation scales with how much a scalar token has to compete, we should
expect the effect to be at least as severe here — which is what we observe.

Note this is orthogonal to the target-leakage bug we fixed (7bcc3b8). That bug
made the heads read their own labels; this is about conditioning strength once
the leak is gone. Fixing the leak was necessary but would not by itself make
RTG causally effective.

**Implication for the fix scope in PROTOCOL.md:** "Decision Transformer
architecture / conditioning" is explicitly in scope, so an AdaLN-style
conditioning change is a legitimate within-frame fix, not a method swap.
