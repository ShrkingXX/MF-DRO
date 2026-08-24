# H20 — removing the linear bottleneck does NOT restore conditioning

| arm | affine residual | state swap moves argmax | in-band RTG sweep |
|---|---|---|---|
| L `use_linear_score_head=True` (default) | **0.000000** | 0/12 = 0.0% | 0/12 = 0.0% |
| **M `use_linear_score_head=False` (MLP `[h;cf]`)** | **0.086347** | **0/12 = 0.0%** | 1/12 = 8.3% |

**MANIPULATION PASS** — arm M's scores are genuinely not affine in `cf`
(residual 0.0863 > 0.05), against arm L's exactly 0.000000. The bottleneck was
really removed; this is not a null from a no-op.

**PRED 1 FAIL** (0.0%, needed >30%). **PRED 2 NULL fires.**

## What this establishes

The conditioning failure is **architecture-independent within this family**. A
score head with no factorisation through an 11-dimensional coefficient vector —
`h` entering jointly and non-linearly with every candidate — still cannot make
the decision depend on the state.

Combined with the attenuation measurement, this sharpens the mechanism. The loss
is ~3× at the encoder and ~3× at the head, but **the encoder half alone is
sufficient**: `h`'s own relative spread across real states is just **0.0745**,
and an unbottlenecked head reading that `h` still moves the argmax 0/12. Fixing
the head therefore *cannot* help, and H20 is the direct test of that.

## A dead flag, found by an assert

The first run of this probe **failed its own assertion**: `use_linear_score_head`
was never forwarded into `dt_cfg`, the `SimpleNamespace` handed to
`DecisionTransformer`, so `getattr(config, 'use_linear_score_head', True)`
always defaulted to `True` and the `False` branch had been **unreachable since it
was written**. `findings.md` listed it among the available ablations; that was
wrong and is corrected.

Default behaviour is unchanged, so no previously-recorded result moves. An audit
of every other config attribute the DT reads (`cand_feature_dim`,
`rtg_conditioning`, `score_temp`) confirms those are forwarded — this was the
only dead one.

The lesson from the H5 audit paid off immediately: **assert inside the probe that
the manipulation took effect.** Without that assert this experiment would have
reported a confident null from an arm identical to the control.

## Status of the programme

Eleven pre-registered interventions on the conditioning pathway — conditioning
mechanism (H4), features (H5), context length (H11), RTG schema (H8/H9/H10),
reward (H12–H17), dynamics (H18/H19), and now score-head architecture (H20) —
all null or void. The one intervention that moved *regret* (H17, joint MES,
−0.104 mean) did so by changing the training signal, not the conditioning
pathway, and still failed the frozen success test.

The empirical programme is complete.
