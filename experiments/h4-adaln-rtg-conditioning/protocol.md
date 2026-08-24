# Protocol — H4: RTG attention under-allocation

**Locked before running.** Results commit must be separate.

## Motivation (literature-grounded, not a guess)

We measured, across every configuration tried, that sweeping the RTG target
0.1x-10x leaves the proposed query's argmax **bit-stable**. Score vectors do
shift (corr 0.986-0.9999, never bit-identical), so the pathway is connected but
far too weak to change a decision. We ruled out a degenerate reward embedding.

This is a **named, published DT failure mode**, not a quirk of our code:

- **RADT** (arXiv:2402.03923): DT "struggles to align the actual return with the
  target return due to the **under-allocation of attention scores to the
  return-to-go tokens**", and argues the fix must be **structural, not
  parametric** — i.e. tuning temperature/loss weights cannot fix it.
- **DDT** (arXiv:2601.15953): remove RTG from the transformer input; condition
  via **AdaLN on the last RTG only**. Reports large D4RL gains (hopper-medium
  99.4 vs 68.3) and attention maps that stop dispersing over RTG tokens.

See `literature/rtg-attention-underallocation.md`.

**Why our case should be worse than standard DT.** Our layout is 4 tokens per
step, `[rtg, btg, state, action]` — **two of four** tokens are scalar
conditioning signals competing for attention, versus one of three in standard
DT. If under-allocation scales with competition, we should see it at least as
severely, which matches the observation.

This is orthogonal to the target-leakage bug (7bcc3b8). That bug let heads read
their own labels; this is about conditioning *strength* once the leak is gone.
Fixing the leak was necessary but would not make RTG causally effective.

## Hypothesis

**H4**: MF-DRO's RTG-insensitivity is attention under-allocation to the scalar
conditioning tokens. Replacing token-based conditioning with AdaLN-Zero
modulation will make RTG **causally effective** on the emitted decision.

## Change under test (in `PROTOCOL.md` fix scope: "DT architecture / conditioning")

New flag `rtg_conditioning`:
- `"token"` — current: 4 tokens `[rtg, btg, state, action]`. Default preserved.
- `"adaln"` — 2 tokens `[state, action]`; `rtg` and `btg` instead produce
  per-feature scale/shift applied to the state hidden state:

      gamma, beta = Linear(2 -> hidden*2)([rtg, btg])   # no activation, AdaLN-Zero init
      h_cond      = gamma * LayerNorm(h) + beta

  AdaLN-Zero init (gamma->1, beta->0 at start) so the run begins equivalent to
  no conditioning and the effect is learned, not imposed.

Consequent index changes (must be applied to BOTH paths or the train/inference
mismatch returns): readout `h[:, 2::4]` -> `h[:, 0::2]`, position embedding
`repeat_interleave(4)` -> `(2)`, causal mask 4x4 -> 2x2 in `propose_mf`.

## Primary measurement — MECHANISM, not the downstream metric

**Locked prediction (confirmatory):** under `adaln`, sweeping the RTG target
over {0.1, 0.5, 1, 2, 5, 10}x with everything else fixed changes the proposed
argmax on **>30% of sweeps**, versus a measured **0%** under `token`.

Report, for both arms, over >=10 independently resampled candidate pools:
- fraction of sweeps where argmax moves at all
- number of distinct argmaxes per sweep
- pairwise score-vector correlation across multipliers
- argmax(score) vs argmax(mu_H) agreement (is it still a fixed acquisition?)

This is a direct causal test of the conditioning pathway and needs no full BO
run, so it is cheap and fast to falsify.

## Secondary — only if the primary passes

Then, and only then, run the frozen evaluation (Hartmann 6D, seeds 42-51,
matched cost 200, identical init) with `rtg_conditioning="adaln"` and compare
against the `token` arm from `h1-leak-fix-validation`. One variable.

## What each outcome means

- Argmax moves AND regret improves -> under-allocation was a real driver;
  a structural conditioning fix is the within-frame answer.
- Argmax moves but regret does NOT -> RTG conditioning now works and simply
  is not worth much on this problem. Publishable negative result: it would say
  the DT's return-conditioning premise adds little for BO.
- Argmax still does not move -> H4 refuted; the insensitivity is NOT attention
  under-allocation, and the cause is downstream (score head / candidate
  features). Rules out the leading literature explanation.

## Compute

Primary probe is single-process and cheap. Do not launch while the
h1-leak-fix-validation grid still occupies the pool.
