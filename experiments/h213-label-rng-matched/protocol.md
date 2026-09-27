# h213 — The label comparison, finally as a one-factor test

Registered BEFORE any h213 run. CONFIRMATORY. One new flag (`rtg_rng_parity`, default
False = bit-identical; identity gate PASS at 122.29066752728207).

## Why this exists

`mes_entropy` computes `b_τ` at every rollout step via `_rollout_gumbel_b` →
`thompson_sample_y_star` → `posterior.rsample()`, which draws from the **global torch RNG**.
`terminal_improvement` makes no such draw. At 60–120 rollouts × 8 steps per BO iteration
the two labels run on different random streams. **Measured** (same seed, same initial
design, Hartmann): the first REAL query already differs on 3/3 seeds checked —
CTRL-K1 1.992218 vs K1-TI 1.915485 (seed 42), 1.191680 vs 0.951391 (43), 1.116973 vs
1.368100 (44).

So h211's P-LABEL (+5.80 on Hartmann, 0/5) compared arms differing in **three** things:
the label values, the inference target schema, and the entire training set. It is a real
configuration effect but an unattributed one. Same class of defect as h198's config
override. This experiment removes the third.

## The fix

`rtg_rng_parity=True` draws `b_τ` under **every** label and discards it when unused —
per step, plus the post-loop `b_T` draw. Both labels then consume identical RNG, and the
label becomes the only thing that varies. The flag changes no label's semantics: under
`mes_entropy` the code path is unchanged (its draws are used), so the parity arm and the
existing `mes_entropy` arms are the same procedure.

## Arms — 10 runs, one wave

| arm | bench | seeds | config |
|---|---|---|---|
| **TI-PAR-H** | Hartmann | 42–46 | CTRL-K1 + `terminal_improvement` + `rtg_rng_parity=True` |
| **TI-PAR-B** | Borehole | 42–46 | same on Borehole |

Reference, **same RNG stream by construction**: CTRL-K1 (`mes_entropy`) **5.93** Hartmann /
**11.59** Borehole. Unmatched reference: K1-TI-H **11.73** (h211).

Borehole is included because the label was measured as *neutral* there (h207 P-NIR, +1.23,
inside band) — also an unmatched comparison. If the label is genuinely neutral on Borehole
and harmful on Hartmann, that is a benchmark-dependent label effect; if the matched
comparison moves on both, the earlier readings were largely RNG.

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-LABEL-H:** TI-PAR-H − CTRL-K1-H > +1.26 on ≥ 4/5. The label really is harmful on
  Hartmann and h211's +5.80 mostly survives RNG matching. **Lean: it does NOT — I expect
  |Δ| ≤ 1.26 or a much smaller gap**, because the user's objection is well-founded: the
  measured RTG sensitivity is 1.1–1.4×, so a channel that barely moves the decision head is
  a weak candidate for a 5.8-point effect, and RNG divergence is now a known live
  alternative. Registering the prediction against my own lean so the lean can be wrong.
- **P-LABEL-B:** |TI-PAR-B − CTRL-K1-B| ≤ 1.26. Reproduces h207's neutrality under
  matching.
- **Ordering, so it can be wrong:** TI-PAR-H ≈ CTRL-K1-H and TI-PAR-B ≈ CTRL-K1-B, i.e.
  the label is inert once RNG is matched.

## What each outcome RETRACTS

- **Label inert under matching** → **h211's P-LABEL is RETRACTED**; the +5.80 was RNG
  divergence, not the label. Then h210's "the Hartmann gap is the K=8 configuration" —
  which h211 had already retracted — is **reinstated as the live reading**, and h212's
  P-FIX becomes a test of a schema whose defect costs nothing.
- **Label harmful under matching** → h211's P-LABEL stands with the confound removed, and
  the cause narrows to the label values themselves (their distribution entering the
  saturating `Linear(1→H)+LayerNorm`, h177/h178) or the target schema — which h212
  separates.
- **Either way**, every earlier cross-label comparison (h207 P-NIR, h210 CTRL-K1-H vs
  NIR-H, h211 P-LABEL) is re-scored against this arm rather than quoted as-is.

## Relationship to h212 (running)

h212 compares TI-PCT vs K1-TI — both `terminal_improvement`, so RNG-matched to each other
already. It tests the **target schema** cleanly and is unaffected by this defect. h213 tests
the **label**. Together they decompose h211's unattributed +5.80 into schema, label, and RNG.

## Stage 0 (smoke)

(a) both arms build and run for BUDGET=30; (b) `rtg_rng_parity` is True on the constructed
config; (c) **the parity check itself**: a `terminal_improvement` run with parity ON must
produce the SAME first real query as the `mes_entropy` control on the same seed — that is
the whole point, and a mismatch means the streams still differ. This is the gate.

## Compute

10 runs, cap 15. Runs alongside h212's 15 only if slots allow; otherwise queued. Hartmann
first. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
