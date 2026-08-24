# Audit of H5 — the original probe was invalid; the conclusion survives re-measurement

**Status: EXPLORATORY** (unplanned self-audit, triggered by a `STATE-DIAG` line
reading `uniq_tau0_states=10` for 200 trajectories).

## The defect

H5's h-sensitivity test drew its comparison state as

    st_other = batch[(p + 7) % len(batch)]["states"][0]     # p = 0..11

so it used indices **7..18**. The batch is built
`for ko in ko_ensemble: for _ in range(rollouts_per_model)` with
`rollouts_per_model=20`, so **indices 0..19 all come from model 0 and share a
bit-identical τ=0 state**. Measured directly:

| check | result |
|---|---|
| H5's actual comparisons (idx 7..18) identical to `batch[0]` | **12/12** |
| across model blocks (idx 20, 40, …) identical to `batch[0]` | **0/9** |
| unique τ=0 states in a 200-trajectory batch | **10** |

**H5 swapped a state for itself.** Its "argmax unchanged 12/12" is exactly what
feeding the identical state must produce, and carried no information about
whether the score head reads `h`.

## The corrected measurement

Comparison states drawn from *different* ensemble-model blocks (verified
distinct, 0/9 identical):

    argmax changed when h comes from a genuinely different state : 0/12 = 0.0%
    mean score-vector correlation across states                  : 1.000000

**The conclusion survives.** The score head really is insensitive to `h` — a
correlation of `1.000000` between score vectors computed from different states
is stronger evidence than the original probe could ever have given.

## What must be said carefully

The claim "the score head barely reads `h`" is **true**, but until now it rested
on an **invalid probe**. Everything I built on it over the last two ticks — the
H19 conclusion, the intervention-ladder figure, the `to_human` report's lead —
was correct by luck, not by evidence. It is now correctly evidenced.

Also note: H8/H19's RTG results are **unaffected** by this defect. Those probes
varied the RTG target, not the state, so they never used the broken swap.

## A second hypothesis, refuted in passing

I suspected the insensitivity was **feature-scale domination**: one candidate
feature so much larger in spread that it fixes the ranking regardless of the
state-dependent coefficients. Measured contribution `|w_f| · sd_k(cf[:,f])`:

| feature | share of ranking spread |
|---|---|
| `mu_H` | 31.8% |
| `mu_L` | 22.8% |
| `x[1]` | 13.8% |
| `dist_inc` | 8.0% |
| others | ≤ 6.7% each |

No domination — `mu_H`'s across-candidate sd is **1.0×** the median feature's.
And `argmax(score) = 22` vs `argmax(mu_H) = 152`, so the old finding that the
score tracks `mu_H` on 67–75% of pools **does not reproduce** post-fix.

## Lesson

When a probe compares "two different X", assert that they are different **inside
the probe**. H5 would have failed a one-line assertion. This is the third
instrument defect this phase (H13's dead-signal metric, H19's diversity
signature, now H5's state swap) — all three found by checking the instrument
against the data rather than trusting it.
