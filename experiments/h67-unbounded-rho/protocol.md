# H67 — reparameterise rho with unbounded support (the repair h63 identified)

**CANCELLED BEFORE LAUNCH — PREMISE REFUTED BY PRE-FLIGHT.** Briefly marked
LAUNCHED earlier the same day; that amendment is superseded and no h67 arm was
ever run. The regression gate PASSED (bit-for-bit, 339 bytes) and the
`ko_rho_link` implementation is retained and committed, but the experiment it
was built for tests a constraint that is not active.

**Why cancelled.** The protocol assumes the sigmoid ceiling BINDS -- that rho
wants to exceed 1 on Borehole and is capped. Measured on real benchmark data at
h57's initial-design sizes (`src/analysis/rho_capacity.py`, seeds 44/46/48):

| benchmark | true slope | fitted rho (sigmoid) | ceiling binds? |
|---|---|---|---|
| Borehole 8D | 1.2566 | **0.8436** | **no** -- 0.84 is nowhere near 1.0 |
| Hartmann 6D | 0.9792 | 0.7797 | no |
| Currin 2D | 1.0104 | 0.8435 | no |

rho never approaches the ceiling on ANY benchmark, so removing the ceiling
cannot be what releases the shortfall. Predictions 1 and 2 are both untestable
as written: softplus would be inert on Borehole for the same reason it is inert
on Hartmann, and the "helps where it binds, inert where it does not" contrast --
the whole discriminator -- has no binding case to contrast against.

**What actually limits rho** (`src/analysis/rho_budget.py`): `fit()` takes ONE
Adam step on rho per round, 3 rounds per call, lr=0.1 from rho_init=0.8. Given
90 accumulated steps on synthetic data with a known slope of 1.2566, sigmoid
reaches 0.8499 and softplus 0.8718 -- and rho **oscillates** rather than climbing
(0.8436 -> 0.8132 -> 0.8454 -> 0.8477 -> 0.8239). rho and delta(x) are jointly
unidentifiable and delta, a flexible GP, absorbs the shortfall first, which kills
the gradient on rho. The link's range is irrelevant because rho is never pushed
toward it.

**This also revises h63's interpretation.** RHOTRUE's Borehole gain (+1.9%) was
NOT "removing the ceiling" -- it supplied a rho the fitter could never have
found under any link. And RHOTRUE was catastrophic on Hartmann (-16.6%) where
the true slope 0.9792 IS representable, which says the under-fitted rho ~= 0.78
is BETTER there than the true slope. So rho is not functioning as an estimate of
the LF->HF slope at all; it behaves as a regularisation term trading LF transfer
against delta flexibility. "Estimate rho better" is therefore not obviously a
repair either, and h63's own control result argues against it.

**Cost avoided:** 6 jobs. **Cost spent:** three short pre-flight fits.

## Why

h63 supported the user's KO-misspecification hypothesis via its pre-registered
contrast:

| benchmark | true slope | representable? | BASE | RHOTRUE | gain |
|---|---|---|---|---|---|
| Borehole 8D | **1.2566** | **NO** | 23.7% | 21.8% | **+1.9%** |
| Hartmann 6D | 0.9792 | yes (control) | 14.7% | 31.3% | **-16.6%** |

`rho = sigmoid(log_rho)` confines rho to (0,1). Borehole's true LF->HF slope is
1.2566 with OLS residual sd **0.0001** — the relation is essentially exact and
the model **cannot express it**. The shortfall `0.2566 * f_L(x)` is forced into
`delta(x)`, a GP with a zero-centred prior.

But **pinning rho is not the fix**: it is catastrophic (-16.6%) where the slope
is already representable, so it cannot ship as a default. What h63 identifies is
a **modelling defect**, not a tuning knob.

## The repair

Replace the sigmoid link with one that has unbounded positive support, so rho is
**fitted** rather than capped:

    current:  rho = sigmoid(log_rho)            in (0, 1)
    proposed: rho = softplus(raw_rho)           in (0, inf)

This lets Borehole's fit reach ~1.26 while leaving Hartmann free to settle near
its fitted ~0.75-0.98. One change, opposite predictions per benchmark.

## Design

Borehole 8D and Hartmann 6D, seeds 44/46/48, cost budget 200. `ko_rho_link`
config key: `"sigmoid"` (default, unchanged) or `"softplus"`. BASE reuses h57.
6 jobs.

**A regression gate must pass before any arm runs**: with `ko_rho_link="sigmoid"`
a 3-iteration Currin run must reproduce the pre-edit result bit-for-bit, as the
h61 and h63 edits both did.

## Locked predictions

1. **PRIMARY**: softplus beats BASE on **Borehole** >= 2/3 — the ceiling binds
   there, so removing it should help.
2. **CONTROL**: softplus is **within noise of BASE on Hartmann** — the ceiling
   does not bind (fitted 0.7478 is far from 1.0), so removing it should be
   approximately inert. **This is the prediction that distinguishes a repair
   from a knob.** RHOTRUE failed exactly here (-16.6%); if softplus also craters
   Hartmann, it is another knob and not a fix.
3. **NULL**: no movement on Borehole. Then the (0,1) ceiling was not what cost
   the regret, and h63's contrast is explained by one of its four confounds
   after all — most plausibly the fidelity shift.
4. **HARMFUL**: softplus worse on both. Unbounded rho may destabilise the MLL
   fit, since rho and the delta-GP's outputscale become partly unidentifiable
   once rho can grow without limit.

Prediction 2 is the primary discriminator, not prediction 1. A method that helps
where the defect binds and is inert where it does not is a repair; one that helps
in one place and harms in another is a knob, and this project already has one of
those.

## What this cannot settle

n = 3 per cell. Whether softplus is the *right* unbounded link (versus an
unconstrained scalar, or a per-member prior) is untested — this asks only whether
removing the ceiling helps.
