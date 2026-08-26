# H63 — is MF-DRO's Borehole failure KO misspecification? (user hypothesis)

**CONFIRMATORY. Protocol committed before any run.**

## The hypothesis and the measurement behind it

The KO surrogate assumes `f_H(x) = rho * f_L(x) + delta(x)` with
`rho = sigmoid(log_rho)`, i.e. **rho is confined to (0,1)**
(`src/models/ko_gp.py:154`). The true relationship, by OLS over 8192 Sobol
points:

| benchmark | corr | OLS slope | representable as rho? | residual sd |
|---|---|---|---|---|
| Hartmann 6D | 0.925 | 0.9792 | yes | 0.1460 |
| Currin 2D | 0.997 | 1.0104 | no (marginal) | 0.1928 |
| **Borehole 8D** | **1.000** | **1.2566** | **NO** | **0.0001** |

On Borehole the LF/HF relationship is **essentially exact** — residual sd 0.0001
— and purely affine with slope 1.2566. The model cannot represent it. The
shortfall `0.2566 * f_L(x)`, which spans ~60 units over f_L's range, must be
absorbed by `delta(x)`: a GP with a **zero-centred prior** asked to carry a
systematic multiplicative term.

And Borehole is where MF-DRO fails worst (23.7% vs MI-Greedy's 8.3%), while
Hartmann — whose slope *is* representable — is where it does relatively best.

### Correction recorded before running

I first predicted rho would **saturate near 1**. It does not. Fitting a KO on
each benchmark's h57 initial design gives:

| benchmark | fitted rho | true slope |
|---|---|---|
| Hartmann 6D | 0.7478 | 0.9792 |
| Borehole 8D | 0.8436 | 1.2566 |

So the fit lands *short of even the representable range*, not pinned against it.
The saturation story is withdrawn; the range-violation story stands, and the
undershoot is larger on Borehole (-33%) than Hartmann (-24%).

## Design

`KennedyOHaganGP` already exposes `rho_fixed` (`ko_gp.py:167`), and it bypasses
the sigmoid (`ko_gp.py:245-246`), so a value above 1 is settable. One variable.

| arm | rho | benchmarks |
|---|---|---|
| **BASE** | fitted (free) — **reuses h57's cells** | Borehole, Hartmann |
| **RHOTRUE** | `rho_fixed` = OLS slope (1.2566 / 0.9792) | Borehole, Hartmann |

6 new jobs (2 benchmarks x 3 seeds), seeds 44/46/48, cost budget 200.

**Hartmann is the control, and it is what makes this a test rather than a
tuning exercise.** Its true slope is inside (0,1), so fixing rho there corrects
only the fit's undershoot, not a representability failure. Borehole's fix
corrects both.

## Locked predictions

1. **PRIMARY**: RHOTRUE beats BASE on Borehole on >= 2/3 paired seeds.
2. **THE DISCRIMINATING CONTRAST**: the Borehole improvement is **larger** than
   the Hartmann improvement. If misspecification-by-range-violation is the
   mechanism, correcting it should help most where the violation is largest.
3. **NULL**: neither benchmark moves. Then rho misspecification is excluded and
   the surrogate hypothesis narrows to the discrepancy GP or the kernel, not rho.
4. **EQUAL-IMPROVEMENT**: both benchmarks improve by similar amounts. That would
   indicate the gain is from correcting the *undershoot* (present in both), not
   the *range violation* (Borehole only) — a different and weaker claim than the
   hypothesis under test, and it must be reported as such rather than as
   confirmation.

Prediction 4 is the one I expect to be under-weighted: the fit undershoots on
BOTH benchmarks, so a naive "RHOTRUE helps" result does not by itself establish
the range-violation mechanism.

## What this cannot settle

n = 3 per cell. `rho_fixed` also freezes rho against per-iteration refitting,
so RHOTRUE differs from BASE in two ways at once — value *and* adaptivity. A
third arm fixing rho at the *fitted* value (0.8436) would separate those, and is
the follow-up if RHOTRUE wins.

---

## CONFOUND, recorded before any result exists (0 files on disk)

`rho_fixed` is passed through `_ko_kwargs`, so it applies to **every ensemble
member**. But the ensemble deliberately diversifies rho:

    torch.manual_seed(config.seed + 2000)
    _rho_inits = (0.3 + 0.65 * torch.rand(config.M)).tolist()   # ~U(0.3, 0.95)

with the stated purpose of producing "dense, diverse rollout training data for
the DT" — M members that are not M copies of one model. **RHOTRUE collapses that
diversity**: all 10 members get rho = 1.2566 (or 0.9792) exactly.

So RHOTRUE differs from BASE in **three** ways, not one:

1. rho's **value** (fitted ~0.84 -> pinned 1.2566 on Borehole)
2. rho's **adaptivity** (refit each iteration -> frozen)
3. the ensemble's **rho diversity** (U(0.3,0.95) across members -> identical)

Difference 3 is the one that could produce a *negative* result for reasons
unrelated to the hypothesis: h45 established this architecture is sensitive to
demonstration diversity, and collapsing rho across the ensemble reduces exactly
that.

**Consequence for interpretation, fixed now rather than after seeing numbers:**

- **RHOTRUE wins on Borehole, and by more than on Hartmann** -> supports the
  range-violation hypothesis, since differences 2 and 3 are identical across
  both benchmarks while only Borehole's rho is unrepresentable.
- **RHOTRUE loses or is flat** -> does **not** refute the hypothesis, because
  differences 2 and 3 could be masking it. The correct follow-up is an arm
  pinning rho at each member's own *fitted* value, preserving diversity, which
  isolates value from diversity.

I am recording this because the same class of error — attributing a gap between
two configurations to the dimension named in the arm, when several things moved
at once — has already produced one retraction in this project (the
multi-fidelity attribution, lesson 22).

---

## FOURTH difference, observed live before any result was written

RHOTRUE does not merely change the surrogate's fit — it **inverts the fidelity
policy**. Live checkpoints against BASE's finals on Borehole:

| arm | seed | HF | LF | HF% |
|---|---|---|---|---|
| BASE | 44 | 100 | 1 | **99%** |
| BASE | 46 | 100 | 1 | **99%** |
| BASE | 48 | 99 | 3 | 97% |
| **RHOTRUE** | 44 | 6 | 128 | **4%** |
| **RHOTRUE** | 46 | 23 | 111 | **17%** |

This is mechanistically expected, not a bug: `mu_H = rho*mu_L + mu_delta` and
`var_H = rho^2*var_L + var_delta`, so a larger rho makes each LF observation more
informative about f_H, and the cost-normalised MES teacher shifts to the cheap
fidelity. Pinning rho at 1.2566 instead of the fitted ~0.84 is a **50%** increase
in how much LF is believed to say about HF.

So RHOTRUE differs from BASE in **four** ways: rho's value, rho's adaptivity, the
ensemble's rho diversity, **and the realised fidelity mix (99% HF -> 4-17% HF)**.

### This weakens the experiment, and the weakening is worth stating plainly

Borehole was chosen as the test bed *because fidelity is inert there* — BASE
runs 99-100% HF, so no fidelity explanation could apply. **RHOTRUE breaks that
property.** It is no longer a fidelity-inert comparison, which was the whole
reason for choosing this benchmark.

Consequences, fixed before results:

- A RHOTRUE **win** is now ambiguous between "correcting rho misspecification"
  and "shifting budget to a cheap, perfectly-correlated fidelity" — and on
  Borehole corr(f_LF, f_HF) = 1.000, so the second is a live alternative that
  h58's floor result already showed can matter.
- A RHOTRUE **loss** is similarly ambiguous: it could be the fidelity shift
  hurting rather than rho being right.
- The **Hartmann control still discriminates**, because differences 2-4 apply to
  both benchmarks while only Borehole's rho is unrepresentable. A *larger*
  Borehole gain remains the diagnostic signal; a bare "RHOTRUE wins" does not.

The clean follow-up, if RHOTRUE moves anything: pin rho AND force the BASE
fidelity mix (`minimum_hf_fraction` at inference, h58's mechanism), isolating
the surrogate change from the policy change.

---

## VALIDITY CHECK (not a result): rho_fixed applied on Hartmann

RHOTRUE's Hartmann seed 44 returned **0.7531 — identical to BASE's 0.7531**,
which would be the signature of the config key silently not applying. It did
apply:

| run | regret | nq | HF | LF | improv |
|---|---|---|---|---|---|
| BASE | 0.7531 | 31 | 25 | 6 | 0 |
| RHOTRUE | 0.7531 | **95** | **15** | **80** | 0 |

95 queries at 16% HF against 31 at 81% — different runs entirely. The identical
regret is because **neither ever beat the initial design's incumbent**
(`improv = 0` in both), so both report the init's best value. A tie by that
mechanism, not by the treatment failing to take.

Note this also reproduces on Hartmann the fidelity inversion recorded as the
fourth confound on Borehole: pinning rho at 0.9792 (from a fitted ~0.75) moved
the mix 81% HF -> 16% HF. So the confound is **not** Borehole-specific, which
strengthens the case that a RHOTRUE gain anywhere is ambiguous between the
surrogate correction and the budget shift.

Both h63 arms remain **incomplete and withheld** (Borehole 3/3, Hartmann 1/3),
and the protocol's discriminating signal requires the full contrast.
