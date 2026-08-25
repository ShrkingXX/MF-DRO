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
