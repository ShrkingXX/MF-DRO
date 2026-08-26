# H70b — Does the KO-style GP construction really cost 6.41 points on Hartmann?

**CONFIRMATORY.** Locked before any h70b number exists. Bars name magnitudes,
after h68 and h65 both passed direction-only bars while the claim failed.

## Why

h70 turned up an **unpredicted** result: swapping `_build_ko_style_gp` (LogNormal
lengthscale prior + geometric-mean initialisation) for the plain
`mf_baselines._build_gp` moved Hartmann from 19.89% to **13.48%** — 2 wins, 1 tie
at n=3 — while being **exactly neutral** on Borehole (0.00 points).

That builder is used by every DRO and MES arm in this project. If it really costs
~6 points on Hartmann, it is a larger effect than most things measured here, and
it would be a defect in the shared harness rather than in any method. But it was
exploratory, at n=3, from a run I did not design to test it — exactly the shape
this project has retracted before.

These runs cost **under a second each**, so there is no excuse for n=3.

## Design

SF-EI with each GP builder, Hartmann 6D and Borehole 8D, **seeds 42-51 (n=10)**.
Everything else identical, including the 200-candidate pool. 40 jobs, seconds
total. Seeds 44/46/48 will reproduce h70 exactly and serve as a built-in control.

## Locked predictions

1. **PRIMARY.** On Hartmann at n=10, ALTGP beats KO-style by **>= 3.0 points** of
   relative regret **and** wins **>= 7/10** seeds. Both bars must hold. h70's n=3
   estimate was 6.41 points; requiring half that plus a clear win count guards
   against the h64 failure, where a large mean came from one seed.
2. **SECONDARY.** On Borehole at n=10, |difference| **< 1.0 point** — the exact
   neutrality h70 found at n=3 persists.
3. **NULL.** Hartmann difference **< 3.0 points** or wins **<= 6/10**. Then h70's
   6.41 was an n=3 artifact and the finding is withdrawn, as h64's was.
4. **REVERSED.** KO-style beats ALTGP on Hartmann by >= 3.0 points.

## What this cannot settle

It tests the builder inside a greedy single-fidelity EI loop, not inside MF-DRO's
KO ensemble, where the same construction feeds a two-fidelity model with a
learned rho. A harness defect here would motivate, but not by itself establish,
the same defect in MF-DRO.
