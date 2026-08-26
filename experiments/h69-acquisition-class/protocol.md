# H69 — Is Borehole an ACQUISITION-CLASS failure? (MES vs EI, surrogate matched)

**CONFIRMATORY.** Predictions locked before any h69 number exists.

## Why

h68 concluded Borehole is a MODEL failure: MF-DRO maximises its own MES
acquisition *well* — its choices outrank MI-Greedy's in 8 of 9 cells — and still
finishes 23.7% against MI-Greedy's 8.3%. So the acquisition does not correlate
with outcome there. "Model" is still two things: **surrogate** or **acquisition**.

Reading the standings by acquisition class (verified from source, not inferred —
`MFMIGreedyOptimizer._select_hf_by_ei` uses `LogExpectedImprovement`; its
information-gain term drives only the LF phase, which is inert at 100% HF):

| Borehole 8D | | Hartmann 6D | |
|---|---|---|---|
| MI-Greedy **[EI]** | **8.3%** | MF-MES [MES] | **8.5%** |
| MF-MES [MES] | 11.3% | SF-DRO [MES] | 11.5% |
| SF-MES [MES] | 13.3% | MF-DRO [MES] | 14.7% |
| SF-DRO [MES] | 15.1% | SF-MES [MES] | 21.4% |
| MF-DRO [MES] | 23.7% | MI-Greedy **[EI]** | **23.9%** |

The single improvement-seeking method is **best on Borehole and worst on
Hartmann**. Every information-seeking method is ordered against it consistently.

**This is not yet attributable to acquisition class.** MI-Greedy's HF GP is built
by `mf_baselines._build_gp`; SF-MES's by `_build_ko_style_gp`. Both are
SingleTaskGP + RBF/ARD + ScaleKernel + Normalize/Standardize, but the latter also
imposes an Interval lengthscale constraint and geometric-mean initialisation. Two
differences again — the lesson-22 trap. Also, MI-Greedy is only 12% HF on
Hartmann, so that column is confounded by fidelity too.

## Design — one difference, exactly

New arm **SF-EI**: SF-MES's optimizer with `LogExpectedImprovement` substituted
for the MES acquisition. **Identical surrogate** (`_build_ko_style_gp`), identical
loop, identical regret convention, identical initial design, no LF queries.
SF-MES vs SF-EI then differs in the acquisition and nothing else.

Borehole 8D and Hartmann 6D, seeds 44/46/48, cost budget 200. 6 jobs. Baselines
reused from h59 — no rerun.

A regression check runs first: SF-EI with the MES acquisition restored must
reproduce h59's SF-MES result exactly on one cell.

## Locked predictions

1. **PRIMARY.** SF-EI beats SF-MES on **Borehole** in >= 2/3 seeds and by >= 2
   points on the mean (SF-MES is 13.3%). This isolates the improvement-vs-
   information contrast with the surrogate held fixed.
2. **CONTROL / DISCRIMINATOR.** SF-EI is **worse than or equal to** SF-MES on
   **Hartmann** (21.4%). If EI helps on *both*, the story is not acquisition
   class at all — it is simply that EI is a better acquisition here, and the
   Borehole/Hartmann contrast in the table above is driven by something else
   (most plausibly MI-Greedy's 12%-HF fidelity mix on Hartmann).
3. **NULL.** No separation on Borehole. Then acquisition class is not the lever,
   the 5-point SF-MES-vs-MI-Greedy gap is surrogate or implementation, and
   h68's "model failure" narrows to the surrogate.
4. **REVERSED.** SF-EI worse on Borehole. Then the MI-Greedy advantage is
   definitely not its acquisition and the surrogate becomes the sole candidate.

## Why this matters for the north star

If PRIMARY and CONTROL both hold, the actionable consequence is concrete: DRO's
rollout reward is locked to `mes_entropy`, an information-seeking objective, and
the benchmark where it loses worst is the one where improvement-seeking wins. A
DRO variant with an improvement-aware reward becomes the first method change in
this project motivated by an isolated measurement rather than by intuition.

## What this cannot settle

n=3. It tests the acquisition in a *greedy single-fidelity* setting, not inside
DRO's rollout; a reward that works greedily need not work as a rollout target.
It also cannot explain *why* Borehole favours improvement-seeking.
