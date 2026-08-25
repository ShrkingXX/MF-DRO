# H48 — standalone Takeno MF-MES: validation and what it changes

## Answer to the question that prompted it

**Variant D (h47) does not implement the spec.** It monkeypatches a proposal
into `DirectMFRegretOptimization`, so the rollouts, the DT training and the
2-iteration cold-start HF override all still run (D3 violated wholesale), it
uses uniform random rather than Sobol, shrinking-ball search rather than
L-BFGS-B, top-8 rather than top-10, and Thompson rather than Gumbel f*. It also
imports the very mf_dro kernels V5 exists to audit, so its V5 would have been
a tautology at r=1.0 by construction. H48 is the spec, built independently.

## Validation — all six pass

| | result | bar |
|---|---|---|
| V1 HF closed form vs fine quadrature | max rel 1.5e-8 | < 1e-6 |
| V2 32-pt GH vs 10,000-pt trapezoid | max rel 2.3e-5 | < 1e-4 |
| V3 MI non-negativity, 10,000 inputs | 0 clamps | I >= -1e-8 |
| V4 rho=1, delta->0 => MES_L == MES_H | rel 3.7e-7 at var_delta=1e-8 | MC error |
| V5 vs mf_dro on identical inputs | **Pearson 1.000000**, max abs 1.1e-16 HF / 2.2e-8 LF | report |
| V6 MF-MES vs SF-MES at fixed cost | 0.2069 vs 0.8104, **10/10**, p=0.0020 | must win |

### Two implementation facts found along the way

**numpy's `hermgauss` returns NaN weights at n=512.** An auto-escalation to 512
nodes silently produced NaN, i.e. escalating made the sharp regime *worse*.
Switched to `scipy.special.roots_hermite` (stable past 1024).

**No polynomial rule resolves a step**, and Phi_cond becomes a step in v as
s -> 0. The KO decomposition gives that limit in closed form
(H1 = (G/Phi_H)[H_trunc - log(G/Phi_H)]), so above r = rho*sigma_L/s = 25 the
analytic branch takes over. 25 is the measured crossover against a 4,000,000-
point trapezoid:

| r | 2 | 6 | 12 | 20 | 40 | 100 |
|---|---|---|---|---|---|---|
| GH-256 | 1.8e-14 | 4.6e-8 | 2.1e-3 | 2.3e-2 | 4.1e-2 | 3.0e-2 |
| analytic | 4.6e-1 | 1.5e-1 | 7.0e-2 | 4.2e-2 | 2.1e-2 | 8.2e-3 |

Neither is better than ~4% for r in [12,60] — stated, not hidden. Real
Hartmann 6D runs at r median 0.77, p99 1.41, max 2.06, i.e. the GH-exact
regime throughout, where V2 holds at 2.3e-5.

## V5 is the substantive validation result

Two independent implementations — physicists' vs probabilists' Hermite, 128 vs
32 nodes, separately derived KO algebra — agree to **machine precision**.
**mf_dro's MES kernels are mathematically correct.** The teacher's weakness was
never the acquisition formulas; it is entirely the 200-point pool and the
pool-derived y*.

## Results, 10 seeds, identical initial design, cost budget 200

| method | final HF regret |
|---|---|
| MF-MES (natural KO-GP defaults) | 0.2069 +/- 0.0506 |
| MF-MES (mf_dro member-0 surrogate) | 0.3408 +/- 0.0727 |
| MF-DRO / joint MES | 0.4007 +/- 0.0475 |
| h31 teacher (pool-200) | 0.4781 +/- 0.0414 |

Four paired tests were run, so the bar is Bonferroni 0.05/4 = 0.0125.

| test | diff | wins | p | survives |
|---|---|---|---|---|
| MF-MES-def vs MF-MES-matched | -0.1339 | 9/10 | **0.0039** | **yes** |
| MF-MES-def vs MF-DRO | -0.1938 | 8/10 | 0.0488 | no |
| MF-MES-matched vs h31 pool | -0.1373 | 7/10 | 0.1602 | no |
| MF-MES-matched vs MF-DRO | -0.0599 | 6/10 | 0.6250 | no |

### What is and is not claimed

**NOT claimed: "proper MF-MES beats MF-DRO."** p=0.0488 is a bare-threshold
result that does not survive correction for the four tests run. This project
already retracted one significance claim that came from exactly this pattern.

**Claimed, and it survives correction: mf_dro's ensemble lengthscale-diversity
grid handicaps its own surrogate.** Member 0 is anchored at
`initial_lengthscale=0.1839`, the short end of a grid whose stated purpose is
decorrelating rollouts for the DT. Holding the acquisition fixed and changing
only that costs 0.134 regret on 9 of 10 seeds, p=0.0039. This is a defect in
the DRO pipeline with nothing to do with the Decision Transformer.

**Also claimed: multi-fidelity itself works here.** MF-MES beats SF-MES on
10/10 seeds at matched cost (0.2069 vs 0.8104, p=0.0020).

With the surrogate matched, MF-MES and MF-DRO are not distinguishable at n=10
(p=0.625). That is a failure to detect a difference, **not** evidence of
equivalence — s.e. ~0.07 against an observed -0.06 means the comparison is
underpowered for effects this size.

## Deviations from the spec, stated

- Surrogate is KO-GP, not SLFM (D1, deliberate: holds the surrogate fixed
  against MF-DRO). `n_models > 1` averages the acquisition over an ensemble.
- f* by Gumbel sampling, not RFM with 1000 bases (RFM for the KO kernel is
  nontrivial). `fstar_method="thompson"` available for cross-checking.
- Sobol pool is 2048, not 2000: Sobol's balance properties hold only at powers
  of 2 and scipy warns otherwise.
- L-BFGS-B uses **finite differences, not autograd**: the path runs through
  gpytorch posteriors under `no_grad`, scipy CDFs and a numpy Gauss-Hermite
  sum, so no tape exists. Differences are batched — one vectorised call of
  d+1 points per step. L-BFGS-B improved 99.9% of starts over 20 runs, i.e.
  the Sobol pool alone is essentially never at the optimum.
