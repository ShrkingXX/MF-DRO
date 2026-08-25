# H56 — does the DRO paper's ROI filtering help MF-DRO?

**CONFIRMATORY. Protocol committed before any run.**

## What the paper says

DRO (`papers/DRO.pdf`) §4.2: each ensemble member GP_m computes, from its own
posterior over D_{t-1},

    UCB_{m,t}(x) = mu_m(x) + sqrt(beta_t) sigma_m(x)
    LCB_{m,t}(x) = mu_m(x) - sqrt(beta_t) sigma_m(x)
    X_hat_{m,t}  = { x in X | UCB_{m,t}(x) >= max_{x'} LCB_{m,t}(x') }

the plausible-maximizer set. It constrains **rollout simulations only**, never
the real query (§4.3). Fig. 4b (Ackley 10D, 10 trials) reports DRO ROI beating
DRO GLOBAL: "converging faster and to a much higher objective value".

## What we currently do

**No ROI.** `simulate_mf_trajectory` builds a variable *named* `roi_candidates`
that is 200 uniform draws over the full domain, and the comment above it says so.
ROI was removed after a measurement that an earlier ROI version "produced zero
rollout steps within L2=0.2 of the true optimum across 280 sampled steps".

**That measurement had no control.** Measured here on the same setup, the
*global* pool also puts 0.000-0.050% of candidates within 0.2 of x*, i.e. ~0 per
200-point pool. The stated reason for deleting ROI describes the unfiltered pool
equally well.

## Choice of beta, and why it is a deviation

The paper calls beta_t "an exploration-exploitation trade-off parameter" without
giving a value. Measured on Hartmann 6D at this experiment's initial design
(seed 44, initial_hf=6/initial_lf=45), the HF posterior has mu spread 0.88-0.99
and mean sigma 0.457, so 2 sqrt(beta) sigma exceeds the entire mean range at any
conventional beta and the ROI admits everything:

| sqrt(beta) | accept m0 | accept m1 |
|---|---|---|
| 4.56 (Srinivas et al. 2010, as used elsewhere in this repo) | 100.00% | 100.00% |
| 2.00 | 100.00% | 100.00% |
| 1.00 | 99.90% | 94.35% |
| **0.50** | **20.30%** | **9.20%** |
| 0.25 | 3.55% | 0.95% |

Running the paper's criterion as literally specified would make both arms
identical and waste the compute. This experiment therefore uses
**sqrt(beta) = 0.5**, the largest value at which the ROI binds, and reports the
realised acceptance rate per run so a vacuous ROI is visible rather than silent.
This is a documented deviation from the paper, forced by our GP being far less
confident than theirs at this budget.

## Design

Hartmann 6D, seeds **44, 46, 48**, **cost_budget = 100** (post-init),
`initial_hf=6`, `initial_lf=45`, regression head (the new default),
`rollout_reward="mes_entropy"`. One variable:

| arm | rollout candidate pool |
|---|---|
| **GLOBAL** | 200 uniform draws over the full domain (current behaviour) |
| **ROI** | 2000 uniform draws filtered to X_hat_m, subsampled to 200 |

6 jobs, 4 concurrent workers (the autoresearch loop's h49 grid holds 10 of 15
cores).

## Locked predictions

1. **PAPER HOLDS**: ROI mean final HF simple regret < GLOBAL's, on >= 2/3 paired
   seeds.
2. **NULL**: ROI within noise of GLOBAL.
3. **HARMFUL**: GLOBAL better on >= 2/3, which given the enrichment measurements
   below would point at over-concentration.

Recorded per run regardless of outcome: mean/min ROI acceptance fraction, mean
and min normalised distance from ROI candidates to x*, and the fraction within
0.2 of x*. Global-pool references at this init: mean dist 0.876, frac<0.2
0.000-0.050%. If ROI's acceptance is >90% the run is a no-op and must be
reported as such, not as evidence about ROI.

## What this cannot settle

n = 3, one benchmark, one beta. The paper's ablation is Ackley 10D with 10
trials; this is neither. A null here does not refute Fig. 4b, it says the
mechanism does not transfer to MF-DRO on Hartmann 6D at a 100 cost budget.
