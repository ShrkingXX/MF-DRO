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

---

## Addendum (before any h56 result existed): third arm MESROI

The paper's UCB/LCB rule fails here because `max_x' LCB(x')` is *pessimistic* and
sinks as sigma grows, so every UCB clears it. A third arm replaces the bar with
the Thompson-sampled y*, which is a sampled MAXIMUM and does not degrade:

    p(x) = mean_k Phi( (mu_H(x) - y*_k) / sigma_H(x) ) = mean_k Phi(-gamma_k(x))
    ROI  = top-q of the raw pool by p(x),  q = 0.10

gamma is already computed by MES, so the criterion costs one Phi call. y* is
drawn from the fixed Sobol `y_star_pool`, never from the raw pool, because
`thompson_sample_y_star` output depends on |X| in location and scale.

### Measured before locking (Hartmann 6D, seed 44, same initial design)

- `corr(p(x), ||x - x*||)` = **-0.382** (member 0), **-0.370** (member 1), n=2000.
  The criterion carries real signal about where the optimum is.
- Mean normalised distance to x*, member 1: GLOBAL 0.8787, UCB/LCB sqrt(b)=0.5
  (accept 9.2%) 0.6895, MES-ROI top-10% (accept 10.0%) **0.7050**, top-5% 0.6305.
- **At matched acceptance the two rules concentrate equally.** MES-ROI is not
  claimed to be sharper.
- The `frac<0.2` "enrichment" figures this probe also produced are **not usable**:
  the base rate is 1 point in 2000, so every such ratio turns on whether one
  point survives. Not quoted, not locked against.

### Why it is still worth an arm

Control, not sharpness. UCB/LCB acceptance is an uncontrolled function of beta
and GP state -- 100% to 0.05% across a narrow beta range, and 20.3% vs 9.2% on
two members of the same ensemble at the same beta. top-q accepts exactly q
always, cannot go vacuous, and needs no beta.

### Locked prediction for the third arm

4. **MESROI vs GLOBAL**: same criterion as prediction 1 -- lower mean final HF
   simple regret on >= 2/3 paired seeds.
5. **MESROI vs ROI**: if MESROI beats ROI, the gain is attributable to a stable
   acceptance rate rather than to the shape of the plausibility set, since the
   two were measured to concentrate equally at matched acceptance.

---

## Addendum 2: the initial design was changed before any arm was scored

The first configuration (`initial_hf=6, initial_lf=45`, init cost 93 against a
post-init budget of 100) had **zero resolving power**. Both arms on seed 44
returned a regret curve that was flat at 0.7531 for all 16 iterations, and the
inference-regret curve was flat at the same value: at `c_H=8` a budget of 100
buys ~12 HF queries, and none of them beat the best of the initial 6. Two
different arms therefore produce a bit-identical number by construction.

Those runs are void and are kept, unanalysed, under
`results/void-budget100-init93/`. No arm comparison was computed from them.

Two candidate inits were validated on seed 44, GLOBAL arm only, before any arm
was scored:

| init hf/lf | init cost | init share of total | iters | regret | improvements |
|---|---|---|---|---|---|
| 6/45 | 93 | 65% | 16 | 0.7531 | **0 (flat)** |
| **3/20** | **44** | **31%** | 15 | 1.050 -> 0.3336 | **3** |
| 4/30 | 62 | 38% | 15 | 0.822 -> 0.2502 | 2 |

**The experiment runs at `initial_hf=3, initial_lf=20`**, chosen for the larger
number of improvement events and the smallest init share. The cost budget stays
at 100 as specified.

Caveat carried forward: 3 HF points is a thin basis for the KO model's rho, and
15 iterations is short. This ablation can detect a large ROI effect and nothing
subtle.
