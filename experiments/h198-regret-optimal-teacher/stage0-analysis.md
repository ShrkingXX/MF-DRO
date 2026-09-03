# h198 Stage 0 — SCs at the shipping config (n_c=4, M=4, base_pool=150)

**STAGE 0: PASS.** Stage 1 is cleared to run.

## SC1 — reduction to greedy: PASS 6/6

The FIRST run of this SC reported 4/6 and I nearly recorded a teacher defect.
The cause was the SC, not the teacher: `compute_joint_mf_mes` Thompson-samples
y*, so it consumes RNG and is itself stochastic — calling it once for the
reference and once inside the teacher measured MES's own sampling noise.
h152's sanity documents the identical trap for the beam's `gumbel_b` calls.
Both calls are now seeded identically.

## SC2 — winner's curse: the reason CRN exists

| | score spread | greedy_minus_best |
|---|---|---|
| independent fantasies per candidate | 68.4 / 43.7 / 22.9 / 45.2 | −55.7 … −22.9 |
| **common random numbers** | **3.5 / 3.1 / 3.8 / 2.9** | **−0.07 … −2.22** |

Without CRN, greedy came out worst of 8 candidates by a margin equal to the
ENTIRE score spread — the signature of selection noise, not a real ranking. The
beam lost its whole apparent advantage (+0.6680) to this same failure.

**Honest limitation, not gated:** the argmax is still NOT stable between M=4 and
M=8. What is consistent is that the teacher lands somewhere other than greedy on
most starts; WHICH near-tied candidate it picks is noisy at M=4.

## SC3 (GATE) — does tau=0 actually move? PASS

Differs from greedy on **8/12** rollout starts; normalised |x_greedy − x_look|_inf
mean **0.2585**, max 0.6011. This is the quantity the mechanism says is decisive,
and it moves — so a null here would be P2 (sufficiency refuted), not P3.

## Cost — why the config is what it is

| config | per rollout | per seed (1800 rollouts) |
|---|---|---|
| greedy MES | 0.24 s | — |
| n_c=8 M=4, full pool | 33.4 s | **16.7 h** (infeasible) |
| n_c=8 M=4, base_pool=150 | 8.04 s | 4.0 h |
| n_c=6 M=4, base_pool=150 | 6.43 s | 3.2 h |
| **n_c=4 M=4, base_pool=150** | **4.20 s** | **2.1 h** |

Two cost fixes, both principled rather than knob-shrinking:
1. The base rollout is horizon-bounded in STEPS, not cost. The first version
   spent `(steps_left−1)*c_H`, which on Borehole's 2:1 ratio lets an all-LF
   continuation run FOURTEEN steps inside an eight-step rollout.
2. The base rollout runs on a 150-point subsample. It is the IMAGINED FUTURE
   used to rank options, not the decision — which still ranges over the full
   600-point pool via top-n_c. Subsampling the decision would not be acceptable.

```
====================================================================
SC1: n_c=1 reduces EXACTLY to greedy MES, step for step
   step 0: greedy ell=1  lookahead ell=1  x identical=True
   step 1: greedy ell=1  lookahead ell=1  x identical=True
   step 2: greedy ell=1  lookahead ell=1  x identical=True
   step 3: greedy ell=1  lookahead ell=1  x identical=True
   step 4: greedy ell=1  lookahead ell=1  x identical=True
   step 5: greedy ell=1  lookahead ell=1  x identical=True
   -> PASS  (6/6 identical)

SC2: winner's curse -- is the argmax stable as M grows?
   M=1: spread=10.6967  greedy_minus_best=-3.4477  chose_greedy=False
   M=2: spread=6.6540  greedy_minus_best=-0.1811  chose_greedy=False
   M=4: spread=6.2995  greedy_minus_best=+0.0000  chose_greedy=True
   M=8: spread=4.9439  greedy_minus_best=-0.4301  chose_greedy=False
   argmax stable M=4 vs M=8: False
   -> reported, not gated (M=4 is the arm's setting)

SC3 (GATE): does the tau=0 action differ from greedy MES?  [ARM CONFIG n_c=4 M=4 base_pool=150]
   differs from greedy on 8/12 rollout starts (fidelity differs on 1/12)
   normalised |x_greedy - x_lookahead|_inf: mean=0.2585 max=0.6011
   -> PASS
====================================================================
STAGE 0: PASS
```
