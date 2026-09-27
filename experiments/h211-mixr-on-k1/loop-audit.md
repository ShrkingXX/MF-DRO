# Audit: the MF-DRO-MIXR loop vs original DRO (`papers/Old_dro.py`) and DT (`papers/DT.pdf`)

EXPLORATORY: code reading plus a measurement on saved runs. No new runs.

## 1. What is labelled, and how

### State — `_extract_mf_state`, dim 5M + d + 5 + 4R
`[gp_lf lengthscale, gp_lf outputscale, gp_delta lengthscale, gp_delta outputscale, rho] x M`
(constant across a whole BO iteration's batch), `best_value_HF`, `step_norm = n_real_iter /
T_real`, `best_position_HF` (d), `recent_hf_frac` (last 5 fidelities), `c_L/c_H`,
`use_dkl_flag`, and `[mu_H, sigma_H, mu_L, sigma_L]` at R fixed reference points computed
from **that rollout's own progressively-conditioned model**. The reference block is the
only part that varies within a rollout. **MIXR's random rollouts are labelled by the same
function** — the policy identity never enters the state.

Original DRO: `_extract_state(data_x, data_y, n)`, single fidelity, no ensemble block, no
reference grid. Ours is a superset; no inconsistency.

### Reward and RTG

| | DT paper | `Old_dro.py` | MF-DRO `mes_entropy` | MF-DRO `terminal_improvement` (MIXR/NIR) |
|---|---|---|---|---|
| r_τ | environment reward | `max(0, y_τ − best)`, **clamped ≥ 0** (L816–829) | — | 0 for τ<T |
| RTG[τ] | Σ_{τ'≥τ} r | Σ_{τ'≥τ} r (L899) | `log b_τ − log b_T` (endpoint entropy difference) | `max(fantasy HF) − real incumbent`, **constant over τ** |
| normalised? | no | no | no | **÷ running max \|rtg[0]\|** |

`terminal_improvement` is formally a *valid* return-to-go: with intermediate rewards zero,
Σ_{τ'≥τ} r = r_T for every τ (DT §5.6's delayed-return setting). But it **drops Old_dro's
clamp**, so RTG is usually negative — on Borehole at cost 40, the MES half labels ≈ −25 and
the random half ≈ −70.

## 2. The RTG signal actually fed at inference — measured, not inferred

`rtg_tgt = max(batch_max rtg[0], alpha_rtg · running_max rtg[0])`, alpha = 0.5,
`running_max_rtg` initialised **0.0** (mf_dro.py:2290), evaluated *after* the batch
normalisation. Measured on saved runs (the `rtg_target` array each run records):

| run | n iters | distinct values | range | mean |
|---|---|---|---|---|
| MIXR-B (`terminal_improvement`) | 117 | **9** | 0.500 – 1.000 | 0.511 |
| NIR-B (`terminal_improvement`) | 134 | 34 | 0.500 – 1.000 | 0.546 |
| CTRL-K1-H (`mes_entropy`) | 116 | 82 | 0.299 – 0.859 | 0.539 |
| h84 ROI-Q10-B (`mes_entropy`) | 115 | 110 | 0.319 – 1.127 | 0.676 |

**Under `terminal_improvement` the target is pinned at 0.5 for essentially the whole run.**
Mechanism: early on the incumbent is low, some fantasy rollouts beat it, rtg[0] > 0 and
`running_max` climbs to ≈ 1. Once the incumbent rises, no rollout beats it, so `batch_max`
is negative and `max(negative, 0.5 · 1)` returns the alpha term forever. The schema
`max(batch_max, alpha·running_max)` is a *floor* for non-negative RTG (Old_dro's regime);
with negative RTG the alpha term becomes a permanent **ceiling above everything in the
training data**.

**And under K=8 every window token carries that same constant** — the terminal-return
relabel sets each history token to `rtg_tgt` (intermediate rewards are zero). So all 8 RTG
inputs are the identical constant 0.5, every iteration, all run.

## 3. Consistency verdicts

- **vs DT**: RTG *form* is faithful (return-to-go of a terminal reward). The *inference
  rule* is not: DT initialises a target and decrements it by the **achieved** reward
  (`R = R + [R[-1] − r]`); we recompute from the current rollout batch each iteration.
  `mes_entropy`'s history relabel (h197 telescoping) approximates DT's rule; the
  terminal path does not.
- **vs Old_dro**: ours generalises the reward (MF, endpoint-entropy or terminal) but
  **drops the clamp** that keeps RTG non-negative — which is what breaks the target schema.
  Old_dro's own inference target is a **hardcoded 1.0** with no normalisation, so it is
  constant by construction too, but for a different reason.

## 4. Consequence for the MIXR result

**MIXR's gain cannot be an RTG-conditioning effect: the RTG input is a constant.** This is
an independent confirmation of the measured RTG sensitivity (1.1–1.4×) and of P-NIR's
neutrality, and it narrows the surviving mechanism to the training-data composition
changing the learned **state → action** map, with the conditioning channel inert by
construction. It also means the user's Q2 goal ("both state and RTG should matter") is
structurally blocked at inference under this label, independent of architecture.

**Caveat on h211**: MIXR-K1 uses `mes_entropy` (the h84 base label), so its RTG target will
*vary* (82–110 distinct values) where h207's MIXR's was pinned. MIXR-K1 vs CTRL-K1 remains
a clean one-factor test of the random half; but MIXR-K1 vs h207's MIXR is confounded by
the label, and K1-TI-H is what separates them.

## 5. Concrete fix, if we want RTG to carry information (not registered, not run)

Make the target scale-aware for signed RTG: a high percentile of the batch's own rtg[0]
distribution (e.g. 90th) rather than `max(batch_max, alpha·running_max)`, or min-max
normalise rtg[0] into [0,1] per batch before the schema. Either makes the target track the
batch and stay inside the training distribution. Cheap to test as an A/B on one benchmark.
