# h207 — Random-rollout dataset: does behaviour diversity make RTG selective? (Q3)

Registered BEFORE any h207 code. CONFIRMATORY on the arms; the Stage-0 diagnostics are
labelled as such.

## What this tests

Two questions, now separable because h198's "RTG is disconnected" has been retracted
(it was a config override; the channel is wired):

- **Q3-data:** every training rollout comes from ONE teacher, so at a given state there is
  exactly one action. RTG then varies only with the *world* (already in the state's 5M
  hyperparameter slots), never with *behaviour quality*, and a trained model has no reason
  to consult it. Put several behaviours at the same state into the data and RTG should
  become selective. **Untested. This is the live hypothesis.**
- **Q3-teacher (the user's spec):** best-of-N random search under each GP's own posterior
  is a realistic, deployable teacher (no oracle x*). Is its winner a better teacher than
  MES?

The DT paper's own §5.1 is the template: %BC (clone the top-X% trajectories) vs DT (train
on everything, let RTG select). Arm W is %BC; arm R is DT; arm MIX is Medium-Expert.

## The random-rollout generator (per ensemble member m, per BO iteration)

- N_rand = 100 rollouts, each `rollout_length` = 8 steps (fixed, matching every existing
  arm; see Concern 3 for why not cost-budgeted).
- Step τ: x ~ Uniform over the member's ROI candidate pool (the SAME 600-point pool MES
  chooses from — keeps the state/action support comparable); fidelity ~ Bernoulli(p_HF)
  with **p_HF = 0.5** (the existing `random` teacher hardcodes 0.25; exposed as
  `random_p_hf`, default 0.25 so h149 is unchanged).
- Fantasy y drawn from the member's progressively-conditioned posterior
  (`fantasy_mode='sample'`), exactly as MES rollouts do. State extraction unchanged.

## Score = expected FINAL SIMPLE REGRET under the rollout's own final posterior

For posterior P_τ (after τ fantasy observations), over the fixed pool X:

    IR_τ  =  b_τ  −  max_{x∈X} μ^HF_τ(x)

- `b_τ` = `_rollout_gumbel_b(P_τ)`: the existing Gumbel/Thompson estimate of E[max f]
  under P_τ, K_rtg draws, **CRN across all rollouts of a member** (same base normals).
  This is the "true optimum estimated from the posterior".
- `max μ^HF_τ` = expected value of the point the model would RECOMMEND now.
- IR_τ is therefore the expected simple regret of the recommendation — the frozen metric's
  quantity, estimated under the posterior instead of the true function.

**Alternatives for f\*, considered and not chosen:**
| estimator | why not |
|---|---|
| max posterior mean at τ=0, shared | ignores uncertainty; a rollout that shrinks b gets no credit |
| one Thompson draw's max | a *sample*, so argmin over 100 rollouts is a winner's-curse machine (h152's lesson, whole advantage vanished when fixed) |
| max of the rollout's own fantasy y's | biased low; HF-only; LF queries worth exactly 0 |
| **Gumbel E[max] (chosen)** | an expectation (low variance), already implemented, CRN-able, credits LF sharpening |

Within a member the argmin over rollouts uses IR_T; b_T differs per rollout so it is not
a constant shift and must be computed.

## RTG labelling (faithful to DT's return-to-go; a new `rollout_reward="inference_regret"`)

    r_τ      =  IR_τ − IR_{τ+1}           regret reduction bought by step τ (may be < 0)
    RTG[τ]   =  Σ_{τ'≥τ} r_τ'  =  IR_τ − IR_T
    RTG[0]   =  IR_0 − IR_T               IR_0 shared within a member ⇒ argmax RTG[0] ≡ argmin IR_T ≡ the winner

Relation to the two existing labels: `mes_entropy` = log b_τ − log b_T rewards shrinking
uncertainty about the max only; `improvement` = Σ max(0, y_HF − best) rewards realised HF
gains only and gives LF exactly 0. IR rewards both channels in one currency, which is the
one the frozen metric scores.

**At inference** (the DT eval loop `R ← R − r`): `rtg_tgt` from the existing schema
(running max of training RTG[0] with alpha), decremented each real step by the realised
IR reduction on the REAL posterior, `IR_real(n) − IR_real(n+1)` — computable from
`build_roi_pool` + `_rollout_gumbel_b` on the real KO, so every quantity is deployable.
Window history relabels as `rtg_tgt + IR_τ − IR_now`, the same telescoping shape h197
uses with log b. `mes_entropy` path untouched, bit-identical.

## State labelling — unchanged

`_extract_mf_state` on the rollout's own conditioned posterior. Random rollouts visit
states MES never would (that is the coverage Q3 wants). `step_norm` and `recent_hf_frac`
follow the rollout's own history as now.

## Arms — 3 x 5 seeds, Borehole_8D 42–46, all K=8, no positional embedding (h206), ROI-Q10

Matched control: **h206N = 11.39** (K=8, nopos, MES teacher, 60 MES rollouts/iter).
Reference: CTRL-K1 11.59; h149 random teacher (K=1, mes_entropy label) 43.94 = floor.

| arm | per member per iteration | total/iter | what it is |
|---|---|---|---|
| **W** | top-20 of 100 random by RTG[0] | 60 | %BC / the user's argmin spec |
| **R** | 20 random, NO selection | 60 | DT regime: RTG must do the selecting |
| **MIX** | 20 MES (= control's data) + 20 random | 120 | Medium-Expert |

W and R have the SAME count as the control (60), so "more data" is not a confound between
W, R and N. MIX is control + diversity; its MES half is byte-for-byte the control's.

**Why top-20 and not the argmin (k=1).** M = 3 here, so argmin gives 3 trajectories per
iteration against the control's 60. A failure would be a data-size failure, not a
design failure. k=20 matches the count exactly; k=1 is a registered dose if W is
interesting. Under `use_candidate_scoring=False` (the only acceptable head) the random
teacher's `scores=None` falls through to hard targets, as it must.

## Stage 0 — diagnostics and gates, run and reported BEFORE Stage 1

- **SC1 (GATE) — the reward fork is REAL this time.** Assert the constructed policy's
  effective `rollout_reward == "inference_regret"`, and that a short IR-labelled run and
  a short `mes_entropy` run produce DIFFERENT per-iteration `rtg_target`. This is the
  exact failure h198 died of.
- **SC2** — identity gate, `use_roi=False`, `122.29066752728207`.
- **SC3 — there is something to select.** Within one member's 100 random rollouts: spread
  of RTG[0] (report CV), and IR_T of the top-20 vs bottom-20. Report the top-20's LF
  fraction against the population's (Concern 3).
- **SC4 — the Layer-3 claim, measured.** R² of RTG[0] on the τ=0 state vector, for (a) 60
  MES rollouts and (b) 60 random rollouts from the same iteration. **Registered:
  R²(MES) > R²(random).** If not, the single-teacher account is wrong before any arm runs.
- **SC5 — winner quality vs MES.** IR_T of the top-20 random vs the 20 MES rollouts under
  the same member. Tells us whether W's teacher is competitive before spending ~15 h.
- **SC6** — the H168 RTG-sensitivity probe (re-query the same state at a sweep of RTG
  targets) runs on a control-config model and returns finite |dx/dRTG|.
- **SC7** — wall time of one iteration under W (300 rollouts generated) vs control.

## Predictions, committed now (band ±1.26; quality = final simple regret only)

- **P-R:** R − N > +1.26 — my lean is that R collapses toward h149's floor. If instead
  R lands within the band of N, RTG-based selection from diverse data WORKS and Q3-data
  is confirmed in the strongest possible way.
- **P-W:** W − N > +1.26. Best-of-100 random 8-step rollouts in 8-D is a weaker teacher
  than directed MES; W is h145 without the oracle.
- **P-MIX:** |MIX − N| ≤ 1.26, with the mechanism decided by SC6-style probing on the
  trained arms: **RTG sensitivity |dx/dRTG| of MIX ≥ 2× that of N** if the data
  hypothesis is right (RTG became selective and picked the MES-like half). MIX ≈ N with
  NO sensitivity gain means the DT ignored the random half by ignoring RTG.
- **Ordering, stated so it can be wrong:** MIX ≥ N > W > R. (My last two ordering
  predictions were inverted and half-inverted respectively; weight the argument, not me.)

## What each outcome RETRACTS

- **R within the band of N** retracts "best-of-random is a weak teacher" AND confirms the
  single-teacher account of RTG inertness — diversity alone was the missing ingredient.
- **R at the floor AND MIX ≈ N with no sensitivity gain** REFUTES the single-teacher
  account: diversity did not make RTG selective, so the cause is elsewhere (inference
  target OOD, `standardize_conditioning`, training scale). Q2 reopens on the architecture.
- **W within the band of N or better** retracts the standing assumption that MES is a
  competitive τ=0 teacher — random search under the posterior matches it.
- **SC4 with R²(MES) ≤ R²(random)** retracts Layer 3 as stated before any arm runs.

## Concerns and how each is handled

1. **h149 precedent** — all-random + mes_entropy at K=1 collapsed to the floor (0/3).
   R differs in three things: IR label, K=8 window, no positional embedding. If R
   collapses again the three are not separated; that is accepted, because R is a
   *falsifier* of Q3-data, not a dose.
2. **Winner's curse on the argmin** — the score is an expectation (Gumbel E[max]), not a
   draw, and CRN ties every rollout of a member to the same base normals. SC3 reports
   whether the top-20 separate from the bottom-20 by more than the Gumbel estimator's
   own noise.
3. **Fidelity bias** — under a random policy an LF query cannot inform a later random
   pick, so realised-improvement scores make LF worth 0 and winners HF-only. IR credits LF
   for sharpening the posterior, which partly repairs this; SC3 measures the residual
   bias (winners' LF fraction). Cost-budgeted rollouts would fix it fully but change what
   "a trajectory" is relative to every existing arm; registered as the follow-up if W's
   real-run LF fraction collapses.
4. **Two RTG conventions in one sequence** — the suspect for why h205A hurt. Not present
   here: no prefix, every token in a sequence carries the IR label.
5. **MIX imbalance** — 120 trajectories, half random; if RTG is not selective the
   location head regresses toward a blurred policy. That is the risk MIX is FOR.
6. **Compute** — 300 rollouts/iter under W vs 60 now, but random steps skip the MES
   argmax over 600 candidates; IR labelling costs one Gumbel per step, same as
   `mes_entropy`. SC7 measures it. 15 workers x 1 thread = 15 <= 15.

## Implementation plan (core changes, all gated, identity gate must hold)

1. `rollout_reward="inference_regret"` in `simulate_mf_trajectory`: per step `IR_τ`
   from `_rollout_gumbel_b` + `hf_posterior(pool).max()`; `r_τ`, backward cumsum.
2. `random_p_hf` config on the existing `random` teacher (default 0.25).
3. `rollout_mix` config consumed by `_generate_rollout_batch`: per member, a list of
   (policy, n, topk) specs; `topk` scores by rtg[0] after generation.
4. Inference: `IR_real` each iteration from the real KO; decrement rule; window relabel
   `rtg_tgt + IR_τ − IR_now` under the IR mode only.
5. Workers set the reward via `h83.ROLLOUT_REWARD` (never inside `_build`).
6. Readout: frozen metric via h83's `grid`, plus the H168 sensitivity probe on each
   arm's final model.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, metric IMPORTED from h83's
`sr_curve`/`grid`. Finals only, never `results/ckpt/`. Endpoint only. No p-values at n=5.
Every run and every gate reported, including misses.
