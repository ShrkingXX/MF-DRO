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

---

## AMENDMENT (before launch, 2026-09-15) — a confound I missed, and a 4th arm

W, R and MIX all use the `inference_regret` label. The control h206N uses `mes_entropy`.
So every arm-vs-N comparison would have changed TWO things: the data composition AND the
reward label. h60's genuine reward fork moved the outcome by ~+1 (0/3), so the label
alone is not nothing. As written, MIX ≠ N could not be attributed.

**Arm N-IR added:** h206N's configuration exactly (20 MES rollouts/member, K=8, no
positional embedding, ROI-Q10) with `inference_regret` as the label. It is the matched
control for W/R/MIX — one change from each — and it carries the H168 probe, which h206N
did not, so the RTG-sensitivity comparison for MIX's mechanism has its baseline.

- **P-NIR:** |N-IR − N| ≤ 1.26. The label alone should not move the endpoint; if it does
  by more than the band, h60's small effect was real and the IR label is itself an
  intervention, which reframes the other three arms.
- **Predictions P-R, P-W, P-MIX are now scored against N-IR**, not N. Their signs and
  bands are unchanged.

**Compute:** 4 arms × 5 seeds = 20 > 15. Launch order: N-IR, R, MIX first (15 workers);
W is queued by the supervisor and launches seed-by-seed as slots free. W is last because
it is the arm whose interpretation depends least on the others landing first.

**Every arm carries the H168 probe** (RNG-neutral: state saved and restored around it,
so the trajectory is bit-identical to an unprobed twin). Sweep [0, 0.02, 0.05, 0.1, 0.2,
0.3, 0.5, 0.75, 1.0] in normalized-RTG units. Sensitivity = mean over the last 30
iterations of max over the sweep of |x(rtg) − x(rtg=0)|.

## AMENDMENT 2 (before launch) — longest-first scheduling

20 jobs on 15 slots. The first launch order (NIR, R, MIX, then W) put the slowest arm
entirely after the fast ones: ~5 workers on 15 cores for W's whole ~3 h. Reordered
**longest-first** — W, MIX, NIR launch immediately (15), R fills slots as they free —
so wall ≈ max(T_W, T_NIR + T_R) instead of T_NIR + T_W, with cores busy throughout.
The supervisor derives the order from SC7's *measured* per-iteration wall rather than
my estimate. Threads stay 1/worker: raising `torch.set_num_threads` mid-run changes
floating-point reduction order and would break bit-reproducibility against a 1-thread
run, which this project's CRN and identity checks depend on. No arm, seed, or analysis
changes.

## AMENDMENT 3 — Stage 0 v1 GATE MISS (reported in full), diagnosis, fix, new SC

**Stage 0 v1: FAIL** (SC0 PASS, SC1 PASS, SC6 PASS, SC7 PASS; **SC3 FAIL, SC4 FAIL**).
The supervisor launched nothing. Full log kept as `logs/stage0_v1_FAIL.log`.

```
SC3: m0 IR_0 spread=2.28e+01  top20 IR_T=10.05  bot20=25.89   (m1, m2 similar)
SC4: eta^2(MES)=0.064  eta^2(random)=0.096  -> registered inequality FAILS
SC5: MES-20 IR_T 11.5/13.9/13.8  vs  top20-of-100-random 14.5/21.2/16.0  (MES better, all 3)
SC7: MES-20/member 22.1 s   random-100/member 72.7 s   -> W is 3.3x the control per iteration
```

**Diagnosis — one cause for both failures.** `_rollout_ir` scored each rollout on its own
`roi_candidates`, which is a fresh `torch.rand` draw per rollout. IR_0 — meant to be
shared within a member — varied by ~23 IR-units from pool sampling alone, comparable to
the top20/bottom20 gap (~16). The argmin was partly selecting on pool luck: the exact
winner's-curse failure the CRN was added to prevent, re-entering through the pool rather
than the Thompson draws. SC4's η² was computed on labels dominated by that same noise, so
0.064 vs 0.096 is a noise-to-noise ratio and says nothing about Layer 3 yet.

**Fix (identity gate PASS).** A **run-fixed, domain-spanning scoring pool** `ir_pool`
(600 scrambled-Sobol points, seeded per run, built only under the IR label). Every
rollout of every member and the real posterior at inference score on the same points.
IR_0 is now exactly shared within a member; IR_τ is comparable across real iterations
(the per-iteration `_b_real` pool resample, which the code's own comment records as
making b "flat and non-monotone", no longer touches the IR label). Behaviour path
(`roi_candidates` for choosing x) unchanged, so SC0 parity is re-run, not assumed.

**SC8 added — label reliability (test-retest).** Each rollout's final posterior is
rescored with an independent Thompson draw (`ir_probe_second_seed`, Stage 0 only,
RNG-restored). reliability = (V_within − V_noise)/V_within. **Registered:
reliability(random) > reliability(MES)** — diversity raises the signal fraction of the
label. This is Q3-data restated in its cleanest form: for single-teacher data the label
may be mostly fantasy noise, in which case a DT that ignores it is behaving correctly and
the remedy is behaviour variance, not a different architecture.

**SC3 tightened:** IR_0 spread must be exactly 0; the top-20/bottom-20 gap must exceed
2× the SC8 noise sd; ranking is by rtg[0] as the code selects.

**SC5 (diagnostic) as measured in v1:** MES beat the best-20-of-100 random winners on all
three members (+4.2 IR-units mean). Under pool noise, but directionally as leaned (P-W).
Re-measured in v2.

**SC7:** W ≈ 3.3× the control per iteration (~4.6 h/seed). Longest-first order confirmed
by measurement: W, MIX, NIR, R.

## AMENDMENT 4 — Stage 0 v2 GATE MISS on SC8 (a finding), and a score diagnostic before v3

**Stage 0 v2: FAIL on SC8 only** (SC0/SC1/SC3/SC4/SC6/SC7 PASS). Nothing launched. Log
kept as `logs/stage0_v2_FAIL.log`.

```
SC3  IR_0 spread 0.00e+00 on all members; top20/bot20 gap = 16-21 x noise sd   PASS
SC4  eta^2(MES)=0.042  eta^2(random)=0.020                                    PASS (inequality)
SC8  reliability  MES 0.993 (within sd 7.85, noise sd 0.67)
                  random 0.978 (within sd 4.56, noise sd 0.68)               FAIL
SC5  MES IR_T ~28 vs top-20 random ~19: random winners BETTER 3/3  (v1: MES better 3/3)
SC7  W = 2.98x control per iteration
```

**SC8 refutes the "cleanest form" I registered in Amendment 3.** The single-teacher label
is NOT mostly noise: 99% of within-member IR_T variance is real. MES rollouts vary hugely
in outcome (sd 7.8) for the same τ=0 action, driven by fantasy luck. So RTG in
single-teacher data is reliable outcome information that is **independent of the action**
— I(action; RTG | state) ≈ 0 by construction. SC4's inequality passes but its magnitude
(η² = 0.04) refutes the "RTG tracks the world, which is in the state" story as well:
member identity explains 4% of RTG variance. **Layer 3, restated correctly: diversity's
job is to make RTG depend on the action, not to reduce its noise or its redundancy with
the state.** Stage 0 v3 gates on that quantity directly (R² of the score on the τ=0
action within member, random vs MES).

**SC5 flipped and is not trusted.** Changing the scoring pool from per-rollout ROI-600 to
a fixed global Sobol-600 moved MES's IR_T from ~12 to ~28 while moving the random
winners' from ~14 to ~19. A 600-point global pool in 8-D cannot resolve the peak MES
rollouts sharpen, so the score under-credits exactly what MES does well — a bias toward
the arm under test. Launching W on such a score would be launching on a thumb on the
scale.

**Stage 0b (running):** the same batches scored four ways — IR on Sobol-600 (v2), IR on
Sobol-3000 (resolution check), IR on ONE seeded ROI-600 pool per member (shared within
member, resolves the peak), and IMP = best OBSERVED fantasy HF minus the real incumbent,
which is the frozen endpoint metric's own quantity and needs no pool. Reports SC5 under
each, MES's IR level by pool, rank agreement between scores, and the action-R² above.
The score for v3 is chosen from this and registered before v3 runs; the frozen metric's
own quantity (IMP) is the default unless the diagnostic gives a reason to prefer IR-ROI.

## AMENDMENT 5 — Stage 0b result; label changed to the frozen metric's own quantity; W deprioritised

**Stage 0b (score diagnostic, same batches, four scores):**

```
SC5 by score    MES-20 vs top-20-of-100 random (selected by that score), 3 members
  IR-Sobol600   MES 27.9/29.4/29.0   winners 16.8/21.0/20.3   random wins 3/3
  IR-Sobol3000  MES 24.8/29.7/25.7   winners 18.3/23.6/19.7   random wins 3/3
  IR-ROI600     MES 26.4/28.7/30.2   winners 17.7/23.7/19.8   random wins 3/3
  IMP (best observed HF − incumbent)
                MES  −4.8/+1.6/−15.2  winners −40.9/−27.2/−42.7   MES wins 3/3
MES IR level by pool: 28.8 / 26.7 / 28.4 (random 25.8 / 26.4 / 25.6)  -> resolution hypothesis REFUTED
Spearman between scores on 300 random rollouts: 0.00 to 0.27 (IR variants disagree with EACH OTHER)
Action-R² at τ=0 (random, n=100/member): 0.07–0.13 under every score
```

**Three conclusions, each of which changes the design:**

1. **The resolution explanation for SC5's flip was wrong** — MES's IR level barely moves
   across pools. Under IR, MES rollouts genuinely score no better than random ones on
   average. IR = E[max f] − max μ over a broad pool is dominated by variance at the many
   far-from-data points, so it rewards *global variance reduction*, which spread-out random
   queries do about as well as MES. IR is an information criterion in disguise, not a
   regret.
2. **IR variants do not even agree with each other** (ρ 0.14–0.27) on which random rollouts
   are good. SC8's 0.98 reliability was reliability *given a pool*; the ranking is a
   property of the pool. IR is unusable as a selection score.
3. **Under the frozen metric's own quantity, the best of 100 random 8-step rollouts is
   27–43 units below the real incumbent on every member**, while MES rollouts sit at it.
   Random search in 8-D over 8 steps does not find good points. **The user's argmin
   spec produces a dataset that never beats the incumbent** — W is now predicted to fail
   with high confidence, and MIX's random half is ~40 units below its MES half.

**Label for v3: `terminal_improvement`** = best observed fantasy HF − real incumbent,
unclamped, written at every τ (a terminal-only return; DT §5.6). No pool, no f\*: the
posterior optimum is a per-member constant that cancels in every ranking and difference.
This IS "final simple regret" of the rollout up to that constant — faithful to how the
endpoint metric reads `max y_HF`. The existing `improvement` label clamps at 0 and would
tie every random rollout at 0, destroying the ranking W and R depend on. Inference
relabel: history tokens carry the current target (intermediate rewards are zero). SC0
parity re-run against the last pre-implementation commit (8b25650).

**Registered under the new label, scored against N-IR (now "NIR" = 20 MES + this label):**
- **P-NIR:** |NIR − h206N| ≤ 1.26 (h60's `improvement` fork moved things ~+1, 0/3; this
  label is its unclamped terminal cousin).
- **P-MIX (the arm this experiment now exists for):** the halves are separated by ~40
  units in the label (SC5 gate: Cohen d > 1 on every member). If RTG is selective at all,
  MIX ≈ NIR (|MIX − NIR| ≤ 1.26) AND MIX's RTG-sensitivity ≥ 2× NIR's. If the DT ignores
  RTG, it regresses toward a blurred policy: MIX − NIR > +1.26 with no sensitivity gain.
  **Lean: MIX − NIR > +1.26** (RTG not selective; the mechanism question of Q2 reopens on
  the architecture side).
- **P-R:** R collapses toward the floor (R − NIR > +5). Its dataset never beats the
  incumbent; no label can select what is not there.
- **P-W:** W − NIR > +1.26, high confidence, for the same reason. Queued LAST; if the user
  prefers, it can be dropped without loss to the MIX/NIR/R question.

**Stage 0 v3 gates:** SC0, SC1, SC3 (distinct ranking among random rollouts), **SC5 (halves
separated, d > 1)**. Diagnostics: SC4 action-R², SC6, SC7. SC8 dropped (label is a
deterministic function of the rollout).
