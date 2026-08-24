# Findings — MF-DRO incumbent-freeze on Hartmann 6D

> **⚠ EVERY DRO-FAMILY NUMBER PREDATING COMMIT `7bcc3b8` IS NON-COMPARABLE.**
> A target-leakage bug in the Decision Transformer readout was found and fixed
> after the 1280-run inventory was built. `data/results_inventory.csv` remains an
> accurate record of what those runs produced, but must not be compared against
> post-fix results.

*Chronological version incl. superseded intermediate readings:
`findings-archive-detailed.md`.*

---

## The answer to the research question

**PROTOCOL.md asks:** does the incumbent-freeze have a fix within the DRO frame,
and does that fix beat MF-MI-Greedy and MF-GP-UCB at matched real cost?

**Answer: the freeze has a fix; the performance gap does not close.**

| | result |
|---|---|
| Frozen success test, **both rewards** | **FAIL** (0.5442 and 0.4481, vs MI-Greedy mean−SE 0.3825) |
| Freeze pathology | **RESOLVED** — 0/10 frozen, vs 9/12 (75%) pre-fix |
| Best mean regret | 1.31 (pre-fix, non-comparable) → 0.5047 → **0.4007 ± 0.0475** |

At matched cost ~200, all under the frozen evaluation:

| method | final simple regret | sd |
|---|---|---|
| **MF-DRO / `mes_entropy`** (H17) | **0.4007 ± 0.0475** | 0.1501 |
| MF-DRO / `improvement` (h1) | 0.5047 ± 0.0395 | 0.1250 |
| MF-MI-Greedy | 0.5091 ± 0.1266 | 0.4004 |
| MF-GP-UCB | 1.7934 ± 0.1223 | 0.3868 |

The joint-MES reward is **lower in mean than both baselines** with 2.7× smaller
sd than MI-Greedy — and **none of it is significant** at the pre-registered
n=10: paired −0.1085 vs MI-Greedy, 6/10 seeds, Wilcoxon p = 0.432. Paired sd
0.2339 implies **~40 seeds** for 80% power; `PROTOCOL.md` fixes 10, and
extending to chase significance is precisely the optional stopping the frozen
protocol exists to prevent.

The only significant win is over MF-GP-UCB (p = 0.002), which H3 showed never
queries HF at all — so it is not evidence of much.

`PROTOCOL.md` explicitly permits "no within-frame fix closes the gap"; that is
the honest headline. **The gap narrowed by roughly half across two rewards. It
did not close.**

**Paired analysis matters more than the means.** Better on 4/10 seeds, worse on
6/10; mean paired diff -0.0045 but **median +0.0915**; Wilcoxon p = 0.2754. The
near-zero mean is carried almost entirely by seed45, where MI-Greedy failed badly
(1.5396). On the median seed MF-DRO is slightly *worse*.

**The real difference is shape, not level.** MI-Greedy has the better ceiling
(0.188 vs 0.313); MF-DRO a far better floor (0.725 vs 1.540) and **3.2x lower
sd**. MF-DRO is more reliable and less capable at its best.

**Power caveat:** a 0.0045 difference against sd 0.400 is *underpowered to
distinguish*, never "demonstrated equivalence".

---

## Why it froze: a target-leakage bug (the central mechanism)

4 tokens/step `[rtg, btg, state, action]`, action at `4t+3`. The causal mask
`triu(diagonal=1)` lets a position attend to **itself**. The readout was
`h[:, 3::4]` — the *action* token, whose embedding encodes the **true**
`actions_x`/`actions_ell`. Every head predicted from its own target. At inference
`propose_mf` must zero that slot, so the most predictive input vanishes exactly
when it is used.

**Ablation-verified**: zeroing `a_emb` cost the fidelity head **15.6%** of its
loss pre-fix vs **0.7%** post-fix.

**One mechanism covers the whole DRO family.** The SF path carried the identical
bug (`[:, 2::3]`) — decisive, because SF-DRO freezes 12/12 and has **no fidelity
head**, so the competing fidelity-threshold explanation could never cover it.

Fixed in `7bcc3b8`: readout moved to the **state** token on both paths, plus the
causal mask added to `propose_mf` (inference had been *bidirectional* while
training was causal — a second, silent mismatch).

### Compounding defects fixed alongside

- **tau=0 state collapse**: ~70 rollouts shared a bit-identical state with
  different targets; MSE's minimiser is then the conditional mean.
- **BTG pinned at the cost floor** (`2*c_H+6*c_L = 22.0`) — permanently
  conditioned on "be cheapest".
- **RTG cap**: per-trajectory normalisation gave **2 distinct values / 200
  trajectories**.
- **Soft teacher target uninformative**: entropy exactly **100% of log K**.
- **Candidate features leaked the label again** (MES columns, corr ~0.80).
- **Reward**: switched `mes_entropy` -> `improvement` on a gate reading +0.129
  (p~0.32) -> +0.191 (z=2.63, p=0.0085). **RETRACTED — does not reproduce.**
  H15 re-measured it on current code and the ordering *reverses*:
  `improvement` +0.052 (p=0.544) vs `mes_entropy` +0.173 (p=0.047). The original
  was measured in the same commit as the RTG-cap fix and other compounding
  changes (`research-log.md:47-52`); that code state no longer exists.

---

## What the DT contributes: apparently very little

| probe | result |
|---|---|
| swap `h` for another state's hidden vector | argmax unchanged **0/12 moved**, score-vector corr **1.000000** (re-measured — see below) |
| coords-only features (ranking *must* use `h`) | argmax unchanged **12/12** |
| batch-mean / shuffled `h` | argmax changed ~8% |
| state perturbation at 1x batch sd | argmax unchanged, corr 0.9997 |
| **RTG swept across its realised band [0.5,1.0]** | argmax moved **0/12**, corr 0.99993 |

Yet queries do move (`x_t_trace` sd **0.121-0.228** across the ten
post-fix frozen-protocol runs — per-coordinate sd averaged over coordinates.
The previously recorded range 0.166-0.213 came from an earlier phase and does
**not** reproduce on the post-fix data; corrected during the paper's number
audit). Reconciliation: **the DT is
retrained every iteration** — it changes because its *weights* are re-fit, not
because it conditions. **MF-DRO appears to re-fit rather than condition**,
behaving as a per-iteration acquisition function parameterised by a transformer.

### The policy is a fixed acquisition rule to within 0.13% (H23)

Decomposing `w(s) = w̄ + δ(s)` over the 10 distinct τ=0 states:

| | |
|---|---|
| `w̄` alone reproduces the full argmax | **12/12 pools** |
| median (top-1/top-2 margin) / (max δ contribution) | **77.16** |
| `‖δ‖ / ‖w̄‖` | **0.00129 — 0.13%** |
| `bias_head` changes the argmax | 0/12 (it adds a per-candidate constant, so it *cannot*) |

**MF-DRO's learned policy is, to within a tenth of a percent, a fixed linear
acquisition function.** Not a weakly-conditioning policy — a constant rule with
a negligible state-dependent perturbation on top. This was asserted from
inference in H22 and then measured directly, because three earlier inferences in
this project failed re-measurement.

### And the fixed rule is uncertainty-AVERSE (H24)

Signed `w̄`: `mu_H` **+1.0824**, `mu_L` **+0.8254**, **`sigma_H` −0.5487**,
`dist_inc` −0.2976.

**The weight on HF posterior uncertainty is negative** — the rule penalises
exactly what UCB/EI/MES reward. Confirmed independently: agreement with
`mu_H + β·sigma_H` falls monotonically as β grows (66.7% → 50.0% → 41.7% →
25.0% for β = 1, 2, 3, 5). We do *not* name the rule — the best match is 75.0%
with a two-way tie on 12 pools, too weak for an identity claim.

This is the causal link to the performance result: penalising HF uncertainty
concentrates queries where the model is already confident → little information →
slow incumbent improvement, the pathology this investigation began from.
**Trained on an information-seeking MF-MES teacher, the student inverted the sign
of its teacher's defining term.**

### The mechanism, measured

`coef_head` emits the **same coefficient vector for every state inference
encounters**: pairwise cosine across the 10 genuinely distinct τ=0 states is
≥ **0.99999224**, `‖w‖` varies by 1.001×, and `sv₁/Σsv = 0.997691`
(singular values [5.053, 0.0063, 0.0023, 0.0015] — ten near-identical rows).
The ranking is state-invariant *by construction of the learned head*.

**And it fails across real iterations too, where the state genuinely does
change.** Capturing all 12 real-iteration states from an actual run (mean
pairwise state L2 = **1.4968**, versus ~0 across ensemble members): the head does
respond more — cosine 0.99937, a **2.04° rotation**, ‖w‖ ratio 1.023× — but the
**argmax moves 0/12**, with 1.00 distinct argmaxes per pool. A full run's worth
of state change is far too small a stimulus to move a decision over 200
candidates. The state channel is non-functional at decision level *both within
and across* iterations.

**The two stages have different causes (H21).** Against a randomly initialised
network with identical architecture and states: the encoder's contraction is
*architectural* (random 0.4601× vs trained 0.3898×, ratio 1.18), while the
head's is *learned* — a random head **amplifies** at 1.6039× where the trained
head contracts at 0.3348×, a **4.8× swing** from fitting alone. **And no threshold exists within 100× (H22).** Scaling state deviations
`s' = s̄ + λ(s−s̄)` up to λ=100 with the trained net fixed never moves the argmax
on >1/12 pools, while the coefficient spread grows **76×** and `w` rotates
**11.5°**. The map is invariant to state *direction* across two orders of
magnitude of gain: the ranking is dominated by a state-independent component of
`w` whose margin exceeds what an 11.5° rotation can overturn. (Large λ is OOD —
this characterises the map, it is not a proposed fix.)

And the head is near-constant partly because the within-iteration input is
degenerate: real τ=0 states
vary **2.2× less** than fantasy states within a single rollout
(`ref_block_std` 0.0076 vs 0.0169), with only **10 unique τ=0 states per
200-trajectory batch** — one per ensemble member, all views of the same real
data. At any real iteration the policy is handed essentially one state; there is
nothing to condition *on*.

This is not an architecture refusing to condition. It is an architecture given a
degenerate conditioning input — which is why all ten conditioning-side
interventions were null, and why the only working adaptation channel is
re-fitting (H6/H7: ~18% of decisions changed, ~0 regret bought).

This explains every failed conditioning-side intervention:

- **H4** (AdaLN-Zero, DDT mechanism, arXiv:2601.15953): **REFUTED** — 0% of
  sweeps moved the argmax vs 25% for token conditioning.
- **H5** (deny the score head its GP features): **REFUTED** — manipulation worked
  (argmax(mu_H) agreement 66.7% -> 0.0%) and the head *still* ignored `h` (0/12).
- **H6** (freeze the DT after iter 5): **inconclusive** (below).

### The best remaining lead

`rtg_target` is **structurally clamped to [0.5, 1.0]** (CV 0.232) by
`max(batch_max, 0.5*running_max)` on a batch-max-normalised rtg. Two claims stay
confounded: *the network ignores RTG* vs *RTG carries too little variation to be
worth using*. H8 shows no response within the given support; it cannot show what
a genuinely varying signal would do. Separating them needs a schema change
(`alpha_rtg`, or an un-normalised quantity) plus a retrain.

---

## Two freeze mechanisms were being conflated

**MF-GP-UCB's 100% freeze is definitional.** Under the frozen protocol:
`mean_n_HF = 0.0`, `mean_n_improved = 0.00` on **all 10 seeds**. It never queries
HF once, so the HF incumbent cannot move by construction — the documented
cost-ratio degeneration to all-LF at `c_H=8` (`mf_baselines.py:206-213`). The old
pooled "100% freeze" column mixed this with the DRO family's leakage freeze.

---

## What continued DT training does: changes ~18% of decisions, buys ~nothing

Two experiments, deliberately different instruments, now compose into one
account.

**H6 — regret comparison, pre-registered n=30 (primary AND final).**
Freeze the DT after iteration 5 vs retrain throughout:

    FROZEN 0.5898 (sd 0.230)   LIVE 0.5146 (sd 0.164)
    paired diff +0.0752   95% CI [-0.0105, +0.1608]   Wilcoxon p = 0.0795
    FROZEN better on 9/30

Locked prediction 1 (CI contains zero) **MET** — registered as a *null* in
advance so a null could not later be dressed up as a finding. Prediction 2
(Levene p<0.05 on variance) **NOT MET** (p = 0.196); the variance story does not
survive either.

The estimate's trajectory is itself the cautionary result: n=1 -0.208, n=5
-0.103, n=7 -0.010, n=10 +0.098, n=30 +0.075. Every intermediate reading sounded
publishable and every one differed. Post-hoc power: paired sd 0.239, so ~80 seeds
would be needed for the observed effect.

**H7 — decision comparison, 377 paired decisions over 5 seeds.** Same question,
better instrument: snapshot the DT at iteration 5 and replay every later
iteration's *identical* state/RTG/BTG/candidate pool through it.

| pooled | argmax agreement | mean L2 dist |
|---|---|---|
| 377 decisions | **0.817** | 0.121 |

Prediction 1 (agreement > 0.70) **PASS**. Prediction 2 (no growth with t)
**FAIL** — divergence grows, agreement 0.860 -> 0.798 between run halves. That
failure is useful twice: it corrects the claim, and it proves the instrument is
*sensitive* rather than blind.

**Instrument verified before interpretation**: snapshot independent (0/93 params
share storage), live model genuinely diverged (77/93 params changed, `coef_head`
by ~4.6e-03). A 4-record smoke had shown `dist = 0.000000` and I nearly reported
"bit-identical" proposals; at scale that is **wrong**.

### The reconciliation

> Continued training changes **~18% of decisions**, and those changes are worth
> **~nothing in regret** (+0.075, CI containing zero).

This is the opposite of what I was drifting toward at n=7 ("continued training
contributes ~nothing"). Training *does* change what the policy does; the changes
simply do not buy performance. Combined with H5 and H8:

- the DT's **conditioning** pathway is inert (state, RTG, BTG never move the argmax)
- its **weights** are not — retraining moves ~18% of decisions
- and those weight-driven changes are close to regret-neutral

"MF-DRO re-fits rather than conditions" survives, with the refinement that the
re-fitting does real work on decisions — just not *useful* work here.

## H9: the obvious fix for the RTG confound does not exist

`alpha_rtg` was recorded as the top remaining lead for separating *the network
ignores RTG* from *RTG carries too little variation*. **It is a no-op.**
`max(batch_max, alpha_rtg*running_max)` with batch-max-normalised rtg means
`running_max -> 1`, so the floor binds only when `batch_max < alpha_rtg`:
measured over 790 iterations, 46.3% of the time at 0.5 and **0.0%** at 0.1.
Both arms produced the identical band [0.5672, 1.0000].

The experiment **voided itself** on its own pre-registered manipulation check.
The real lever is the **batch-max normalisation**, which maps every batch's best
trajectory to ~1 regardless of quality — and which my own RTG-cap fix
introduced while correctly removing a different pathology. Fixing one RTG defect
manufactured another.


## RESOLVED: the RTG confound was a starved reward, not an inert network

Three separate experiments (H9 floor, H10 normalisation, H11 arm C DT-style
decrement) each voided on their own manipulation check. They share one cause.

**Under `rollout_reward="improvement"` an LF step earns exactly `0.0`**
(`mf_dro.py:1290+`). Measured on a 200-trajectory batch: **63.0%** of
trajectories have `rtg[0] == 0`, only **23.7%** of steps carry any nonzero
reward, and `Spearman(n_HF, rtg[0]) = +0.355` (p<1e-6) — the return partly just
counts HF queries. **A signal that is identically zero cannot be made to vary by
rescaling it**, which is all H9 and H10 did.

Two structural facts, both provable without compute:

1. `target = max(batch_max, alpha*running_max)` with `batch_max <= running_max`
   caps the band ratio at `1/alpha` — 2x at `alpha=0.5`, always. H10 measured
   1.76x and 2.59x, i.e. at the ceiling. Two lines of algebra would have
   prevented both experiments.
2. `mes_entropy`'s RTG is `log(b_tau) - log(b_T)` with `b_T` drawn from the
   **fully-conditioned** end-of-rollout model. Since `H(Gumbel) = ln b + gamma + 1`,
   that equals `H[y*|D_0] - H[y*|D_T] = I(y*; y_1..y_T)` — the **joint set-level
   information gain**, not a per-step cumsum.

### Reward signal health (200 trajectories, identical seed)

| reward | dead `rtg[0]==0` | LF credit | mean / CV | negative |
|---|---|---|---|---|
| `improvement` (current default) | 63.0% | 0.0% | 0.1022 / 1.964 | 0.0% |
| `kg_incumbent` topk=1 clipped | 4.5% | 27.2% | 0.1396 / 1.374 | 0.0% |
| `kg` topk=1 signed | 0.0% | 100% | −0.0204 / −12.93 | 67.0% |
| `kg` topk=5 signed | 0.0% | 100% | 0.0676 / 3.677 | 38.5% |
| **`mes_entropy` (JOINT)** | **0.0%** | **100%** | **0.2903 / 0.661** | **5.5%** |

The joint-MES mode dominates every variant I built. LF credit is automatic:
an LF observation shrinks the y* distribution through `rho`, so its discount is
derived from the fitted KO model rather than hand-specified.

### Resolved: joint MES is the better reward (H16, 10 seeds, pre-registered)

| axis | `improvement` | `mes_entropy` | paired | Wilcoxon |
|---|---|---|---|---|
| M3 `f_hf(x_0)` (original axis) | +0.1250 | +0.1648 | +0.0398, 7/10 | p=0.193 |
| **M1 best HF point (PRIMARY)** | +0.0863 | **+0.1826** | **+0.0962, 8/10** | **p=0.0195** |
| M1b best any point | **−0.0246** | +0.0706 | +0.0952, 9/10 | p=0.0645 |

`mes_entropy` leads on **all three** axes, including the one originally used to
reject it. `improvement`'s return is *negatively* correlated with the best point
its own trajectory visited — worse than uninformative as a conditioning target.
Effect is modest (would not survive a strict 3-axis Bonferroni, though M1 was the
pre-registered primary). Two independent lines — signal health (decisive) and
teacher quality on a fair axis (modest) — agree.

**The gate that abandoned it does not reproduce.** It was
within-group Spearman(`rtg[0]`, true `f_hf(x_0)`) — i.e. *step-0 greediness*,
an axis `improvement` satisfies by construction and an information reward is
designed not to. H15 re-ran it on current code and the ordering **reverses**
(`improvement` +0.052 p=0.544; `mes_entropy` +0.173 p=0.047). Per H15's own
protocol a reproduction failure makes **both** conclusions suspect, so no winner
is declared — but `improvement`'s selection justification is gone, and it is the
**current default** (`dro_runner.py:439`) under which the frozen headline was
produced. The headline number stands; the reason for that configuration does not.


## The theory that predicts all of it: RCSL's necessary conditions fail here

MF-DRO **is** return-conditioned supervised learning (RCSL). Brandfonbrener et
al. (NeurIPS 2022, arXiv:2206.01079) give RCSL's necessary conditions for
optimality, and our setting violates them by construction. See
`literature/rcsl-necessary-conditions.md`.

**Their Corollary 1** requires dynamics **ε-close to deterministic**:
`J(π*) − J(π_f^RCSL) ≤ ε(1/α_f + 3)H²`. Our rollout transition is
`y_τ = sample_fantasy(x_τ, ·)` — a **draw from the GP posterior**
(`mf_dro.py:1264`). It is Gaussian. The stochasticity *is* the method.

**Their Figure 1c** is the decisive one. They construct a case where

> "the bias of RCSL in stochastic environments can remain **regardless of the
> conditioning function**" — "merely changing the conditioning function is not
> enough to overcome the bias."

Every conditioning-side intervention we ran is an attempt to change the
conditioning function, and every one was null or void:

| ours | changed | outcome |
|---|---|---|
| H4 | AdaLN-Zero (DDT) | REFUTED |
| H5 | deny score head its GP features | REFUTED |
| H8 | sweep RTG in its realised band | 0/12 |
| H9 | `alpha_rtg` floor | VOID |
| H10 | un-normalised RTG | VOID |
| H11 | real history + DT-style decrement | A/B null, C VOID |
| H12–H16 | the reward quantity itself | signal fixed; decision not yet moved |

**Their return-coverage condition `P_β(g=f(s)|s) ≥ α_f` is our starvation
finding in the theory's own language.** 63.0% of trajectories carried
`rtg[0]=0` against targets in [0.57, 1.0], so `α_f` is small and the bound
scales as `C_f/α_f`. And Corollary 2's exact-optimality requirement
`f(s₁)=V*(s₁)` is hopeless for us — our target is a heuristic whose band is
provably capped at `1/α`.

### Provenance warning on the h-insensitivity result

The original H5 probe drew its comparison state as `batch[(p+7) % len(batch)]`
for `p=0..11`, i.e. indices 7..18 — and the batch is built
`for ko in ensemble: for _ in range(rollouts_per_model=20)`, so **indices 0..19
all share a bit-identical τ=0 state** (only 10 unique τ=0 states per
200-trajectory batch). Verified: 12/12 of H5's comparisons were identical to
`batch[0]`; 0/9 across model blocks are. **H5 swapped a state for itself.**

Re-measured with genuinely distinct states: **argmax moved 0/12, score-vector
correlation 1.000000**. The conclusion survives and is now properly evidenced —
but it rested on an invalid probe until this audit, and anything built on it
before then was right by luck. See `experiments/h5-score-head-bottleneck/AUDIT.md`.

Also refuted in passing: **feature-scale domination**. `mu_H`'s across-candidate
sd is 1.0× the median feature's and carries 31.8% of the ranking spread, and
`argmax(score)=22` vs `argmax(mu_H)=152` — so the old "score tracks `mu_H` on
67–75% of pools" does **not** reproduce post-fix.

**But the theory is NOT the whole explanation — H19 checked.** With
`fantasy_mode="mean"` the transition is deterministic (verified: repeat-difference
`0.000e+00`) and diversity is intact (200/200 distinct trajectories). That is the
regime RCSL theory says should work. **The argmax still moves 0/12.**

So near-determinism is *not* the binding constraint in MF-DRO. The honest,
narrower claim: **the failure is architectural — the score head barely reads `h`
(H5) — and RCSL theory explains why the many conditioning-side remedies could
not have rescued it.** Nothing that reaches `h` can move a decision that is not
a function of `h`.

## Methodological lessons (transferable)

1. **Measure the mechanism, not the downstream metric.** Every near-deterministic
   probe settled its question immediately; the one variance-dominated average
   burned hours, changed sign four times, settled nothing.
2. **Correlate a diagnostic against the outcome before calling it an
   explanation.** The secondary-basin story is true as description (7/10 seeds
   sit closer to x2) but corr(d(x*), regret) = +0.09 — regret depends on the
   *best point ever evaluated*, not the average query location.
3. **Check the realised support before choosing sweep bounds.** My RTG sweeps
   spanned 100x against a 2x band; the finding survived (H8) but nearly rested on
   OOD probing.
4. **Report intermediate n as direction, never as result.**
5. **Prefer robust tests on skewed data** — F/Bartlett vs Levene disagreed and it
   changed the conclusion.
6. **Completion order in a cost-budgeted grid is biased** toward HF-heavy runs.
7. "Regret moves only on HF queries *within* a run" does **not** imply "more HF
   *across* runs -> better regret".
8. **Derive the reachable range of a formula you control before experimenting
   on it.** Two experiments died against a cap provable in two lines.
9. **Put the manipulation check first, as a standalone gate, and commit to
   stopping on failure.** H12 did and stopped cleanly; H9/H10/H11 each spent a
   full experiment before discovering the manipulation never happened.
10. **A null-guard must be conditioned on the manipulation check passing.**
11. **A discriminating metric must not be able to saturate** in the arms
   compared — H13's decomposition rule hit 100% in both arms and said nothing.
12. **Spec the codebase before building the fix.** H12/H13 reinvented, worse, a
   joint-information-gain reward that already existed.
13. **The makespan of a mixed grid is the duration of its slowest job CLASS,
   not the total wall-time divided by the job count.** h1 ran 30 jobs in 6979 s
   (116 min) with 15 workers — but 20 of those were fast baselines, so the
   makespan *was* the MF-DRO job duration (~116 min). I estimated H17's 10
   MF-DRO jobs at 50–90 min by implicitly amortising h1's makespan over all 30
   jobs. The same 10 MF-DRO jobs take the same ~116 min however many cheap jobs
   run beside them. Compounding it: `mes_entropy` is LF-heavier, so more
   iterations are needed to burn a fixed cost budget, pushing the true figure to
   ~145 min. **Second ETA miss on this class of estimate; this is the fix.**
14. **A script's auto-verdict must key on the DECISIVE measurement, not a
   proxy that correlates with it.** Four wrong auto-verdicts this phase — H11
   (fired without requiring the manipulation check to pass), H13 (a metric that
   saturated at 100% in both compared arms), H19 (a signature omitting the
   variable of interest), and the `coef_head` probe (cosine threshold instead of
   argmax movement). Each proxy looked reasonable when written and printed a
   confident, wrong sentence when run.
15. **When a probe compares "two different X", assert inside the probe that
   they are different.** H5 would have failed a one-line assertion for months.
   Three instrument defects this phase (H13's dead-signal metric, H19's
   diversity signature, H5's state swap) — all found by checking the instrument
   against the data rather than trusting it.
16. **Do not fix a scale-free quantity with an absolute threshold** (bit twice:
   `bes_delta`, soft-target temperature).

---

## Open questions

- ~~Is RTG inert, or merely starved?~~ **RESOLVED: starved.** See above.
- **Is the teacher-quality gate regret-anchored?** If so it disqualified the
  joint-MES reward unfairly and must be re-measured. Blocks the reward switch.
- **Does continued DT training matter?** H6 underpowered; H7 (decision agreement
  between the live policy and an iteration-5 snapshot — ~50-200 paired decisions
  per run instead of one regret scalar) is implemented and queued.
- Is the gap to MI-Greedy closable within the DRO frame? On current evidence
  MF-DRO reaches parity in the mean while being more reliable and less capable at
  its peak — but underpowered.

## Confounding to respect

The `7bcc3b8` fixes landed **together**; residual conditioning-insensitivity
cannot be attributed to any single one. Ablation flags exist for exactly this:
`use_candidate_scoring`, `use_candidate_features`, `soft_targets`,
`use_state_standardization`, `use_teacher_pool`, `use_linear_score_head`,
`fidelity_sampling`, `rollout_reward`, `rtg_conditioning`, `freeze_dt_after`,
`decision_snapshot_at`.

**Correction:** `use_linear_score_head` was **dead** until H20 found it.
`dt_cfg` — the `SimpleNamespace` handed to `DecisionTransformer` — never
forwarded it, so the DT's `getattr(config, 'use_linear_score_head', True)`
always defaulted to `True` and the `False` branch was unreachable. Found by an
`assert` inside H20's probe, not by reading the code. Default unchanged, so no
recorded result moves. An audit of every other config attribute the DT reads
(`cand_feature_dim`, `rtg_conditioning`, `score_temp`) confirms those **are**
forwarded; this was the only dead one.
