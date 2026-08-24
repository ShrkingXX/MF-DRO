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
| Frozen success test | **FAIL** (MF-DRO mean+SE 0.5442 >= MI-Greedy mean-SE 0.3825) |
| Freeze pathology | **RESOLVED** — 0/10 frozen, vs 9/12 (75%) pre-fix |
| Mean regret | 1.31 (pre-fix, non-comparable) -> **0.5047 ± 0.0395** |

At matched cost ~200: MF-DRO **0.5047 ± 0.0395**, MI-Greedy 0.5091 ± 0.1266,
GP-UCB 1.7934 ± 0.1223. `PROTOCOL.md` explicitly permits "no within-frame fix
closes the gap"; that is the honest headline.

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
- **Reward**: `mes_entropy` Spearman +0.129 (5/10 groups negative, p~0.32) ->
  regret-based `improvement` **+0.191, z=2.63, p=0.0085**, 9/10 positive.

---

## What the DT contributes: apparently very little

| probe | result |
|---|---|
| swap `h` for another state's hidden vector | argmax unchanged **12/12** |
| coords-only features (ranking *must* use `h`) | argmax unchanged **12/12** |
| batch-mean / shuffled `h` | argmax changed ~8% |
| state perturbation at 1x batch sd | argmax unchanged, corr 0.9997 |
| **RTG swept across its realised band [0.5,1.0]** | argmax moved **0/12**, corr 0.99993 |

Yet queries do move (`x_t_trace` sd 0.166-0.213). Reconciliation: **the DT is
retrained every iteration** — it changes because its *weights* are re-fit, not
because it conditions. **MF-DRO appears to re-fit rather than condition**,
behaving as a per-iteration acquisition function parameterised by a transformer.

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
8. **Do not fix a scale-free quantity with an absolute threshold** (bit twice:
   `bes_delta`, soft-target temperature).

---

## Open questions

- **Is RTG inert, or merely starved?** The [0.5,1.0] clamp confounds these. Top
  lead; needs a schema change plus retrain.
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
