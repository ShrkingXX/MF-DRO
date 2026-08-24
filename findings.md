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

## H6: an underpowered experiment, reported as such

The paired estimate **changed sign** as seeds accumulated: n=1 -0.208, n=5
-0.103, n=7 -0.010, n=9 +0.062, n=10 **+0.098**. Final n=10: 95% CI
**[-0.097, +0.292]**, Wilcoxon p = 0.322.

Post-hoc power: paired sd **0.311**, *larger than the effect being chased*.
~80 seeds needed for +0.098; the design resolves only effects >= ~0.3 against
arms near 0.5. **H6 cannot answer its own question.** A pre-registered extension
to n=30 is running with anti-optional-stopping guards (n fixed in advance,
primary *and* final, primary prediction is a null).

Variance tests disagreed (F 0.029, Bartlett 0.029, **Levene 0.209**); regret is
right-skewed so the robust test governs — the variance claim does not hold either.

---

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
