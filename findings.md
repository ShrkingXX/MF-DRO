# Findings — MF-DRO incumbent-freeze on Hartmann 6D

> **⚠ ALL NUMBERS BELOW DATED BEFORE COMMIT `7bcc3b8` ARE SUSPECT.**
> A target-leakage bug in the Decision Transformer readout was found and fixed
> after the 1280-run inventory was built. Every DRO-family result in
> `data/results_inventory.csv` was produced with that bug active. Re-measure
> before comparing anything to a post-fix number.

## Current Understanding

### The DRO-family freeze has a concrete, identified mechanism: target leakage

The MF sequence is 4 tokens per step, `[rtg, btg, state, action]`, with the
action token at index `4t+3`. The causal mask is `triu(diagonal=1)`, which
lets a position attend to **itself**. The readout was `h[:, 3::4]` — the
*action* token, whose `a_emb` embeds the **true** `actions_x`/`actions_ell`.

So the location head, the fidelity head and the score head were each
**predicting from their own target**. At inference, `propose_mf` has no action
to embed (it is what we are computing) and necessarily zeroes that slot — a
total train/inference distribution shift on the model's single most predictive
input. That is sufficient on its own to produce a policy that ignores its
conditioning and emits a near-fixed query.

Verified by ablation: after training, zeroing `a_emb` cost the fidelity head
**15.6%** of its loss pre-fix versus **0.7%** post-fix; `L_loc` dependence went
to ~0.

**This mechanism covers the whole DRO family, not just MF-DRO.** The SF path
(`get_action_hidden_states`) carried the *identical* bug (`[:, 2::3]` — also
the action token, in the 3-token `[r, s, a]` layout). This matters because
SF-DRO freezes 12/12 and has **no fidelity head**, so the fidelity-threshold
bug could never have explained it. One mechanism, both methods.

Fixed in `7bcc3b8`: readout moved to the **state** token on both paths, and the
causal mask added to `propose_mf` (inference had been *bidirectional* while
training was causal — a second, silent train/inference mismatch).

### MF-GP-UCB's freeze is a *different*, already-documented mechanism

Previously an open question ("why does MF-GP-UCB also freeze 100% while
MI-Greedy never does?"). It has no DT, so the leak cannot apply. The cause is
documented in `src/baselines/mf_baselines.py:206-213`: on Hartmann 6D
(`c_H=8`), cost-normalized UCB degenerates to **all-LF selection** once the
cost ratio exceeds the fidelity GPs' prior variance ratio. HF is never queried,
so the HF incumbent — and therefore regret — cannot move. Expected behavior of
the baseline, not a shared pathology.

So the "100% freeze" column pooled **two unrelated causes**: target leakage
(DRO family) and cost-ratio degeneration (MF-GP-UCB).

### Freeze rate table (PRE-FIX — retained for reference only)

| Method | Frozen | Median distinct regret values |
|---|---|---|
| SF-DRO | 12/12 (100%) | 1 |
| MF-GP-UCB | 10/10 (100%) | 1 |
| MF-DRO | 9/12 (75%) | 1 |
| MF-MI-Greedy | 0/10 (0%) | 6 |
| Greedy-MES | 0/12 (0%) | 4 |
| KO-MES / Additive-MES / SF-MES | 0/5 each | 5–9 |

## Patterns and Insights

1. **H0 (init coverage) stays REFUTED.** MF-MI-Greedy freezes 0/10 on the same
   initial design. Unaffected by the leak finding — MI-Greedy has no DT.
2. **H1 (freeze lives in the DRO/DT component) is now SUPPORTED with a
   mechanism**, not just a correlation. The leak is *in* the DT, and is present
   on both the SF and MF paths, matching the SF-DRO ≥ MF-DRO freeze ordering.
3. **Several independent defects compounded the leak** (all fixed in `7bcc3b8`):
   - τ=0 state collapse: all ~70 rollouts in a batch shared a bit-identical
     τ=0 state with different targets → MSE's minimizer is the conditional
     mean. Residual 10/70 (one per ensemble member) is expected and correct.
   - BTG target pinned at exactly the cost floor (`2·c_H + 6·c_L = 22.0`),
     permanently conditioning the policy on "be cheapest" → LF bias.
   - RTG cap: per-trajectory self-normalization forced `rtg[0]==1.0` for every
     improving trajectory — **2 distinct values across 200 trajectories**.
   - Soft teacher target had entropy **exactly 100% of log K** — uniform, so
     the KL term was learning nothing at all.
   - Candidate features leaked the label a second way: the two MES columns
     reconstructed the teacher at corr ≈ 0.80. Removed from candidate
     features; deliberately kept in the state's ref-grid block.
4. **Reward: `mes_entropy` carried almost no action-relevant signal.**
   Within-group Spearman(rtg[0], true f_hf(x₀)): mean **+0.129** with **5/10
   groups negative** (p ≈ 0.32). Switching to regret-based `"improvement"`:
   mean **+0.191**, SE 0.0725, **z = 2.63, p = 0.0085**, Wilcoxon p = 0.0039,
   9/10 groups positive. The reward change is what moved this.
5. **Regret is controlled by HF-query count, not cost.** Across 6 runs, every
   regret drop coincides exactly with an increment in cumulative HF queries;
   regret is flat regardless of accumulated LF cost. Report both axes.
6. **The freeze was partly a mislabel: the policy was converging on the wrong
   basin.** Queries sat closer to Hartmann-6D's *second* optimum
   x₂ = [0.405, 0.882, 0.846, 0.574, 0.139, 0.038] (f = 3.2031, ‖x₂−x*‖ = 1.103)
   than to x* — 0.62–0.70 vs 0.97–1.10 on all 3 seeds. A flat
   `query_dist_to_xstar` cannot distinguish "not converging" from "converging
   elsewhere"; `query_dist_to_x2_per_iter` now separates them.

## Lessons and Constraints

- **Pre-`7bcc3b8` DRO numbers are not comparable to post-fix numbers.** This
  includes the 1.31 MF-DRO regret and every freeze rate above.
- Iteration counts differ wildly across prior runs (MI-Greedy median 53,
  MF-DRO 100, MF-GP-UCB 800). Prior cross-method comparisons are **not**
  obviously cost-matched. Re-verify matched real cost in any comparison.
- `REVISION_LOG.md`'s Hartmann init claim is verified wrong (6.2% not 12%; 86%
  gate-failure rate; seed=42 at the 30th percentile). Do not build on it.
- **Do not fix a scale-free quantity with an absolute threshold.** Bit twice:
  `bes_delta` (documented in-code) and the soft-target temperature. Standardize
  within the comparison set instead of tuning `score_temp`.
- **`rollouts_per_model` 7 → 20 roughly tripled per-iteration wall time.**
  Budget compute accordingly; measure before scaling a sweep.
- `minimum_hf_fraction`'s τ=0 override was tried and **reverted** — it drove the
  τ=0 HF label rate to a constant 1.0 in the all-LF regime, i.e. a constant
  label at the one position inference actually uses.

## Open Questions

- **Does the leak fix actually change end-to-end regret?** No post-fix
  multi-seed run under the frozen protocol exists yet. This is the binding
  question — `experiments/h1-leak-fix-validation/`.
- **RTG still does not move the decision.** Sweeping the RTG target 0.1×–10×
  leaves the argmax bit-stable in every configuration tested; score vectors do
  shift (corr 0.986–0.9999, never bit-identical), so the pathway is connected
  but far too weak. Ruled out: degenerate reward embedding. Also measured: the
  score head is near-indifferent to *which* `h` it receives (batch-mean or
  shuffled `h` changes the argmax only ~8% of the time), and state perturbation
  at 1× batch std leaves the argmax unchanged.
- **Is the score head still close to a fixed acquisition function?**
  argmax(score) matched argmax(mu_H) on 7/10 resampled candidate pools (down
  from 100%, so partial independence exists), corr(score, mu_H) = 0.944 — yet a
  state-conditioned linear head shows `w` genuinely varies. Reconcile.
- **The fidelity head drifts toward LF during training** (`fid_mean_per_iter`
  declines over every run), and median `p_pred` sits below 0.5 even under
  Bernoulli sampling. Unexplained.
- Is the ~4× gap to MI-Greedy closable within the DRO frame at all?

## Confounding to respect

Items 3–6 above were changed *together* before the RTG/score-head measurements
were taken, so the residual RTG-insensitivity cannot currently be attributed to
any single change. Ablate deliberately — the flags exist for exactly this:
`use_candidate_scoring`, `use_candidate_features`, `soft_targets`,
`use_state_standardization`, `use_teacher_pool`, `use_linear_score_head`,
`fidelity_sampling`, `rollout_reward`.
