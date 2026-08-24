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

## Interim results — h1-leak-fix-validation (grid still running)

**Baselines complete (10/10 each), at genuinely matched cost = 200:**

| method | mean regret | range | realized cost |
|---|---|---|---|
| MF-MI-Greedy | **0.509** | 0.188 – 1.540 | 207.2 |
| MF-GP-UCB | 1.933 | 1.471 – 2.375 | 200.0 |

The inherited MI-Greedy number this project calibrated against was **0.279**.
At matched cost it is **0.509**, with large seed spread — confirming the
standing suspicion that prior cross-method numbers were never cost-matched
(MI-Greedy median 53 iters vs MF-DRO 100 vs MF-GP-UCB 800). This roughly
**halves the gap** MF-DRO has to close. Minor asymmetry to flag: MI-Greedy
overshoots the budget ~3.6% because it checks at round start and a round costs
up to 2·c_H.

**The fidelity collapse is gone.** In the running MF-DRO jobs the realized
fidelity mix is **39.4% HF** (61 HF vs 94 LF observed in flight), against a
pre-fix `lf_fraction` of ~0.98, i.e. **~2% HF**. Since regret was established to
move *only* on HF queries and never on accumulated LF cost, this is the single
most plausible route by which the fixes could change end-to-end regret — and it
is a direct, mechanical consequence of the leak + BTG-cost-floor + fidelity
fixes rather than a tuning artifact. Final regret is not yet in; do not
pre-judge it.

## The central conceptual finding: MF-DRO re-fits, it does not condition

Two literature-grounded interventions on the conditioning pathway were locked,
run, and **both refuted**:

- **H4** (AdaLN-Zero conditioning, the DDT mechanism): argmax moved on 0% of RTG
  sweeps vs 25% for token conditioning. Made it *worse*.
- **H5** (deny the score head its GP features so any ranking must flow through
  `h`): the manipulation worked perfectly — `argmax(mu_H)` agreement went
  66.7% -> 0.0% — and the head **still** ignored `h` entirely (0/12), while RTG
  movement *fell* (16.7% -> 8.3%).

Five independent measurements now agree that, **within a single trained model,
the proposal is very nearly independent of the conditioning**:

| probe | result |
|---|---|
| swap `h` for a different state's hidden vector | argmax unchanged **12/12** |
| coords-only features (ranking *must* use `h`) | argmax unchanged **12/12** |
| batch-mean / shuffled `h` | argmax changed only ~8% |
| state perturbation at 1x batch std | argmax unchanged, score corr 0.9997 |
| RTG sweep 0.1x-10x | argmax essentially pinned in every config |

**But queries do move** (`x_t_trace` std 0.166-0.213), so the honest
reconciliation is:

> The DT is **retrained every BO iteration**. Within an iteration it is close to
> a fixed function of the candidate set; across iterations it changes because
> its *weights* were re-fit on fresh rollouts.

So MF-DRO's apparent adaptivity comes from **re-fitting, not conditioning**. It
behaves as a per-iteration acquisition function that happens to be parameterised
by a transformer, rather than as a return-conditioned policy. This single claim
explains the entire failed sequence of conditioning-side interventions: they
were all strengthening a pathway that is not being used.

**Caveat kept attached:** these probes train 10 epochs on one rollout batch.
That is the *production* setting (`num_epochs=10`), so it is the operationally
relevant regime — but it is not a claim about this architecture at large
training scale.

**This escalates the question to the frame itself.** PROTOCOL.md asks whether a
fix exists *within the DRO frame*. If the learned policy contributes nothing
beyond re-fitting, the frame is what is in question — which is why H6 (freeze
the DT after k=5 and see whether anything changes) is the next experiment
rather than another component patch.

## Variance, not just mean: MF-DRO trades peak performance for reliability

Partial h1 (8/10 MF-DRO seeds in; baselines complete):

| method | mean | sd | SE | range |
|---|---|---|---|---|
| MF-DRO | 0.5011 | **0.0887** | 0.0313 | [0.407, 0.650] |
| MF-MI-Greedy | 0.5091 | **0.4004** | 0.1266 | [0.188, 1.540] |
| MF-GP-UCB | 1.7934 | 0.3868 | 0.1223 | [1.182, 2.375] |

The **means are essentially tied** (0.501 vs 0.509), but MI-Greedy's standard
deviation is **4.5x larger**. MI-Greedy sometimes reaches 0.188 — far better
than MF-DRO ever does — and sometimes lands at 1.540, far worse. MF-DRO lands in
[0.407, 0.650] every time.

That is a substantive difference the mean alone hides: post-fix MF-DRO is not
"as good as" MI-Greedy, it is **differently shaped** — worse peak, much better
worst case, far more predictable.

### An honest limitation of the pre-registered success test

PROTOCOL.md's frozen criterion is `MF-DRO mean+SE < best-baseline mean-SE`.
Because MI-Greedy's SE is ~4x MF-DRO's, `mean-SE` = 0.3825 is a *low bar set by
the baseline's own instability*. A method with an identical mean and one-quarter
the variance is structurally penalised by this test.

The test is **frozen and will be reported exactly as specified** — this is an
observation about what it measures, not grounds for changing it, and
`PROTOCOL.md` explicitly anticipates "no within-frame fix closes the gap" as a
valid outcome. But any write-up should report the variance alongside the test
result, because "failed the criterion while being 4.5x more consistent" is a
materially different claim from "failed the criterion".

### The freeze itself is resolved

**0/8 completed MF-DRO seeds are frozen** (`n_improved == 0`), against a pre-fix
freeze rate of 9/12 (75%). Incumbent-improvement counts run 1-5 per seed. The
original pathology this whole investigation was named after is gone.

Note also that `n_improved` and final regret are only loosely coupled: seed51
improved once and reached 0.407; seed46 improved five times and reached 0.441.
"Freeze" and "regret" are separate axes and should be reported separately.

## H6 (n=9/10): the experiment is UNDERPOWERED — and that is the finding

The paired estimate did not converge. It **changed sign**:

| n | mean paired diff (FROZEN - LIVE) | reading at the time |
|---|---|---|
| 1 | -0.2084 | "freezing helps 40%" |
| 5 | -0.1033 | outside 1 SE, prediction refuted |
| 7 | -0.0102 | within 1 SE, prediction supported |
| **9** | **+0.0623** | outside 1 SE, *opposite* sign |

At n=9: FROZEN 0.5577 vs LIVE 0.4953, Wilcoxon **p = 0.4961**, FROZEN better on
only 3/9.

### The power calculation that should have been done first

    sd of paired differences = 0.311
    SE (n=9)                 = 0.104
    95% CI on the difference = [-0.141, +0.265]

    n needed at 80% power, alpha=.05:
      to detect 0.05 regret units  ->  ~303 seeds
      to detect 0.10               ->   ~76 seeds
      to detect 0.20               ->   ~19 seeds
      to detect 0.30               ->    ~8 seeds

**The sd of the paired differences (0.311) is larger than the entire effect we
are trying to detect.** With 10 seeds this design can only resolve effects of
roughly 0.3 regret units or bigger — and both arms sit around 0.5 total. H6 as
specified cannot answer its own question, and neither the "freezing helps"
reading nor the "freezing is neutral" reading was ever supported.

The 95% CI **[-0.141, +0.265] straddles zero**, so the honest statement is:
*we cannot distinguish freezing the DT from retraining it at this sample size.*
That is weaker than the n=7 conclusion I recorded last tick, and it supersedes it.

### One thing that IS reasonably clear

Freezing roughly **doubles the variance** (FROZEN sd 0.248 vs LIVE sd 0.129,
1.93x). That is a larger, more consistent effect than anything in the means, and
it suggests continued training has a *stabilising* role even if it does not
improve the average. That is a more defensible claim than either mean-based
story, though still n=9.

### Consequence for the whole project

This same power problem applies to the **headline h1 result**: MF-DRO vs
MI-Greedy differ by 0.0045 in the mean, with MI-Greedy's sd at 0.400. Detecting
a difference that small would need many hundreds of seeds. The frozen success
test's FAIL verdict stands as pre-registered, but "MF-DRO and MI-Greedy are
statistically indistinguishable" should be stated as *underpowered to
distinguish*, not as demonstrated equivalence.

## (superseded, n=7) freezing the DT costs NOTHING

| n | mean paired diff (FROZEN - LIVE) | verdict vs the locked 1-SE band (0.0395) |
|---|---|---|
| 1 | -0.2084 | looked like freezing *helps* a lot |
| 5 | -0.1033 | OUTSIDE — locked prediction looked refuted |
| **7** | **-0.0102** | **WITHIN — locked prediction SUPPORTED** |

At n=7: FROZEN 0.4909 vs LIVE 0.5011, **Wilcoxon p = 0.9375** — about as
non-significant as a result can be. FROZEN better on 3/7, worse on 4/7.

**This is a cautionary datum in its own right.** The same experiment "showed"
freezing helps by 40% (n=1), then helped moderately (n=5), then nothing at all
(n=7). Every intermediate reading would have been publishable-sounding and
wrong. I flagged n=1 and n=5 as directions rather than results, and that
restraint is the only reason the record is not now carrying a retracted claim.

### What this supports

The locked H6 prediction was that FROZEN lands within 1 SE of LIVE, meaning
**continued DT training contributes ~nothing measurable**. At n=7 that is what
the data show — and FROZEN gets there **1.6x faster** (47 min vs 76 min per run).

Combined with H4 and H5, the picture is coherent and now has three independent
lines of support:

1. H4: changing *how* the return signal enters (AdaLN) does not help.
2. H5: denying the score head its shortcut does not make it use the state.
3. H6: freezing the policy entirely, after 5 of ~60 iterations, costs nothing.

**MF-DRO's Decision Transformer contributes essentially nothing beyond its first
few iterations of training.** The method's behaviour is carried by the GP
ensemble and the MES teacher it distills, not by the learned return-conditioned
policy.

### A distinction this forces, which I had been eliding

FROZEN issues **more** HF queries (21.0 vs 17.6, more in 5/7 seeds) and yet
regret is unchanged. That is not compatible with the loose reading "more HF
queries -> better regret" that I drifted toward earlier.

The correct, narrower statement is: *within* a run, regret only ever moves on an
HF query (an LF query cannot update the HF incumbent). That does **not** imply
that *across* runs, more HF queries produce better final regret — extra HF
queries at poor locations buy nothing. Both facts are true and I had been
sliding between them.

### Still provisional

3 seeds outstanding. Given this estimate moved from -0.208 to -0.103 to -0.010,
the n=10 value could move again. No final claim until the arm completes.

## (superseded) H6 first result (n=1): continued DT training may be *harmful*

seed43, FROZEN (DT weights frozen after iteration 5) vs LIVE (retrained every
iteration, the current default):

| | regret | HF rate | n_improved | iters | wall |
|---|---|---|---|---|---|
| FROZEN | **0.3134** | **24/34 = 70.6%** | 4 | 34 | 30 min |
| LIVE | 0.5218 | 21/58 = 36.2% | 1 | 58 | 57 min |

Freezing the policy after 5 iterations made this seed **40% better** on regret,
improved the incumbent 4x more often, and did it in half the wall time.

**This is pre-registered outcome #3**, written into the protocol before the run:
*"FROZEN is clearly better -> continued training is actively harmful
(overfitting to fresh rollouts), which is itself a concrete, actionable finding
about num_epochs / retraining cadence."*

### It also supplies the missing mechanism for a standing open question

Three previously-separate observations now compose into one causal chain:

1. `fid_mean_per_iter` **declines over every run** — the fidelity head drifts
   toward LF as training continues. (Previously logged as "unexplained".)
2. Regret moves **only** on HF queries, never on accumulated LF cost.
3. Freezing the DT at iteration 5 **stops the drift**, preserving a much higher
   HF rate (70.6% vs 36.2%) — and regret improves accordingly.

So the proposed mechanism is: **continued training progressively destroys the
policy's willingness to spend on HF, and since only HF queries can move the HF
incumbent, more training makes the method worse.** That is a far more specific
and testable claim than "the DT contributes nothing".

### CORRECTION: the proposed mechanism does NOT survive contact with all 9 runs

I proposed above that "continued training progressively destroys the policy's
willingness to spend on HF". Testing that directly on the 9 completed LIVE runs
(HF rate in the first vs last quartile of each run):

    mean HF rate, first quartile : 39.8%
    mean HF rate, last  quartile : 31.9%
    drop                          : only 7.9 points

    per-run: 40%->0%, 71%->57%, 10%->33%, 18%->18%, 50%->6%,
             15%->35%, 15%->35%, 100%->86%, 39%->17%

**Four of nine runs INCREASED their HF rate.** The trend is weak, noisy, and
reverses in a third of runs. This is not the clean monotonic collapse the
mechanism story needs.

The distinction I blurred: `fid_mean_per_iter` (the head's mean predicted P(HF)
over the *training batch*) declining — measured earlier on 3 seeds — is **not**
the same quantity as the realized HF *query* rate declining. The first may still
be true; the second is largely not, across 9 runs.

So the seed43 FROZEN result (0.313 vs 0.522) still stands as an observation, but
my causal explanation for it is **not supported** and should not be carried
forward as if it were. Wait for the full 10 FROZEN seeds before proposing any
mechanism, and prefer a mechanism that survives all seeds rather than one built
from the single most favourable comparison.

### Do not over-read this

**n = 1.** One seed, one comparison. It is a direction, not a result. The
remaining 9 FROZEN seeds are running and the locked primary prediction was that
FROZEN would land *within 1 SE* of LIVE — a clearly-better FROZEN arm would
refute that prediction in the informative direction. Report the full 10 before
claiming anything, and note that seed43 was LIVE's *worst* HF-rate seed, so
regression to the mean could account for part of the gap.

## Methodological turn: measure the mechanism, not the downstream metric

Reviewing which findings held up and which did not reveals a clean pattern:

| finding | instrument | outcome |
|---|---|---|
| target leakage | `a_emb` ablation (15.6% -> 0.7%) | decisive |
| MF-GP-UCB freeze definitional | `n_HF = 0` on 10/10 seeds | decisive |
| score head ignores `h` | argmax unchanged 12/12, both arms | decisive |
| RTG never moves the argmax | pinned in every configuration | decisive |
| freeze resolved | 0/10 vs 9/12 | decisive (large effect) |
| **H6: does training help regret?** | paired regret, n=10 | **useless** (CI [-0.097,+0.292]) |

Every **near-deterministic** measurement in this project settled its question
immediately. The single **variance-dominated average** (H6's regret comparison)
consumed ~2 hours of compute, changed sign four times, and settled nothing.

This is not bad luck — it is a design lesson. H6 asked a *mechanistic* question
("does continued training change the policy?") and answered it with the noisiest
available proxy (final regret, which depends on the GP, the teacher, the fidelity
mix and the seed's initial design as well as on the policy).

**H7 (locked, 93429a0) applies the lesson**: run one trajectory, snapshot the DT
at iteration 5, and at every later iteration ask the live and snapshot policies
to act on the *identical* state/RTG/BTG/candidate pool. Record whether they pick
the same candidate. That yields **~50-200 paired decisions per run** instead of
one regret scalar, and agreement is near-deterministic.

If agreement is high, H6's claim ("continued training is near-inert") becomes
defensible on mechanism even though its own regret comparison is underpowered.
If agreement is *low* while regret is unchanged, that is an equally strong and
more surprising result: the policy changes substantially and it does not matter,
meaning the decision space is flat and the GP/MES teacher is doing the real work.

## The secondary-basin story is real as description but does NOT explain regret

Mined the 10 post-fix seeds for `query_dist_to_xstar` vs `query_dist_to_x2`
(x2 = Hartmann-6D's second optimum, f=3.2031, ||x2-x*||=1.103):

| | post-fix (n=10) | pre-fix (n=3) |
|---|---|---|
| mean d(x*) | 0.899 | 0.97-1.10 |
| mean d(x2) | 0.709 | 0.62-0.70 |
| closer to x2 | **7/10 seeds** | 3/3 seeds |

So the attraction to the secondary basin **persists but is no longer
universal** — seeds 46, 49 and 50 now sit closer to x*. Seed46 in particular
gets genuinely close (d(x*) = 0.307).

### The part that kills the explanation

    corr(mean d(x*), final regret) = +0.091
    corr(mean d(x2), final regret) = -0.145

**Where the policy queries does not predict how well it does.** Concretely:

- seed46 gets *closest to x\** (0.307) and lands at a middling regret 0.4412
- seed42 stays *far from x\** (1.038) and achieves the **best** regret 0.3126

I had recorded "the policy was converging on the wrong basin" as an explanation
for poor performance. It is not one. The mechanism is straightforward in
hindsight: **regret depends on the single best point ever evaluated, not on where
queries sit on average.** A policy can average far from x* and still stumble onto
one good HF point; a policy can hover near x* and never evaluate HF at the right
spot. Average query location and best-found-value are only loosely coupled.

This retires a plausible-sounding story. It also explains why `query_dist_to_x*`
looked so damning earlier: it is a real description of behaviour that happens to
carry almost no information about outcome.

**Generalisable lesson**: a diagnostic can be simultaneously (a) correctly
measured, (b) a true description of the policy, and (c) causally irrelevant to
the metric. Before promoting any diagnostic to an explanation, correlate it
against the outcome it is supposed to explain. I did not do that when I first
recorded the secondary-basin finding.

## Lessons and Constraints

- **Completion order in a cost-budgeted grid is biased toward HF-heavy runs.**
  A run that spends its budget on HF finishes in few iterations (seed50: 25 HF
  x c_H=8 = 200 cost in 28 iterations, 27 min); an LF-heavy run needs ~200
  iterations for the same cost and takes hours. So the *first* results to land
  systematically over-represent HF-heavy behaviour. Never generalize from a
  partial cost-budgeted grid — wait for all seeds.
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
