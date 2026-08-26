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

### The ceiling is the acquisition, not the transformer (H31)

Running the MF-MES teacher under the frozen evaluation with **no DT deciding**
(DT still trained, so the RNG stream matches and both arms see identical
candidate pools):

| method | final simple regret |
|---|---|
| MF-MES teacher, **no DT** | 0.4781 ± 0.0414 |
| MF-DRO / joint MES | **0.4007 ± 0.0475** |
| MF-MI-Greedy | 0.5091 ± 0.1266 |

Paired teacher − MF-DRO **+0.0774**, teacher better on **3/10**, Wilcoxon
**p = 0.2324**. H32's pre-landing expectation of near-parity is **confirmed**,
and *"the transformer is a net negative" is refuted* — the apparatus neither
costs nor buys performance.

**The single most consequential number: the teacher alone also fails the frozen
test** (mean+SE 0.5195 vs the 0.3825 bar). It is not the transformer that cannot
clear the bar — the whole MF-MES-based approach cannot, distilled or not. The
ceiling is set by the acquisition.

Fidelity mix over full runs: teacher **11.4%** HF (12.8 of 112 iters) vs MF-DRO
**26.7%** (18.9 of 70.7) — 2.3× the rate, 37% fewer queries per budget.
**This corrects H33's framing**: the head's wrong *level* is not demonstrably
harmful here (it coincides with slightly lower regret). What survives is its
*uninformativeness* (sd 2.4e-4, corr 0.155).

### But the frozen metric samples one point of a trajectory (H37)

Cost-weighted regret (AUC over the budget) tells a different story:

| method | AUC/budget | r@100 |
|---|---|---|
| **MF-DRO / joint MES** | **0.6805 ± 0.070** | **0.490** |
| MF-DRO / improvement | 0.8716 ± 0.057 | 0.815 |
| MF-MI-Greedy | 1.0639 ± 0.127 | 1.042 |
| MF-GP-UCB | 1.7934 ± 0.122 | 1.793 |

- The reward change is **consistently better** here: paired −0.1911, **9/10
  seeds**, lower at every checkpoint. **NOT a significance claim** — the p of
  0.0371 came from a post-hoc metric computed after the pre-registered one
  failed; over the 7 tests reported in that table the Bonferroni threshold is
  0.00714 and **none survive**.
- MF-DRO beats MI-Greedy by **2.1×** at cost 100, narrowing to 0.401 vs 0.593 by
  cost 200. Its advantage is **early**; MI-Greedy closes late.

Consistent with the fidelity finding: MF-DRO queries HF ~26–31% vs the teacher's
~12%, buying early incumbent progress at late-budget cost. **Supplementary only**
— the frozen success test is on final regret, still FAILS, and vs MI-Greedy this
axis gives p = 0.0645, no more significant at n=10.

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

### Two channels, and only one is dead

A recurring source of confusion, worth stating explicitly:

| channel | status | evidence |
|---|---|---|
| **Training signal** — the reward/RTG *definition* sets training targets and so changes the learned `w̄` | **LIVE** | H17: joint-MES reward moved regret **0.5047 → 0.4007** |
| **Inference conditioning** — the RTG/BTG *value fed at decision time* | **INERT** | H8/H22/H23/H26 |

Both are true simultaneously. An RTG-*schema* ablation varies the first; every
0/12 probe here tests the second. "The RTG value you hand the network at
inference does nothing; the RTG definition you train against does something."

**On the archived `results/rtg_schema` ablation** (raised as a counterexample):
it does not show a genuine improvement. No Mann-Whitney comparison reaches
significance (all p ≥ 0.4633); quantile RTG wins on 1 of 4 benchmarks and is the
*worst* arm on Ackley 5D; and on Hartmann 6D all four quantile levels return
**bit-identical** results (2.6298 ± 0.1545), i.e. the quantile parameter had no
effect there at all. It is also single-fidelity, N=5, and pre-`7bcc3b8`
(regret 2.4476 vs post-fix MF-DRO's 0.4007). It is evidence *for* inertness.

### Measured scope of the inertness (H26)

| channel | in-band | far out of band |
|---|---|---|
| BTG | 0/12 (corr 1.000000) over [22,52] | 0/12 at {5,100,500} |
| RTG | 0/12 (corr 0.999933) over [0.5,1.0] | **1/12 = 8.3%** over 1e-3→1e6 |

Absolute invariance is **not** literally true — a single pool flips under a
10⁹-fold stretch. What is true: no in-distribution value of either channel moves
the decision.

### And the fixed rule is uncertainty-AVERSE (H24, robustness H25)

Signed `w̄`: `mu_H` **+1.0824**, `mu_L` **+0.8254**, **`sigma_H` −0.5487**,
`dist_inc` −0.2976.

**The weight on HF posterior uncertainty is negative** — but H30 shows this is a
**partial** coefficient, not an aversion. `corr(mu_H, sigma_H) = −0.4696` across
candidates, and under that collinearity the *teacher's own* score has a negative
partial `sigma_H` coefficient (−0.0419, negative on 91.4% of sets) despite a
*positive* marginal correlation (+0.1585) — textbook suppression. The student
reproduces its teacher's partial coefficient; it does nothing MES does not.
**The robust claim is behavioural, not coefficient-based**: selections land in
the bottom 3% of posterior uncertainty. **H25 confirms the sign across seeds: `w[σ_H]<0`
on 9/10 independently trained models, `w[μ_H]>0` on 10/10.** The *magnitude* is
not robust (span −0.687 to +0.130, mean −0.2544, sd 0.2578), so the distribution
is quoted rather than any one model's coefficient. Confirmed independently: agreement with
`mu_H + β·sigma_H` falls monotonically as β grows (66.7% → 50.0% → 41.7% →
25.0% for β = 1, 2, 3, 5). We do *not* name the rule — the best match is 75.0%
with a two-way tie on 12 pools, too weak for an identity claim.

This is the causal link to the performance result: penalising HF uncertainty
concentrates queries where the model is already confident → little information →
slow incumbent improvement, the pathology this investigation began from.
**RETRACTED (H28): the student inverts nothing.** Over 1600 teacher decisions,
the MF-MES teacher's own choices sit at the **2.9th percentile** of `sigma_H`
within their pools (control: 94.2nd percentile of `mu_H`). The aversion is
**inherited by faithful imitation**. The teacher's *score* mildly rewards
uncertainty (Spearman +0.1585) but its *argmax* is dominated by the posterior
mean (+0.8517), and high-`mu` regions are where data already sits. **MF-MES's
realised behaviour here is exploitative even though its scoring rule is not** —
so a better student cannot fix it.

**And it is INTRINSIC (H29).** Sweeping cost ratio {2,4,8,16} × y* samples
{5,10,50}, the chosen-`sigma_H` percentile never exceeds **5.5%** in any of 12
cells (control 93.6–94.7%). The y* sample count is irrelevant (K=5/10/50
identical), and an 8× cost-ratio swing moves it only ~1% → ~5%. So: **a student
imitating the *choices* of an argmax-of-MES teacher cannot learn to explore at
any operating point tested.** The fix is not a better student or a different
cost ratio — it is learning from the teacher's *scores* over the full candidate
set, or an exploration-augmented demonstrator.

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


## What the policy actually is: a faithful copy of an exploitative teacher

Five experiments (H28–H33) rewrote the causal account three times. The final
version:

| claim | status | evidence |
|---|---|---|
| The teacher selects low-uncertainty points | **solid** | chosen `sigma_H` at the **2.9th percentile**; control `mu_H` at 94.2nd (H28) |
| ...at every operating point | **solid** | never above **5.5%** across cost ratio {2,4,8,16} × y* samples {5,10,50} (H29) |
| The student *inverted* its teacher | **RETRACTED** | it inverts nothing — it imitates (H28) |
| `w[sigma_H]<0` means uncertainty aversion | **CORRECTED** | it is a *partial* coefficient under `corr(mu,sigma) = −0.4696`; the **teacher's own** partial coef is also negative (−0.0419, on 91.4% of sets) despite a positive marginal (+0.1585) — suppression, not aversion (H30) |
| The distillation is lossy | **CORRECTED** | teacher's rank of the student's pick: **median 2 of 200**, range [1,12]. Faithful. My "lossy" read came from argmax agreement, a poor metric when the top candidates are near-ties (H32) |
| The **fidelity** channel diverges | **solid** | `p` spans 0.5570–0.5577 (sd 2.4e-4) while the teacher's `ell` varies; `corr = +0.155`. Uninformative, and 0.557 vs the teacher's 4.2% on identical pools (H33) |

**The synthesis.** MF-DRO's transformer reproduces its teacher's *location*
choices closely, fails entirely to reproduce its *fidelity* choices, adds no
conditioning (fixed rule to within 0.13%), and inherits the teacher's
exploitative selection behaviour. The apparatus is **redundant rather than
harmful** on location, and actively uninformative on fidelity.

A student cannot out-explore a demonstrator that never explores. That is a
statement about the *approach* — distilling an argmax-of-MES teacher — and H29
shows it is not tunable away.

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
14. **Pick the decisive metric, not a proxy that correlates with it.** Four
   instances: H13's saturating decomposition metric, H19's diversity signature,
   the `coef_head` cosine threshold, and H24's argmax agreement — which made a
   *faithful* distillation look lossy for three ticks until H32 measured the
   teacher's rank of the student's pick instead.
15. **A component's specification is not its behaviour.** MF-MES is defined as
   information-seeking; its realised choices sit at the 2.9th percentile of
   uncertainty. Checking what a component *does* rather than what it *is for*
   has now caught four errors here.
16. **A script's auto-verdict must key on the DECISIVE measurement, not a
   proxy that correlates with it.** Four wrong auto-verdicts this phase — H11
   (fired without requiring the manipulation check to pass), H13 (a metric that
   saturated at 100% in both compared arms), H19 (a signature omitting the
   variable of interest), and the `coef_head` probe (cosine threshold instead of
   argmax movement). Each proxy looked reasonable when written and printed a
   confident, wrong sentence when run.
17. **When a probe compares "two different X", assert inside the probe that
   they are different.** H5 would have failed a one-line assertion for months.
   Three instrument defects this phase (H13's dead-signal metric, H19's
   diversity signature, H5's state swap) — all found by checking the instrument
   against the data rather than trusting it.
18. **Do not fix a scale-free quantity with an absolute threshold** (bit twice:
   `bes_delta`, soft-target temperature).
19. **When a supporting subset is in hand and the rest is cheap, do not report
   the subset at all.** This generalises lesson 4, which was too narrow — the
   failure is not specific to intermediate *n*. Four instances in this project,
   four different mechanisms, one shape: the supporting subset existed
   *earlier* than the full set, and got reported first.

   | instance | the subset reported | the full set |
   |---|---|---|
   | `p=0.0371` called significant | one metric of several | Bonferroni over the metrics actually examined — nothing survives |
   | h45 head comparison, reported at 5/6 then 7/8 | seeds finished so far | 10/10: worst-on-mean, Wilcoxon **p=1.0000** |
   | H50 criterion drift (caught pre-results at `58081bb`) | the criterion the data would support | the registered 3-of-3 test |
   | H50 smoke test called "encouraging" | seeds 49 vs 42 | all four seeds — criterion #1 points the **wrong way** |

   Note the third row: drift is the same failure applied to *criteria* instead
   of *data*. And the second reached a user and changed a shipped default
   (`use_candidate_scoring`, 6c7989b) that the full set does not support.

   The rule is stronger than "be vigilant" because vigilance already failed
   here four times. Waiting costs little when the remainder is cheap — the four
   outstanding h45 seeds were hours, but the two missing smoke-test seeds were
   **30 seconds**, and they reversed the conclusion. Independently arrived at
   with the peer session, which counted its own instances in the same tally.

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

---

## H57 baselines complete (27/27 cells) — the bar MF-DRO has to clear

**CONFIRMATORY** for the three baseline arms; MF-DRO's 9 cells are still running
and are **withheld** (lesson 19 — do not report the subset when the rest is
coming). Budget 200 post-init, seeds 44/46/48, all methods drawing the identical
initial design via `init_design.make_initial_design`, every result stamped with
its code commit.

Final HF simple regret, normalised by f(x*) so benchmarks are comparable at all:

| method | Currin 2D | Hartmann 6D | Borehole 8D |
|---|---|---|---|
| MF-MES (Takeno, L-BFGS-B) | 0.6% | **8.5%** | 11.3% |
| MF-MI-Greedy | **0.2%** | 23.9% | **8.3%** |
| MF-GP-UCB | 10.0% | 45.3% | 44.1% |

Paired win counts (3 seeds):

| | Currin | Hartmann | Borehole |
|---|---|---|---|
| MES vs MI-Greedy | 1-2 | **3-0** | 1-2 |
| MES vs GP-UCB | 3-0 | 3-0 | 3-0 |
| MI-Greedy vs GP-UCB | 3-0 | 3-0 | 3-0 |

### The finding that matters for the north star

**No single baseline dominates.** MF-MES wins Hartmann 3-0 but loses Currin and
Borehole 1-2 to MF-MI-Greedy. So "at least as good as the baselines" is not one
number to beat — it is the per-benchmark best, and that best changes identity
across benchmarks:

    Currin 2D    0.2%   (MI-Greedy)
    Hartmann 6D  8.5%   (MF-MES)
    Borehole 8D  8.3%   (MI-Greedy)

A method that matched MF-MES everywhere would still lose Currin by 3x and
Borehole by 1.4x. This is a harder target than "beat MF-MES", which is how the
goal has usually been phrased in this project.

**MF-GP-UCB is decisively last**, 0-3 against both others on every benchmark.
It is not a competitive baseline here and should not be used to flatter a
comparison.

n=3 per cell. No p-values. Directions and win counts only.

---

## Lesson 20 + correction: distance to x* is not a proxy for value, and the
## "misdirected search" reading of MF-DRO's stalls was wrong

**RETRACTION.** I classified h57's Hartmann seed-44 stall as MISDIRECTED on the
grounds that its stalled HF queries sat at normalised distance 0.551 from x*
while its own incumbent sat at 0.319 — "spending HF budget farther from the
optimum than where it already stands". That inference is invalid.

Measured on 6000 uniform samples per benchmark:

| benchmark | corr(d*, f) | Spearman | best f at d*<0.3 | best f at d*>1.0 |
|---|---|---|---|---|
| Currin 2D | -0.799 | -0.805 | 13.7985 | 5.5057 |
| **Hartmann 6D** | **-0.417** | -0.568 | **2.9196** | **2.8338** |
| Borehole 8D | -0.587 | -0.571 | *0 samples* | 75.3033 |

On Hartmann the best value reachable **within** 0.3 of x* is the same as the best
reachable **beyond** 1.0 away. In Borehole's 8D domain, 6000 uniform samples
contain zero points within 0.3 of x* at all. Distance to x* only carries
information on Currin.

The direct comparison makes the point sharper. Hartmann seed 44, same initial
design, both traces on disk:

| method | nHF | regret | d*(HF queries) | d*(best HF) |
|---|---|---|---|---|
| MF-DRO | 25 | 0.7531 | **0.542** | **0.319** |
| MF-MES | 19 | **0.3019** | 1.103 | 1.146 |
| uniform reference | | | 0.880 | |

MF-DRO searched twice as close to x* as MF-MES and its best point was 3.6x
closer — and lost on regret by 2.5x. Closeness to x* is not what wins here.

### What the value-based classifier says instead

Replacing distance with each query's PERCENTILE in a 20k-Sobol reference
distribution of f over the domain (`src/analysis/value_reference.py`):

| benchmark | seed | stall | %HF | q pctile | inc pctile | cause |
|---|---|---|---|---|---|---|
| Hartmann | 48 | 42 | 19% | 98.0% | 100.0% | NEAR-MISS |
| Hartmann | 44 | 30 | 80% | 88.7% | 98.7% | PLATEAU |
| Borehole | 44 | 19 | 100% | 99.7% | 99.8% | NEAR-MISS |
| Currin | 44 | 15 | 27% | 99.9% | 100.0% | NEAR-MISS |
| Hartmann | 46 | 13 | 0% | — | 100.0% | HF-STARVED |
| Currin | 48 | 6 | 17% | 99.9% | 100.0% | HF-STARVED |

**Not one cell is searching badly.** Every stalled cell's HF queries land in the
top 1-11% of the domain by value, against incumbents already at the 98.7-100th
percentile. There is no low-value search anywhere in the data.

So the stall is not misdirection and not aimlessness. It is that the incumbent
is already near the ceiling and the remaining gap is the last fraction of a
percent — where MF-MES happens to do better. That is a much narrower problem
than "the policy searches badly", and it means diagnostics aimed at query
placement are aimed at the wrong thing.

**Lesson 20**: a geometric proxy for a value question must be validated against
the value before it is used to classify anything. The cost of not doing so was a
confident, wrong mechanism reported to the user.

---

## MF-DRO's fidelity policy is unstable across seeds — a concrete mechanism for
## its regret variance, and a caveat on MF-GP-UCB

**EXPLORATORY** (not in any locked protocol; found while checking why h58's
FLOOR arm was a no-op). n=3 seeds per cell.

HF fraction of post-initial-design queries:

| benchmark | method | s44 | s46 | s48 | spread |
|---|---|---|---|---|---|
| Hartmann 6D | **MF-DRO** | **81%** | **4%** | **28%** | **76 pts** |
| Hartmann 6D | MF-MES | 26% | 38% | 67% | 40 pts |
| Hartmann 6D | MF-MI-Greedy | 12% | 13% | 11% | **2 pts** |
| Currin 2D | MF-DRO | 20% | 31% | 43% | 23 pts |
| Currin 2D | MF-MI-Greedy | 25% | 25% | 25% | 0 pts |
| Borehole 8D | MF-DRO | 100% | 99% | 96% | 4 pts |

On Hartmann, MF-DRO's fidelity head sends 81% of the budget to HF on one seed and
4% on another. That is not adaptation to the seed's landscape — all three seeds
share a benchmark and differ only in the initial design draw. It is the fidelity
head failing to learn a stable policy.

This is a concrete candidate mechanism for the variance that has dogged MF-DRO
all along: h45 measured the regression head at s.e. **0.1483** against scoring's
0.0475 and the teacher's 0.0414, roughly 3x. A policy that spends anywhere from
4% to 81% of its budget on the expensive fidelity will produce exactly that
spread in outcomes. MF-MI-Greedy, whose HF fraction varies by 2 points across
seeds, has correspondingly tight regret.

Note Borehole is the exception: MF-DRO is stable there (96-100%) because at
c_H=2 the HF is nearly as cheap as LF and the choice barely matters.

### Caveat on the h57 baseline table: MF-GP-UCB takes ZERO HF queries

MF-GP-UCB is at 0% HF on **all three benchmarks, all nine cells**. Its incumbent
can only move on initial-design points. That is very likely why it went 0-3
against both other baselines everywhere, and it means its numbers in the h57
table (10.0% / 45.3% / 44.1% relative regret) should NOT be read as a fair
assessment of MF-GP-UCB as a method. Either its cost-weighted acquisition is
degenerate at these cost ratios or there is a defect in that baseline. Flagged,
not yet diagnosed — and until it is, MF-GP-UCB should not be used as a
comparison point in any writeup.

### Correction to the MF-GP-UCB caveat above

I wrote that MF-GP-UCB's 0% HF is "either a degenerate cost-weighted acquisition
or a defect in that baseline. Flagged, not yet diagnosed." That was wrong to the
extent it implied a possible bug — the behaviour is **documented in the class
docstring**, predicted in advance, and attributed to the method:

> On Currin 2D (c_H=3) and Hartmann 6D (c_H=8), MF-GP-UCB degenerates to all-LF
> selection when the cost ratio exceeds the prior variance ratio of the two
> fidelity GPs. This is a known limitation of the cost-normalized UCB approach
> (Kandasamy et al. 2016) and is expected behavior on these benchmarks. The
> regret curve will plateau since HF is never queried.

So the 0-3 record is the documented failure mode of naive cost-normalized UCB at
these cost ratios, not an implementation defect. My "not yet diagnosed" was a
failure to read the docstring of the class I was running.

Two things that survive the correction:

1. **The reporting caveat stands.** MF-GP-UCB in a results table is a method
   operating in a regime where it provably cannot query HF. Quoting its 45.3%
   relative regret next to MF-MES's 8.5% invites the reading that MF-MES is 5x
   better at optimisation, when the actual content is that one method never
   spent on the fidelity that sets the incumbent. It should be labelled as a
   degenerate-regime reference, not a competitor.

2. **The docstring does not cover Borehole 8D, and Borehole degenerates too.**
   At c_H=2 — the *lowest* cost ratio of the three — MF-GP-UCB is still at 0% HF
   on all three seeds. The stated mechanism (cost ratio exceeding the prior
   variance ratio) predicts degeneracy should be *least* likely there. Either
   the LF/HF prior variance ratio on Borehole is under 2, or the mechanism is
   not what drives it. Genuinely undiagnosed, unlike the first two benchmarks,
   and cheap to settle by reading the two GPs' prior variances directly.

### Borehole resolved: MF-GP-UCB cannot query HF on ANY of these three benchmarks

Measured on 4096 Sobol points per benchmark, the docstring's own rule (all-LF
when the cost ratio exceeds the two fidelities' prior variance ratio):

| benchmark | c_H/c_L | sd(f_L) | sd(f_H) | var ratio | cost > var? |
|---|---|---|---|---|---|
| Currin 2D | 3.0 | 2.616 | 2.650 | 0.974 | YES |
| Hartmann 6D | 8.0 | 0.363 | 0.386 | 0.886 | YES |
| Borehole 8D | **2.0** | 36.309 | 45.628 | **0.633** | **YES** |

My suspicion that "the mechanism is not what drives it" on Borehole was wrong.
It does drive it: Borehole's variance ratio is only 0.633, so even the lowest
cost ratio of the three clears it comfortably.

The general statement is stronger than the docstring's benchmark-specific note.
**The variance ratio is below 1 on all three benchmarks** — the LF surrogate has
similar or lower spread than HF in each case — so any cost ratio above 1
triggers the degeneracy. MF-GP-UCB is structurally incapable of buying an HF
query on any benchmark in this suite, at any cost setting where HF is more
expensive than LF at all.

That makes it a *constant* in the h57 table rather than a comparison: it measures
what the shared initial design achieves plus pure LF refinement. Useful as a
floor, worthless as a competitor, and it should be presented that way or dropped.

---

## H57: MF-DRO vs baselines — Currin and Borehole complete (Hartmann withheld, 11/12)

**CONFIRMATORY.** Budget 200 post-init, seeds 44/46/48, shared initial design,
one pinned commit, hash in every result. Hartmann's MF-DRO seed 46 is still
running and that whole benchmark is withheld per lesson 19.

### Currin 2D — MF-DRO is competitive, but the benchmark is saturated

| method | s44 | s46 | s48 | mean | rel | HF% | wall |
|---|---|---|---|---|---|---|---|
| **MF-DRO** | 0.0001 | 0.0031 | 0.0011 | **0.0014** | **0.0%** | 32% | 114.1 m |
| MF-MES | 0.1954 | 0.0049 | 0.0362 | 0.0788 | 0.6% | 6% | 3.5 m |
| MF-MI-Greedy | 0.0000 | 0.0000 | 0.0669 | 0.0223 | 0.2% | 25% | **0.5 m** |
| MF-GP-UCB | 1.5399 | 2.2324 | 0.3614 | 1.3779 | 10.0% | 0% | 2.6 m |

MF-DRO 3-0 over MF-MES, **1-2 against MI-Greedy**. Its mean is the lowest of any
method, but the paired count goes the other way because MI-Greedy hits exactly
0.0000 on two seeds. Every non-degenerate method is inside 0.6% of the optimum
here — Currin at this budget does not discriminate.

### Borehole 8D — MF-DRO loses decisively

| method | s44 | s46 | s48 | mean | rel | HF% | wall |
|---|---|---|---|---|---|---|---|
| MF-DRO | 75.64 | 76.59 | 67.97 | 73.40 | **23.7%** | 98% | 92.1 m |
| MF-MES | 47.12 | 25.03 | 32.58 | 34.91 | 11.3% | 74% | 5.1 m |
| **MF-MI-Greedy** | 22.13 | 20.92 | 33.75 | **25.60** | **8.3%** | 100% | **0.7 m** |
| MF-GP-UCB | 138.82 | 113.30 | 157.41 | 136.51 | 44.1% | 0% | 2.7 m |

**MF-DRO is 0-3 against both real baselines**, at 2.9x MI-Greedy's regret. This
is not a variance story: it loses every seed.

### The bar, and where MF-DRO stands against it

The north star is "at least as good as the baselines". On complete data:

    Currin 2D    MF-DRO 0.0%  vs best baseline 0.2%   -> meets it (saturated)
    Borehole 8D  MF-DRO 23.7% vs best baseline 8.3%   -> FAILS, 0-3

One benchmark where everything works and one where it loses every seed is not
"at least as good as the baselines".

### Compute is the other half of the result

MI-Greedy: **0.5-0.7 minutes** per run. MF-DRO: **92-114 minutes**. Roughly
**150-200x** the compute, for a decisive loss on Borehole and a tie inside the
noise on a saturated Currin. Any claim for a DRO-family method has to answer
that ratio, not only the regret.

n=3 per cell, no p-values. Directions and paired counts only.

---

## H57 COMPLETE (36/36): MF-DRO does not meet the bar

**CONFIRMATORY.** All 36 cells, budget 200 post-init, seeds 44/46/48, shared
initial design, one pinned commit, hash in every result file.

Relative simple regret (regret / f(x*)), and paired wins for MF-DRO out of 3:

| method | Currin 2D | Hartmann 6D | Borehole 8D |
|---|---|---|---|
| **MF-DRO** | **0.0%** | 14.7% | 23.7% |
| MF-MES | 0.6% | **8.5%** | 11.3% |
| MF-MI-Greedy | 0.2% | 23.9% | **8.3%** |
| MF-GP-UCB *(all-LF, floor)* | 10.0% | 45.3% | 44.1% |

| MF-DRO vs | Currin | Hartmann | Borehole |
|---|---|---|---|
| MF-MES | **3-0** | 0-3 | 0-3 |
| MF-MI-Greedy | 1-2 | 2-1 | 0-3 |

### Verdict against the north star

"At least as good as the baselines", per benchmark, against the best baseline:

    Currin 2D    MF-DRO 0.0%  vs 0.2%  -> meets it, but every method is <0.6%
                                          (saturated; does not discriminate)
    Hartmann 6D  MF-DRO 14.7% vs 8.5%  -> FAILS, 0-3 to MF-MES
    Borehole 8D  MF-DRO 23.7% vs 8.3%  -> FAILS, 0-3 to BOTH baselines

**MF-DRO fails on both discriminating benchmarks and ties on the saturated one.**
It never beats the best available baseline anywhere. The one clean win it does
have — 3-0 over MF-MES on Currin — is on the benchmark where all three
non-degenerate methods land inside 0.6% of the optimum.

Note the identity of the best baseline changes: MF-MES on Hartmann, MI-Greedy on
Borehole and Currin. Neither baseline dominates, which makes the bar harder than
"beat MF-MES".

### The compute ratio is not a footnote

| | Currin | Hartmann | Borehole |
|---|---|---|---|
| MF-DRO | 114.1 m | 82.5 m | 92.1 m |
| MF-MES | 3.5 m | 1.9 m | 5.1 m |
| MF-MI-Greedy | **0.5 m** | **0.4 m** | **0.7 m** |

**MF-DRO costs 120-230x MI-Greedy's wall time** and loses to it on two of three
benchmarks. A method needing 200x the compute to lose is not a tuning problem.

### Where this leaves the SF-DRO north star

The multi-fidelity extension is not the thing to keep fixing. Every mechanism
this project has chased — incumbent freeze, aimless search, mean-collapse,
misdirection — has either been retracted or turned out not to be what costs the
regret. What the complete data says instead:

1. The search is **not** bad. Stalled queries sit at the 88.7-99.9th percentile
   of domain value against incumbents at 98.7-100th.
2. The fidelity policy **is** unstable: 81%/4%/28% HF across three Hartmann
   seeds that differ only by initial design.
3. The method plateaus slightly below MF-MES and far below MI-Greedy on
   Borehole, at 100-200x the cost.

Plot: `to_human/h57_regret_vs_cost.png` (relative regret vs cost, min-max band
over seeds, log y).

---

## H59 Hartmann complete (6/6): SF-DRO beats SF-MES 3/3 — and beats MF-DRO

**CONFIRMATORY.** Cost budget 200 post-init, seeds 44/46/48, the same axis every
h57 number is on. **Currin and Borehole SF-DRO cells are still running and are
withheld.** This is one benchmark of three.

| arm | s44 | s46 | s48 | mean | rel | wall |
|---|---|---|---|---|---|---|
| **SF-DRO** | 0.3815 | 0.4406 | 0.3233 | **0.3818** | **11.5%** | 19.4 m |
| SF-MES | 0.7531 | 1.0214 | 0.3575 | 0.7107 | 21.4% | 0.0 m |

**SF-DRO beats SF-MES on 3/3 seeds.** Placed against h57's multi-fidelity
numbers on the identical cost axis:

| method | Hartmann 6D rel. regret |
|---|---|
| MF-MES | **8.5%** |
| **SF-DRO** | **11.5%** |
| MF-DRO | 14.7% |
| SF-MES | 21.4% |
| MF-MI-Greedy | 23.9% |

### Why this matters

H59's locked prediction 3 named this as the outcome that would change the
project: *"SF-DRO beats SF-MES where MF-DRO lost. That would locate the failure
in the multi-fidelity extension rather than in the DRO architecture."* On
Hartmann it fired, and my pre-registered expectation (prediction 2: SF-DRO does
not beat SF-MES) is **refuted** here.

Two contrasts carry it:

1. **DRO beats its own-fidelity MES baseline 3-0 in single fidelity, having lost
   0-3 to it in multi-fidelity** (h57). Same architecture, same benchmark, same
   cost. The only difference is the fidelity machinery.
2. **SF-DRO (11.5%) beats MF-DRO (14.7%).** The multi-fidelity extension makes
   the method *worse* on this benchmark.

That is consistent with everything the diagnostics found and could not explain:
search quality was never the problem (stalled queries at the 88.7-99.9th
percentile of domain value), while the fidelity policy was measurably broken
(81%/4%/28% HF across seeds differing only by initial design; h58's floor
recovering 22% of the regret on the 2%-HF seed by replacing 176 LF queries with
19 HF ones).

### What this does NOT establish

- **One benchmark.** Currin and Borehole are pending and either could reverse the
  direction. On h57 the best method changed identity by benchmark.
- SF-DRO still loses to MF-MES (11.5% vs 8.5%), so it does not yet meet the
  north star's bar — "at least as good as the baselines" is not met by beating
  only the single-fidelity baseline.
- The compute ratio is unchanged: 19.4 minutes against SF-MES's ~2 seconds.
- n = 3, no p-values.

### H59 Currin complete (6/6): SF-DRO loses 0/3 — the Hartmann direction does not generalise

| arm | s44 | s46 | s48 | mean | median | rel |
|---|---|---|---|---|---|---|
| SF-DRO | 0.0012 | **0.1844** | 0.0001 | 0.0619 | 0.0012 | 0.4% |
| **SF-MES** | 0.0002 | 0.0009 | 0.0000 | **0.0004** | 0.0002 | **0.0%** |

**SF-DRO beats SF-MES on 0/3 seeds here**, having beaten it 3/3 on Hartmann. And
on Currin, MF-DRO (0.0%) beats SF-DRO (0.4%) — so "single fidelity is better
than multi" does not hold either.

So the Hartmann result does **not** generalise. Both of last tick's contrasts
reverse on this benchmark.

### The pattern that does hold across both: one catastrophic seed

SF-DRO's Currin mean (0.0619) is 50x its median (0.0012). Seeds 44 and 48 finish
at 0.0012 and 0.0001 — competitive with SF-MES — and seed 46 lands at 0.1844.
This is the same shape as MF-DRO's variance throughout: h45's regression head at
s.e. 0.1483 against scoring's 0.0475, with seeds 49 and 50 blowing up while the
other eight were fine.

**SF-DRO inherits the variance problem.** That matters for the multi-fidelity
story: if the catastrophic-seed behaviour survives removing the fidelity head
entirely, then the fidelity head is not its cause, and the h58 result (a 25%
floor recovering 22% of the regret on one seed) explains less than it appeared to.

### Where the north star actually stands, on 2 of 3 benchmarks

| | Currin 2D | Hartmann 6D |
|---|---|---|
| best baseline | **0.0%** (SF-MES) | **8.5%** (MF-MES) |
| SF-DRO | 0.4% | 11.5% |
| MF-DRO | 0.0% | 14.7% |

Neither DRO variant meets "at least as good as the baselines" on either
benchmark. SF-DRO is closer than MF-DRO on Hartmann and further on Currin.

**Correction to the previous entry.** I wrote that the Hartmann result "changes
the project's direction" and located the failure in the multi-fidelity
extension. On two benchmarks that reading is not supported: the direction is
benchmark-dependent, exactly as h57's baselines were, and the variance that
actually costs DRO its results is present with or without multi-fidelity. The
caveat I attached ("one benchmark of three, either could reverse the direction")
was the right one and it fired immediately.

---

# H57 + H58 + H59 ALL COMPLETE (66 runs) — the north star is not met by either DRO variant

**CONFIRMATORY.** Budget 200 post-init, seeds 44/46/48, shared initial design,
pinned commits, hash in every result file.

## Relative simple regret (regret / f(x*))

| method | Currin 2D | Hartmann 6D | Borehole 8D |
|---|---|---|---|
| SF-DRO | 0.4% | 11.5% | 15.1% |
| MF-DRO | **0.0%** | 14.7% | 23.7% |
| SF-MES | **0.0%** | 21.4% | 13.3% |
| MF-MES | 0.6% | **8.5%** | 11.3% |
| MF-MI-Greedy | 0.2% | 23.9% | **8.3%** |
| MF-GP-UCB *(all-LF floor)* | 10.0% | 45.3% | 44.1% |

**Best baseline changes identity every time**: SF-MES on Currin, MF-MES on
Hartmann, MI-Greedy on Borehole.

## The three results

**1. NEITHER DRO variant meets the bar, on any benchmark.**

| | SF-DRO | MF-DRO | best baseline |
|---|---|---|---|
| Currin | 0.4% | 0.0% | 0.0% (SF-MES) |
| Hartmann | 11.5% | 14.7% | 8.5% (MF-MES) |
| Borehole | 15.1% | 23.7% | 8.3% (MI-Greedy) |

MF-DRO ties on Currin only because the benchmark is saturated — every
non-degenerate method is inside 0.6%.

**2. Dropping multi-fidelity HELPS DRO, on 2 of 3 benchmarks.** SF-DRO beats
MF-DRO 3/3 on Borehole (15.1% vs 23.7%) and 2/3 on Hartmann (11.5% vs 14.7%);
MF-DRO wins Currin 2/3, where both are at the saturation floor. So the
multi-fidelity extension is a net negative for this architecture — but removing
it is not sufficient to clear the baselines.

**3. DRO beats its own-fidelity MES baseline on exactly one benchmark.**
SF-DRO vs SF-MES: 3/3 Hartmann, 0/3 Currin, 0/3 Borehole. The Hartmann win does
not generalise, and Hartmann is the one benchmark whose LF optimum (4.0019)
exceeds its HF optimum (3.3224) — see the failure-mode note.

## Where that leaves the north star

"A novel method based on SF-DRO that is at least as good as the baselines" is
**not reached by SF-DRO as it stands**. The gap is 0.4pp on Currin, 3.0pp on
Hartmann, 6.8pp on Borehole — and the compute ratio is 19-67 minutes per run
against MI-Greedy's 0.4-0.7.

What the 66 runs rule out as the thing to fix:

- **not the search** — stalled queries sit at the 88.7-99.9th percentile of
  domain value, incumbents at 98.7-100th
- **not query freeze** — `distinct == n_queries` in all 9 MF-DRO cells
- **not distance from the optimum** — retracted; on Hartmann the best value
  within 0.3 of x* equals the best beyond 1.0 away
- **not (only) the fidelity head** — SF-DRO has none and still shows the
  catastrophic-seed variance (Currin mean 0.0619 vs median 0.0012, 50x)

What remains, and is common to both variants: **one seed in three blows up**,
and the plateau sits a few percent above the best acquisition method at 100x the
cost. Any novel method has to attack the variance, not the fidelity handling.

### NULL: DRO's catastrophic seeds do not "die early" (exploratory, no new compute)

Direction recorded after h57/h58/h59 was *attack the variance*. First test on
existing data: do the worst seeds stop improving earlier than the healthy ones?

Labelling the worst seed in each (benchmark, method) group across all 18 DRO
runs and comparing the iteration of the last incumbent improvement, as a
fraction of the run:

| group | n | mean | median |
|---|---|---|---|
| WORST seed | 6 | 49% | 52% |
| other seeds | 12 | 62% | 73% |

Directionally consistent, but the distributions overlap almost completely. The
WORST seeds' individual values are **62%, 41%, 95%, 0%, 0%, 96%** — they span
the entire range. Improvement counts are 5.8 vs 7.1, also weak.

**Flaw in my own labelling, stated rather than buried:** on Currin all three
MF-DRO seeds finish at 0.0% relative regret, so "worst" there is a ranking of
noise. Excluding Currin does not rescue the signal — Hartmann MF-DRO seed 46 is
*healthy* at 8.7% with its last improvement at 22%, while Hartmann SF-DRO seed
46 is *worst* at 13.3% with its last improvement at 96%. The two orderings are
uncorrelated.

**Conclusion: early stalling does not explain the variance.** That rules out the
most obvious mechanism and, with it, the interventions that follow from it
(restart-on-stall, early-stopping detection, longer warmup).

One observation that survives and is worth a protocol rather than a guess:
**Hartmann MF-DRO seed 44 spent its entire 200-cost budget in 31 queries at 81%
HF and recorded ZERO improvements.** Its failure is not stalling but
*over-spending* — at c_H=8 it bought 25 HF queries and none beat the initial
design's incumbent. That is the mirror image of seed 46 (2% HF, 179 queries,
also near-zero improvement). Both extremes of the fidelity split fail; the
middle (seed 48, 28% HF) does best. Suggestive, n=3, not tested.

### RETRACTION: "Hartmann's LF is a more attractive objective" — withdrawn

Prompted by a direct challenge: the MES reward and the switching condition both
score LF by information gain about the **HF** optimum, so a high LF value cannot
be intrinsically attractive. Checked, and the challenge is correct.

1. `_compute_mes_lf_vectorized` takes `y_star_arr` = "shared **HF** Thompson
   samples" and works through `rho`, `mu_H`, `var_delta`. LF's own optimum enters
   nothing. Real-inference fidelity comes from the DT's `fidelity_head`, not MES.
2. corr(f_LF, f_HF) = **+0.925** domain-wide. The true HF value at seed 46's 176
   LF-queried points has median **3.0287 = 91% of f(x*)**. LF is an excellent
   surrogate on Hartmann; the policy found the good region cheaply.
3. **Regret is monotone in HF fraction, opposite to my claim**: seed 46 (2% HF,
   179 queries) 8.7%; seed 48 (28%) 12.7%; seed 44 (81%, 31 queries) 22.7%.
   The seed I called pathological is the best one.

The 166/176 statistic is real (uniform sampling puts 0.1% of LF values above
f(x*)_HF, so 94.3% is a 900x concentration) but it measures that the policy
LOCATED the good region, not that it was lured into a bad one.

Still standing: the 81%/4%/28% fidelity spread across seeds differing only by
initial design, and h58's floor improving seed 46 by 22%. Withdrawn: that
LF-heaviness is Hartmann's failure mode. On this evidence LF-heaviness is
Hartmann's *best* strategy, and MF-DRO's Hartmann loss needs another explanation.

**Lesson 21**: an extreme statistic ("94% of LF queries above the HF optimum")
can be real, significant against chance, and still support the opposite reading.
Before attributing it to a mechanism, check whether the mechanism can act at all
— here the scoring rule provably cannot see LF's value.

---

## CONFOUND: "SF-DRO beats MF-DRO" is not a multi-vs-single-fidelity result

Prompted by the question: on Borehole MF-DRO queries 99-100% HF post-init, so it
should behave like SF-DRO — why is it worse (23.7% vs 15.1%)?

Checking the two configurations, they differ in **four** ways at once, and
fidelity is the *least* of them on this benchmark:

| | MF-DRO (h57) | SF-DRO (h59) |
|---|---|---|
| post-init fidelity | 99-100% HF | 100% HF |
| **initial design** | 10 HF + **20 LF** | 10 HF, no LF |
| **surrogate** | `KennedyOHaganGP` ensemble (fits rho + discrepancy GP) | BoTorch `SingleTaskGP` ensemble |
| **rollout teacher** | joint MF-MES (`rollout_reward="mes_entropy"`) | **EI** (`rollout_acq_function="ei"`) |
| **reward / RTG** | MES-entropy reward | `use_mes_reward=False`, `rtg_schema="fixed"` |

So the h59 comparison is **not** "the same method with and without multi-fidelity".
It is one method against another that differs in surrogate, teacher, reward
schema and initial design. On Borehole the fidelity behaviour is essentially
identical between them, which means **the 8.6pp gap is attributable to the other
three differences, not to multi-fidelity.**

### What this retracts

The claim "dropping multi-fidelity HELPS DRO — SF-DRO beats MF-DRO 3/3 on
Borehole, 2/3 on Hartmann" is **confounded**. The direction of the measurement
stands; the *causal attribution to multi-fidelity* does not. That attribution
appears in the h57+h58+h59 summary and in `research-state.yaml`'s
`established:` block, and both are now wrong as stated.

### The decomposition this needs

Four candidate causes, separable with existing machinery, each a one-variable
change from MF-DRO:

1. **Surrogate**: KO-GP vs plain GP ensemble on HF only. h48 already found "the
   surrogate init, not the acquisition, is what survives correction" — this is
   the lead with prior support.
2. **Teacher**: MF-MES vs EI in the rollouts.
3. **Reward/RTG**: `mes_entropy` vs `fixed`.
4. **Initial design**: whether the 20 LF init points help or hurt the KO fit.

On Borehole the KO model must estimate rho and a discrepancy GP from 20 LF + 10
HF points while the plain GP fits 10 HF points directly. If rho is poorly
identified there, the KO posterior is worse than the simpler model — which would
explain a gap with the fidelity *behaviour* held constant, exactly what is
observed.

**Lesson 22**: before attributing a gap between two named methods to the
dimension in their names, enumerate every configuration difference. Here the
names said "multi-fidelity vs single-fidelity" and the configs differed in four
places, of which fidelity was the one that provably did not act.

---

## Borehole: three geometric explanations proposed, three failed. What is left.

Prompted by "if MF-DRO is all-HF on Borehole it should behave like SF-DRO — what
makes it worse?" Each answer below was measured, and each superseded the last.

### Attempt 1 — "an uninformative LF corrupts the KO surrogate". FALSE.

corr(f_LF, f_HF) on Borehole is **1.000** (Pearson and Spearman, 8000 Sobol
points). The LF is a perfect surrogate. Nothing to corrupt.

### Attempt 2 — "MF-DRO fails to refine locally". FALSE, and inverted.

Distance from HF query to the current incumbent, first half vs second half:

| method | 1st half | 2nd half | contraction |
|---|---|---|---|
| **MF-DRO** | 0.1208 | 0.0822 | **0.68x** |
| MF-MES | 0.1539 | 0.1695 | 1.10x |
| MF-MI-Greedy | 0.9977 | 1.0811 | 1.08x |

MF-DRO is the **only** method that refines, and it is the worst. MI-Greedy
samples at ~1.0 from its own incumbent — essentially global — and wins.

### Attempt 3 — "boundary aversion; x* is a corner it cannot reach". UNSUPPORTED.

The premise is true: x* sits **on the boundary in 7 of 8 dimensions**, and
MF-DRO proposes near-bound coordinates 7.4% of the time against MF-MES's 36.4%
and MI-Greedy's 32.8% (uniform expectation 10.0%). On Hartmann it is 1.8%, i.e.
actively boundary-averse.

But the conclusion does not follow. Distance from HF queries to x*:

| method | mean d* | min d* | d*(best point) | rel regret |
|---|---|---|---|---|
| **MF-DRO** | **0.900** | 0.786 | 0.819 | **23.7%** |
| MF-MES | 1.030 | 0.888 | 0.961 | 11.3% |
| MF-MI-Greedy | 1.175 | 0.406 | 0.805 | **8.3%** |
| uniform random | 1.539 | 0.490 | — | — |

**MF-DRO's queries are the CLOSEST to x* of any method and it has the worst
regret** — the ordering is exactly inverted. All three beat the uniform
reference, so all three locate the region. It cannot be "cannot reach the
optimum" when it is the nearest.

This is the same trap as the retracted Hartmann distance claim, and it was
already measured: `corr(d*, f) = -0.587` on Borehole, and 6000 uniform samples
contain **zero** points within 0.3 of x*. Distance is not a value proxy in 8D.

### What IS established on Borehole

1. MF-DRO plateaus at best-HF **236.18** where MI-Greedy reaches **283.97** on
   the **same 100 HF evaluations**.
2. Calibrated against random search: MF-DRO's 99 queries buy what ~1000 random
   draws buy (240.32); MI-Greedy's 100 beat what 20,000 random draws buy
   (266.08). So MF-DRO optimises — roughly 10x better than random — but the
   winners are >200x.
3. Fidelity is inert (99-100% HF), so nothing fidelity-related can explain it.
4. Its queries sit at the 99.7-99.9th percentile of domain value.

### Lesson 23

Three separate geometric accounts of *where* MF-DRO searched — LF quality, local
refinement, boundary reach — each looked compelling and each was refuted by the
next measurement. In every case the refutation came from checking the quantity
against **value** rather than against geometry. On these benchmarks, in 6-8
dimensions, spatial descriptions of a search do not predict its outcome. Stop
proposing them; measure value directly.

The mechanism for Borehole is **unresolved**, and h60's three arms (LF init,
reward, teacher) do not touch the action head. If they are null, an action-head
experiment is the next protocol.

### Borehole: the gap opens in the first ~20 HF queries and never closes

**Value-level, per lesson 23** — indexed by HF query number rather than cost, so
search quality per high-fidelity evaluation is isolated from fidelity spend.
MF-DRO and MI-Greedy share the identical 10-point HF initial design and both
start at 72.07.

| HF query # | MF-DRO | MF-MI-Greedy | gap |
|---|---|---|---|
| 0 (shared init) | 72.07 | 72.07 | 0 |
| 10 | 173.07 | 219.71 | 47 |
| **20** | 190.38 | **264.41** | **74** |
| 30 | 201.29 | 265.97 | 65 |
| 50 | 226.76 | 277.47 | 51 |
| 109 | 236.18 | 283.97 | 48 |

**MI-Greedy reaches 264.41 within 20 HF queries; MF-DRO does not reach it in
109.** The gap is widest at query 20 and never closes — MF-DRO's later queries do
improve, but only enough to partly catch up to where MI-Greedy already was.

This reframes the Borehole failure. It is **not** a late plateau: it is a slow
start that is never recovered. Whatever MF-DRO does in its first ~20 HF
evaluations is worth roughly a third less than what MI-Greedy does with the same
budget from the same starting point.

**Excluded from the comparison**: MF-MES's curve on this index is an artifact.
Its trace was repaired for the `is_init` bug (h57), so the initial design is
absent from the file and its index 0 is already the first optimization query
(199.01, not 72.07). Its regret numbers are unaffected; only this per-query
indexing is unusable for it.

Consistent with the random-search calibration: MF-DRO's 99 post-init HF queries
buy what ~1000 random draws buy, MI-Greedy's buy more than 20,000 do. The
difference is concentrated early.

### The early-HF deficit is BOREHOLE-SPECIFIC, not a general DRO cold start

Same value-level indexing (best-HF-so-far by HF query number, shared 10-point HF
init verified identical at query 0 on all three benchmarks):

| benchmark | q10 | q20 | direction |
|---|---|---|---|
| **Borehole 8D** | **-15.1%** | **-23.9%** | MF-DRO **behind** MI-Greedy |
| Hartmann 6D | — | — | MF-DRO **ahead** +18.3% (at q8, its last shared index) |
| Currin 2D | +6.0% | +5.8% | MF-DRO **ahead** |

So MF-DRO's search is **competitive or better per HF evaluation on two of three
benchmarks**, and specifically deficient on Borehole. The previous entry's
framing ("a slow start that is never recovered") is correct for Borehole and
does **not** generalise — it is not a cold-start property of the architecture.

Worth restating what this implies for the headline table: MF-DRO **beats
MI-Greedy** on Hartmann (14.7% vs 23.9%) and Currin (0.0% vs 0.2%). Its losses
are to **MF-MES** on Hartmann, and to both baselines on Borehole. Borehole is the
only benchmark where it is beaten by everything.

Part of the per-HF-query advantage on Hartmann and Currin is legitimate
multi-fidelity benefit — MF-DRO also holds LF observations that MI-Greedy does
not. That is the mechanism multi-fidelity is supposed to provide, and on those
two benchmarks it works.

### H60 partial: REWARD arm complete, and it is a clean null

**CONFIRMATORY.** Borehole, seeds 44/46/48, one flag changed from h57's MF-DRO.
NOLFINIT (2/3) and TEACHER (0/3) still running and **withheld**.

| arm | s44 | s46 | s48 | mean | rel | vs BASE |
|---|---|---|---|---|---|---|
| BASE (h57 reuse) | 75.64 | 76.59 | 67.97 | 73.40 | 23.7% | — |
| **REWARD** (`rollout_reward="improvement"`) | 79.79 | 80.56 | 69.33 | 76.56 | 24.7% | **beats BASE 0/3** |

Swapping the MES-entropy reward for improvement-based reward makes it slightly
worse on every seed. The reward schema is **not** what separates MF-DRO from
SF-DRO on Borehole. One of the four confounded factors is now excluded.

### H60: NOLFINIT is a NULL — the pre-registered favourite fails

**CONFIRMATORY.** Borehole, seeds 44/46/48, one flag from h57's MF-DRO.
TEACHER (0/3) still running and withheld.

| arm | s44 | s46 | s48 | mean | rel | beats BASE |
|---|---|---|---|---|---|---|
| BASE (h57 reuse) | 75.64 | 76.59 | 67.97 | 73.40 | 23.7% | — |
| **NOLFINIT** (`initial_lf=0`) | 80.38 | 88.03 | **57.88** | 75.43 | 24.4% | **1/3** |
| REWARD (`rollout_reward="improvement"`) | 79.79 | 80.56 | 69.33 | 76.56 | 24.7% | 0/3 |

H60's locked prediction 2 named NOLFINIT as *the arm that moves*, reasoning that
the KO model must identify rho and a discrepancy GP from 20 LF + 10 HF points
where SF-DRO's plain GP fits 10 HF directly. **It does not move** — mean slightly
worse, 1 of 3 seeds better. The prediction is refuted.

Note the spread: NOLFINIT's seed 48 (57.88) is the best single Borehole cell any
MF-DRO variant has produced, while its seed 46 (88.03) is among the worst. Same
flag, same benchmark, opposite outcomes — the variance signature again, and a
reason not to read the 1/3 as a weak positive.

**Two of the three flag-separable factors are now excluded**: the reward schema
and the LF initial design. Neither is what separates MF-DRO from SF-DRO on
Borehole. TEACHER (`rollout_policy="thompson"`) is the last flag.

If TEACHER is also null, H60's locked prediction 3 fires: the gap is
attributable to **the surrogate class itself** (KO-GP vs `SingleTaskGP`), which
is not a flag and needs a code-level swap. That was registered in advance as an
informative outcome rather than a failed experiment, and it would be the first
time this project has narrowed a cause by elimination rather than by proposing
a mechanism and then retracting it.

---

## H60 COMPLETE (9/9): two factors excluded, the teacher is load-bearing, the surrogate remains

**CONFIRMATORY.** Borehole 8D, seeds 44/46/48, budget 200, one flag changed per
arm from h57's MF-DRO. BASE reuses h57's cells (policy code byte-identical).

| arm | s44 | s46 | s48 | mean | rel | beats BASE | HF/LF per seed |
|---|---|---|---|---|---|---|---|
| BASE | 75.64 | 76.59 | 67.97 | 73.40 | 23.7% | — | 100/1 100/1 99/3 |
| NOLFINIT | 80.38 | 88.03 | 57.88 | 75.43 | 24.4% | 1/3 | 75/50 100/0 93/15 |
| REWARD | 79.79 | 80.56 | 69.33 | 76.56 | 24.7% | 0/3 | 97/7 99/2 96/8 |
| **TEACHER** | 138.82 | 113.30 | 154.35 | **135.49** | **43.8%** | 0/3 | **2/196 2/196 5/190** |

### The teacher is load-bearing, and the failure mode is self-explaining

Swapping the joint-MF-MES rollout teacher for `rollout_policy="thompson"`
**collapses the fidelity head to ~99% LF** (2, 2 and 5 HF queries out of ~198).
Since the incumbent moves only on HF, it barely moves: `n_improvements` = 0, 0, 2.

The regret then lands at *exactly* MF-GP-UCB's value on two of three seeds —
TEACHER 138.8226 / 113.2965 vs MF-GP-UCB 138.8226 / 113.2965. That identity is
not a coincidence: MF-GP-UCB is structurally all-LF, so both arms finish at the
best point of the **shared initial design** and must report the same regret.
Seed 48 differs (154.35 vs 157.41) because its 5 HF queries produced 2
improvements.

**Consequence beyond this experiment**: the fidelity head's behaviour is
*downstream of the rollout teacher*. Change the teacher and the fidelity policy
collapses. That is a candidate explanation for the 81%/4%/28% HF spread across
Hartmann seeds — a policy this sensitive to its training signal is not a stable
policy.

### What the decomposition actually settles

| factor | verdict |
|---|---|
| reward schema | **EXCLUDED** — 0/3, slightly worse |
| LF initial design | **EXCLUDED** — 1/3, slightly worse |
| rollout teacher | **NOT excluded** — load-bearing (20pp swing), but the EI variant was untestable |
| surrogate class | **NOT excluded** — not a flag, untested |

H60's locked prediction 3 said a null on all three would leave the surrogate.
That is *not* what happened: TEACHER moved regret by 20 percentage points. So
**two candidates remain**, not one.

Critically, the teacher's effect is in the **wrong direction** for explaining
SF-DRO > MF-DRO: MES → thompson makes MF-DRO much worse, while SF-DRO uses EI
and is *better*. mf_dro offers mes/thompson/random only, so MES → EI could not
be tested — a limitation stated in the protocol before running, not discovered
after. The teacher cannot be ruled out; it was tested against the wrong
alternative.

**Next**: EI as an mf_dro rollout teacher (code change), or the surrogate swap.
Both are code-level, not flags.

### H61 liveness check at 12/~100 queries — intervention confirmed active, direction NOT read

**Operational check only, not a result.** Ran to decide whether REFINE is doing
anything before letting it consume another ~5 hours.

| arm | n post-init | query spread |
|---|---|---|
| REFINE | 12 | **0.2907** |
| BASE (same depth) | 12 | 0.2534 |

The arms diverge, so the config knob is live and the run is worth completing.

**A prediction inside the mechanism already failed.** I expected a local
refinement pass to *concentrate* the teacher's demonstrations — smaller query
spread than BASE. The spread is **larger** (0.2907 vs 0.2534). Whatever
refinement is doing here, it is not tightening the search. That matters because
H61's locked HARMFUL branch was justified by exactly that concentration
("a sharper teacher concentrates its demonstrations, reducing the diversity the
DT trains on"). The premise of that branch is not holding, so if HARMFUL fires
it will need a different explanation.

**Regret at this depth is deliberately not reported.** 12 of ~100 queries, n=3.
That is precisely the subset-with-the-rest-coming situation lesson 19 covers,
and this project has produced five instances of that failure — one of which
changed a shipped default. The number exists in the checkpoints; it will be read
when all six cells finish.

### ETA correction: REFINE is the slow arm, not POOL600

Measured at 37 min: POOL600 has completed 25 queries, REFINE 12.

| arm | min/query | vs BASE (0.92) | projected total |
|---|---|---|---|
| POOL600 | 1.48 | 1.6x | ~148 min |
| REFINE | 3.08 | 3.3x | ~308 min |

The amendment predicted REFINE ~1.5x and POOL600 ~3x — **inverted**. Cause:
`compute_joint_mf_mes` has per-*call* overhead (Thompson sampling, posterior
evaluation, quadrature setup), not just per-candidate cost. REFINE makes **two**
calls per rollout step (200 broad, then 300 union) where POOL600 makes one with
600. Two calls cost more than 600 candidates in one.

Recorded because this project has a documented ETA-miss record and the cause here
is structural — cost scales with the number of acquisition *calls*, not only the
candidate count.

---

## H62 GATE PASSED — SF-DRO diagnosed for the first time

All 9 cells reproduce h59 to **0.00e+00** (exact, not merely within the 1e-9
tolerance). Traces present in 9/9. h59's regret conclusions stand unchanged;
H62 contributes instrumentation only, as its protocol stated.

### SF-DRO diagnostics (single fidelity — every query is HF, so the incumbent can move on any of them)

| benchmark | seed | nq | distinct | Q-FREEZE | improv | stall | regret | q pctile | inc pctile |
|---|---|---|---|---|---|---|---|---|---|
| Currin 2D | 44 | 71 | 71 | no | 11 | 16 | 0.0012 | 92.1% | 100.0% |
| **Currin 2D** | **46** | 71 | 71 | **no** | **3** | **65** | **0.1844** | **89.0%** | **99.1%** |
| Currin 2D | 48 | 71 | 71 | no | 5 | 11 | 0.0001 | 97.8% | 100.0% |
| Hartmann 6D | 44 | 31 | 31 | no | 7 | 12 | 0.3815 | 77.9% | 100.0% |
| Hartmann 6D | 46 | 31 | 31 | no | 9 | 0 | 0.4406 | 73.8% | 100.0% |
| Hartmann 6D | 48 | 31 | 31 | no | 11 | 0 | 0.3233 | 84.7% | 100.0% |
| Borehole 8D | 44 | 110 | 110 | no | 12 | 58 | 50.2414 | 99.9% | 100.0% |
| Borehole 8D | 46 | 110 | 110 | no | 16 | 9 | 44.0491 | 99.7% | 100.0% |
| Borehole 8D | 48 | 110 | 110 | no | 8 | 19 | 46.0989 | 99.9% | 100.0% |

### The catastrophic seed, finally diagnosed

Currin seed 46 (0.1844 against a 0.0012 median) is the clearest
catastrophic-seed instance in a fidelity-free setting. Its trace shows:

- **No query freeze**: 71/71 distinct. The same-point pathology is absent, as it
  has been in every MF-DRO cell too.
- **3 improvements, then a 65-query stall** — it stops improving almost
  immediately and never recovers across the remaining 92% of the run.
- **Its incumbent is at the 99.1st percentile of domain value**, where the two
  healthy Currin seeds reach 100.0%.
- **Its queries sit at the 89.0th percentile** versus 92.1% and 97.8% for the
  healthy seeds — searching *slightly* worse regions, not catastrophically worse.

So the blow-up is **not** a distinct failure mechanism. It is the same search,
marginally less well aimed, that happens to miss the last 0.9 percentile of
value — and on Currin that 0.9 percentile is the difference between 0.0001 and
0.1844 regret, because the benchmark saturates so steeply near its optimum.

**This reframes the "catastrophic seed" story that has run through this project
since h45.** It is not a policy collapse. It is a benchmark whose regret is
extremely sensitive near the optimum, amplifying a small difference in search
quality into a 1500x difference in reported regret. The same 0.9-percentile
shortfall on Borehole would be invisible.

**Note the important negative**: Borehole's SF-DRO cells query at the
**99.7-99.9th percentile** and still finish 14-16% short. The value-percentile
diagnostic cannot discriminate there — every method is at the top of the
distribution and the differences live in the last fraction of a percentile.

### CORRECTION: Currin is the FLATTEST benchmark near its optimum, not the steepest

Last entry explained Currin seed 46's blow-up as the benchmark "saturating so
steeply near its optimum" that a 0.9-percentile shortfall becomes a 1500x regret
difference. **The steepness half of that is backwards.** Relative regret incurred
by landing at a given value percentile:

| benchmark | p50 | p90 | p99 | p99.9 | p100 | p99 -> p99.9 drop |
|---|---|---|---|---|---|---|
| **Currin 2D** | 48.5% | 17.7% | 1.5% | 0.1% | 0.0% | **1.46 pp** |
| Hartmann 6D | 96.9% | 78.2% | 44.4% | 20.6% | 5.8% | 23.79 pp |
| Borehole 8D | 77.6% | 53.9% | 36.2% | 23.8% | 16.2% | 12.36 pp |

Currin is the **flattest** near the top — a percentile shortfall costs 1.46 pp
there against Hartmann's 23.79. The large *ratio* comes from the denominator,
not from steepness: the healthy Currin seeds finish at 0.0001-0.0012 absolute
regret, so any non-trivial value divided by that is an enormous multiple.

**Corrected statement.** Seed 46's 0.1844 is **1.34% relative regret**; the
healthy seeds are at 0.009% and 0.0007%. The absolute spread is ~1.3 percentage
points — the whole p99-to-p100 range on Currin. So:

- The **conclusion holds**: the blow-up is not a policy collapse. Its search is
  only marginally worse (89.0th percentile vs 92.1st and 97.8th).
- The **mechanism was wrong**: it is not steep saturation. It is that Currin
  compresses every competent method into the last 1.5 pp of regret, which makes
  regret *ratios* on that benchmark unstable and close to meaningless.

This is the same point as "Currin does not discriminate" (h57), arrived at from
the other direction, and it is a stronger reason to distrust Currin ratios than
the saturation story I gave.

**Consequence for the variance narrative.** I have twice told the user that
catastrophic-seed variance is the failure mode to attack. On Currin that
variance is a 1.3 pp effect inflated by a near-zero denominator. The variance
that matters, if any, must be demonstrated on Hartmann or Borehole where a
percentile shortfall actually costs regret — not on Currin.

### H61 POOL600 complete: widening the teacher's pool improves Borehole 3/3

**CONFIRMATORY for this arm.** The locked prediction it participates in
(prediction 2: "REFINE beats BASE on >= 2/3, POOL600 lands between them") is
**unresolved** — REFINE is 0/3 and withheld.

| arm | s44 | s46 | s48 | mean | rel | HF/LF per seed |
|---|---|---|---|---|---|---|
| BASE (200, argmax) | 75.64 | 76.59 | 67.97 | 73.40 | 23.7% | 100/1 100/1 99/3 |
| **POOL600 (600, argmax)** | **65.07** | **70.48** | **45.15** | **60.23** | **19.5%** | 89/22 98/5 98/4 |

Better on **3/3** seeds. Mean relative regret falls 23.7% -> 19.5%, and seed 48
improves 33% (67.97 -> 45.15).

**This is the first intervention in this project that has improved Borehole.**
h58's floor, h60's REWARD, h60's NOLFINIT and h60's TEACHER were null, null,
null and strongly negative respectively.

### Why this matters, and what it does not yet establish

It supports the h61 hypothesis directly: MF-DRO's teacher argmaxes over 200
uniform points with no refinement, where SF-DRO's does 1000 + local refinement.
h47-variant-d had measured a 200-point pool finding an acquisition value 4.3x
worse than a 4000-point one and not saturated there; this is the first evidence
that the gap costs *regret*, not just acquisition value.

Not established:

- **n = 3, one benchmark.** No p-value.
- The pool change also shifted the fidelity mix slightly (100/1 -> 89/22 on seed
  44), so it is not a perfectly clean single variable — though far less severe
  than h63's 99% -> 4% inversion.
- **19.5% still loses to every baseline on Borehole** (MI-Greedy 8.3%, MF-MES
  11.3%). Closing 4.2 points of a 15.4-point gap is progress toward the north
  star, not arrival at it.
- Whether a *larger* pool keeps helping, or 600 is near a plateau, is untested.
  h47-variant-d's curve was still rising at 4000.

The natural follow-up if REFINE also helps: a pool sweep (200/600/2000) to find
where the acquisition-quality return flattens, since that determines whether this
is a fix or merely a direction.

---

## OUTER LOOP: is this converging on the north star?

Honest assessment after h57-h64 (~100 runs). The north star is a novel
SF-DRO-based method **at least as good as the baselines**.

| benchmark | MF-DRO | best baseline | gap | best intervention so far |
|---|---|---|---|---|
| Currin 2D | 0.0% | 0.2% | — | n/a, saturated (all methods inside 0.6%) |
| Hartmann 6D | 14.7% | 8.5% | **6.2 pp** | none yet (h64 running, **predicted null**) |
| Borehole 8D | 23.7% | 8.3% | **15.4 pp** | POOL600: 23.7 -> 19.5, **closed 4.2 of 15.4** |

### What the pattern of interventions says

Nine interventions have been run. Their effects:

| intervention | effect |
|---|---|
| h58 HF floor at inference | +22% on one seed, no-op on 5 of 6 cells |
| h60 reward schema | null (0/3) |
| h60 LF initial design | null (1/3) |
| h60 teacher identity (MES -> thompson) | **-20 pp** (strongly harmful) |
| h61 POOL600 | **+4.2 pp** on Borehole, 3/3 |
| h61 REFINE | pending |
| h63 RHOTRUE | pending, four-way confounded |
| h64 POOL600 elsewhere | pending, predicted null |

**One intervention in nine has produced a durable gain, and it closes 27% of one
benchmark's gap.** The only large effect found was *negative*. That is the shape
of a method whose deficit is not concentrated in any single component this
project has been able to name.

### The honest read

Incremental component fixes are moving single-digit points against double-digit
gaps, while the compute ratio stays at **120-230x** against MI-Greedy. Even if
POOL600's mechanism generalised perfectly and stacked with REFINE, Borehole would
land near 15% against a baseline at 8.3%.

So: **the north star is not reachable by continuing to repair MF-DRO's
components.** The evidence supports a different framing of what this project has
produced — a well-controlled negative result with an unusually complete
elimination record:

- not fidelity allocation (99-100% HF on Borehole; excluded by construction)
- not the reward schema, not the LF initial design (h60)
- not search location (queries at the 99.7-99.9th percentile of domain value)
- not query freeze (`distinct == n_queries` in every cell measured)
- not proximity to the optimum (retracted, twice, with the measurement)
- not the catastrophic-seed variance (largely a Currin denominator artifact)
- partially the teacher's acquisition optimisation (h61, +4.2 pp of 15.4)

### What would change the answer

A method that wins needs a lever the size of the gap, not the size of the
components. Two candidates the evidence actually supports:

1. **Drop the distillation.** h48 found MF-MES and MF-DRO indistinguishable at
   n=10 with the surrogate matched, and h60 found the teacher load-bearing at
   20 pp. If the teacher is what carries the performance, the DT is overhead —
   and the honest contribution is that finding, not a repaired DT.
2. **Change what the DT is asked to do.** Every result here measures the DT
   imitating a myopic acquisition. Nothing has tested it doing something the
   acquisition cannot — which was the original non-myopia claim, and remains the
   only untested route to beating a well-optimised acquisition.

### H61 COMPLETE: first locked prediction MET — and refinement collapses the variance

**CONFIRMATORY.** Borehole 8D, seeds 44/46/48, budget 200, one variable per arm.

| arm | s44 | s46 | s48 | mean | rel | wins | sd | spread |
|---|---|---|---|---|---|---|---|---|
| BASE (200, argmax) | 75.64 | 76.59 | 67.97 | 73.40 | 23.7% | — | 4.75 | 8.62 |
| POOL600 (600, argmax) | 65.07 | 70.48 | 45.15 | 60.23 | 19.5% | 3/3 | 13.40 | 25.33 |
| **REFINE (200+100 local)** | **58.79** | **59.09** | **61.36** | **59.75** | **19.3%** | **3/3** | **1.41** | **2.57** |

**Locked prediction 2 MET as stated**: REFINE beats BASE on >=2/3 (it is 3/3),
and POOL600 lands between BASE and REFINE (73.40 > 60.23 > 59.75). This is the
first locked prediction in this project to be met rather than refuted, amended
or superseded.

### The variance result is the finding, not the mean

REFINE and POOL600 have nearly identical means (59.75 vs 60.23) but **REFINE's
spread is 2.57 against POOL600's 25.33** — roughly 10x tighter, and 3x tighter
than BASE's 8.62. Its three seeds land within 2.6 units of each other.

Variance has been the one persistent failure signature in this project: h45's
regression head at s.e. 0.1483 against scoring's 0.0475, the catastrophic seeds
at h45 49/50 and h59 Currin 46, and h60's NOLFINIT producing both the best and
among the worst Borehole cells from the same flag. **A local refinement pass in
the rollout teacher is the first intervention to compress it.**

Mechanistically this is not what I predicted. The liveness check found refinement
*widening* the query spread (0.2907 vs BASE's 0.2534), so it is not concentrating
the search. Yet it stabilises the outcome. Those are compatible — a teacher that
reliably finds a good local optimum from wherever it starts would produce
*diverse* queries and *consistent* results — but that is a description, not a
tested mechanism, and it should not be stated as one.

### What it does not achieve

19.3% still **loses to every baseline on Borehole** (MI-Greedy 8.3%, MF-MES
11.3%). It closes 4.4 of the 15.4-point gap — 29%. n = 3, so the sd figures are
three-point estimates and no p-value is computable. And h64's pre-registered
prediction says this will **not** transfer to Hartmann, where widening the pool
buys zero acquisition value (1.00x at N=600).

### H63 Borehole arm complete — VERDICT WITHHELD, and the reason matters

**Data, not a verdict.** The Hartmann control is at 1/3 and h63's protocol states
that the discriminating signal is a **larger Borehole gain than Hartmann's**, not
a Borehole win on its own. Reporting this as support would be reporting exactly
what the protocol pre-declared insufficient.

| arm | s44 | s46 | s48 | mean | rel | sd | HF/LF per seed |
|---|---|---|---|---|---|---|---|
| BASE | 75.64 | 76.59 | 67.97 | 73.40 | 23.7% | 4.73 | 100/1 100/1 99/3 |
| RHOTRUE | 73.91 | 61.64 | 67.23 | 67.60 | 21.8% | 6.14 | **10/180 32/137 47/106** |

RHOTRUE beats BASE 3/3, mean 23.7% -> 21.8%.

### Three reasons this is weaker evidence than it looks

1. **It is the smallest of the three Borehole gains.** h61's arms, on the same
   benchmark and cells: POOL600 19.5%, REFINE 19.3%, RHOTRUE 21.8%. Pinning rho
   to its true value helps *less* than widening the teacher's candidate pool.
2. **The fidelity mix inverted**, exactly as the pre-registered fourth confound
   said it would: 99-100% HF -> 6-31% HF. So this arm is not "BASE with a
   corrected rho" — it is a different fidelity policy as well, on a benchmark
   where corr(f_LF, f_HF) = 1.000 makes cheap LF queries unusually valuable.
   The gain is ambiguous between the surrogate correction and the budget shift.
3. **Variance went the wrong way** for a "correcting misspecification" story:
   sd 4.73 -> 6.14. Contrast REFINE, which cut it to 1.41 on the same cells.

### What still discriminates

Only the Hartmann contrast. Confounds 2-4 (adaptivity, ensemble rho diversity,
fidelity mix) apply to **both** benchmarks; only Borehole's true slope (1.2566)
is outside the representable range, Hartmann's (0.9792) is inside. So a Borehole
gain materially larger than Hartmann's is the signature, and 2 of 3 Hartmann
cells are still running.

If Hartmann gains as much, the effect is not range-violation — it is one of the
three confounds, most plausibly the fidelity shift, which h58 already showed can
move Borehole regret by 22% on a single seed.

### Fidelity fraction does not explain the differences between MF-DRO configurations

**EXPLORATORY**, on all completed Borehole data — seven distinct configurations
across h57, h60, h61 and h63, spanning 2% to 98% realised HF.

| arm | HF% | nHF | rel regret |
|---|---|---|---|
| TEACHER (`rollout_policy=thompson`) | **2%** | 3 | **43.8%** |
| RHOTRUE (`rho_fixed=1.2566`) | 18% | 30 | 21.8% |
| NOLFINIT (`initial_lf=0`) | 82% | 89 | 24.4% |
| **POOL600** | 90% | 95 | **19.5%** |
| **REFINE** | 94% | 97 | **19.3%** |
| REWARD (`rollout_reward=improvement`) | 95% | 97 | 24.7% |
| BASE | 98% | 100 | 23.7% |

Correlation across all seven is **-0.685**, but it is carried almost entirely by
TEACHER's outlier. **Excluding that one point the correlation flips sign to
+0.071 over the remaining six** — i.e. nothing — and the regret range across
18%-98% HF is just 19.3%-24.7%, a 5.4-point band with no ordering in it.

*(Correction: this entry first stated -0.690 and -0.106. Both were written from
memory of the shape rather than the computed output; the verified values are
-0.685 and +0.071. The conclusion is unchanged and slightly strengthened — the
residual correlation is not weakly negative, it is absent.)*

So: **starving HF entirely is catastrophic (2% -> 43.8%), but anywhere between
18% and 98% the fidelity mix carries essentially no information about regret.**
RHOTRUE at 18% HF beats BASE at 98%; REWARD at 95% is worse than RHOTRUE at 18%.
The two best configurations sit at 90-94%, adjacent to BASE's 98%, and differ
from it by *teacher optimisation*, not by fidelity.

**This closes out the fidelity thread.** Across h58 (the HF floor), h60 (teacher
identity), h63 (rho pinning) and h61 (pool/refinement), every intervention that
moved the fidelity mix has been measured, and the mix is not the explanatory
variable. MI-Greedy reaches 8.3% at 100% HF while MF-DRO reaches 19.3% at 94% —
the gap is not fidelity allocation and cannot be closed by adjusting it.

The one thing the fidelity data does establish is a floor: a configuration that
drops below ~10% HF on Borehole cannot move its incumbent enough to compete,
which is h58's finding arrived at from the other direction.

---

## H63 COMPLETE: your KO-misspecification hypothesis is SUPPORTED by the contrast

**CONFIRMATORY.** Both locked predictions met.

| benchmark | true slope | representable? | BASE | RHOTRUE | wins | gain |
|---|---|---|---|---|---|---|
| **Borehole 8D** | **1.2566** | **NO** | 23.7% | **21.8%** | **3/3** | **+1.9%** |
| Hartmann 6D | 0.9792 | yes (control) | 14.7% | **31.3%** | **0/3** | **-16.6%** |

- **Prediction 1 (PRIMARY)** — RHOTRUE beats BASE on Borehole >= 2/3: **MET (3/3)**
- **Prediction 2 (DISCRIMINATING)** — Borehole gain exceeds Hartmann's:
  **MET, and by a wide margin (+1.9% vs -16.6%)**

The contrast is exactly the signature the protocol pre-registered. The benchmark
whose true LF->HF slope lies **outside** the model's representable range gains
from pinning rho at the measured value; the control whose slope lies **inside**
it is badly harmed. Confounds 2-4 (adaptivity, ensemble rho diversity, fidelity
mix) act on **both** benchmarks — Hartmann's mix inverted 81%->16% HF just as
Borehole's did — so they cannot produce a contrast this asymmetric. That is why
the contrast, not the level, was designated the signal in advance.

**This is the second locked prediction met in this project, and the first
user-originated hypothesis to survive its own control.**

### What it does not license

1. **rho is not the dominant term.** On the same Borehole cells, h61's POOL600
   (19.5%) and REFINE (19.3%) both beat RHOTRUE's 21.8%. Correcting the
   surrogate's representability buys **1.9 points**; improving the teacher's
   acquisition optimisation buys **4.2-4.4**. Both matter; the teacher matters
   more.
2. **Pinning rho is not a usable fix.** It is catastrophic where the slope is
   already representable (-16.6% on Hartmann), so it cannot be shipped as a
   default. What the result identifies is a **modelling defect** —
   `rho = sigmoid(log_rho)` cannot express a slope above 1 — not a tuning knob.
   The principled repair is to reparameterise rho with unbounded support
   (e.g. softplus, or an unconstrained scalar), letting it fit whatever the data
   requires on either benchmark.
3. **n = 3 per cell**, no p-values, and the four-way confound stands. The
   contrast is what survives it, not the levels.

### The concrete, testable follow-up this points to

Replace `sigmoid` with an unbounded link in `KennedyOHaganGP` so rho is *fitted*
without a (0,1) ceiling, and re-run both benchmarks. That would let Borehole
reach ~1.26 while leaving Hartmann free to stay near 0.98 — a single change that
should help where the ceiling binds and be inert where it does not. It is the
first repair this project has identified that is principled rather than a knob.

---

## H64 COMPLETE: the pre-registered null is CONTRADICTED, and MF-DRO clears the best Hartmann baseline

**CONFIRMATORY**, and it goes against what I predicted.

| method | s44 | s46 | s48 | mean | rel | sd |
|---|---|---|---|---|---|---|
| MF-DRO (BASE) | 0.7531 | 0.2875 | 0.4228 | 0.4878 | 14.7% | 0.2395 |
| MF-MES *(best baseline)* | 0.3019 | 0.1979 | 0.3491 | 0.2830 | 8.5% | 0.0774 |
| MF-MI-Greedy | 0.6171 | 1.0834 | 0.6848 | 0.7951 | 23.9% | — |
| **MF-DRO + POOL600** | 0.3162 | **0.1960** | **0.2454** | **0.2525** | **7.6%** | **0.0604** |

Paired: **3/3 over BASE**, **3/3 over MI-Greedy**, **2/3 over MF-MES**.

**This is the first time in this project that a DRO variant has beaten the best
baseline on a discriminating benchmark.** 7.6% against MF-MES's 8.5%, with the
lowest seed-to-seed spread of any method tested on Hartmann (sd 0.0604 vs
MF-MES's 0.0774 and BASE's 0.2395).

### The mechanism is UNEXPLAINED, and that was written down in advance

H64's pre-registered prediction was **NULL on Hartmann**, on a measured basis:
widening the teacher's pool 200 -> 600 buys Hartmann **1.00x** additional
acquisition value (against Borehole's 1.44x), with `y*` held fixed from an
independent 4096-point reference. If the channel were acquisition quality, there
was no gain available to convert into regret.

The regret gain is large anyway — 14.7% -> 7.6%. So **a wider teacher pool helps
through some channel other than the acquisition value it finds**, and this
project does not currently know what that channel is. The addendum committed
before these results stated: *"If Hartmann POOL600 DOES improve regret, the
acquisition-value channel is wrong... That outcome is more interesting than the
predicted one and must not be explained away."*

I am not going to propose a mechanism now. Five of the six mechanisms I have
proposed in this project were refuted by the next measurement, and the honest
statement is that the effect is real, reproduced on 3/3 seeds across two
benchmarks, and unexplained.

### Currin: the no-harm check behaved as designed

BASE 0.0014 vs POOL600 0.0026, wins 1/3 — both at **0.0% relative regret**.
Pre-registered as a no-harm check on a saturated benchmark and **not counted as
support either way**, which is what the protocol required.

### Where the north star now stands

| benchmark | best baseline | MF-DRO+POOL600 | met? |
|---|---|---|---|
| Currin 2D | 0.0% (SF-MES) | 0.0% | tie (saturated) |
| **Hartmann 6D** | **8.5%** (MF-MES) | **7.6%** | **MET** |
| Borehole 8D | 8.3% (MI-Greedy) | 19.5% | not met |

One benchmark of three now meets "at least as good as the baselines". n = 3, no
p-values, and the compute ratio is unchanged. But it is the first arrival, not a
direction.

### What POOL600 changed on Hartmann, measured (descriptive — no mechanism claimed)

h64's Hartmann gain (14.7% -> 7.6%) is unexplained: the acquisition-value channel
measured **1.00x** at N=600 there. Narrowing what actually differs, as a
precondition for any future mechanism test. **n = 3, descriptive only.**

| arm | seed | regret | nq | nHF | HF% | improv | HF spread |
|---|---|---|---|---|---|---|---|
| BASE | 44 | 0.7531 | 31 | 25 | 81% | 1 | 0.1538 |
| BASE | 46 | 0.2875 | 179 | 3 | 2% | 3 | 0.3432 |
| BASE | 48 | 0.4228 | 67 | 19 | 28% | 7 | 0.3095 |
| POOL600 | 44 | 0.3162 | 28 | 25 | 89% | 3 | 0.3448 |
| POOL600 | 46 | 0.1960 | 130 | 10 | 8% | 7 | 0.2910 |
| POOL600 | 48 | 0.2454 | 151 | 7 | 5% | 6 | 0.3255 |

| quantity | BASE | POOL600 | ratio |
|---|---|---|---|
| HF queries | 15.67 | 14.00 | 0.89x |
| HF fraction | 0.369 | 0.339 | 0.92x |
| **improvements** | **3.67** | **5.33** | **1.45x** |
| HF query spread | 0.269 | 0.320 | 1.19x |

**POOL600 does not buy more high-fidelity queries — it buys fewer (0.89x) — yet
it converts them into 1.45x as many incumbent improvements.** Seed 48 is the
sharpest case: HF queries fall 19 -> 7 while improvements rise 7 -> 6 and regret
halves (0.4228 -> 0.2454). Seed 44 keeps HF fixed at 25 and triples improvements
(1 -> 3).

The HF query spread also *widens* (1.19x), consistent with h61's liveness check
on Borehole where refinement widened rather than concentrated the search.

**No mechanism is proposed.** Five of six mechanisms proposed in this project
were refuted by the next measurement, and "more improvements per HF query" is a
restatement of the outcome, not an explanation of it. What this rules out is the
simplest candidate: the gain is **not** from spending more of the budget on the
fidelity that moves the incumbent.

It also matches h63's fidelity-thread conclusion from a different direction —
across seven Borehole configurations the HF fraction carried no information about
regret once the 2%-HF outlier was excluded (corr +0.071). Both point at *what is
done with* the HF queries rather than *how many* there are.
