# MF-DRO: what we measured, what it means, and what to ask

Every number below is from a recorded experiment in `experiments/`. Claims that
were **retracted** are marked. Claims that are **not established** are marked.
Nothing here is a hypothesis presented as a result.

---

## 0. Notation and vocabulary

| term | plain meaning |
|---|---|
| **HF / LF** | high-fidelity / low-fidelity. The expensive accurate oracle vs the cheap rough one. |
| **incumbent** | the best HF value found so far. "Frozen incumbent" = it stops improving; the optimizer keeps querying but never finds anything better. |
| **simple regret (SR)** | `f* − (best HF value queried so far)`. How far the best point we actually evaluated is from the true optimum. |
| **inference regret (IR)** | `f* − f(argmax μ_H)`. How far the model's *recommendation* is from the optimum. SR needs a good point to be **evaluated**; IR only needs the model to **believe** correctly. |
| **amortization** | "pay training cost once, reuse it many times." The usual reason to use a neural policy: train offline, then act cheaply forever. |
| **acquisition function** | the rule that scores candidate points and picks the next query (EI, UCB, MES…). |
| **MES / MF-MES** | Max-value Entropy Search. Picks the query expected to tell you most about the optimum's *value*. MF-MES is the multi-fidelity version, cost-normalised. |
| **teacher** | inside MF-DRO, MF-MES is used to generate training data. The DT learns to imitate it. |
| **rollout** | a simulated 8-step trajectory on the GP's *imagined* data, used as DT training data. |
| **fantasy** | an imagined observation `y ~ GP posterior`, not a real function evaluation. |
| **RTG / BTG** | return-to-go / budget-to-go: the "goal" numbers a Decision Transformer is conditioned on. |
| **h** | the transformer's internal summary vector of (RTG, BTG, state) at the current step. |
| **w(h), b(h)** | the coefficient vector and bias the DT outputs; they define the scoring rule. |
| **cf_k** | the feature vector of candidate *k*: `[x_norm, μ_H, σ_H, μ_L, σ_L, dist_inc]`. |
| **μ_H, σ_H** | GP posterior mean and standard deviation at high fidelity. |
| **argmax** | "the item with the highest score." |

---

## 1. What MF-DRO does, per real iteration

1. Fit an ensemble of 10 Kennedy–O'Hagan two-fidelity GPs on the real data.
2. Simulate **200 rollouts** of length 8. At each rollout step, **MF-MES picks the
   query**; the observation is a *fantasy* draw from the GP.
3. Compute a reward for each rollout (joint information gain about the optimum).
4. **Train the Decision Transformer** on these rollouts (~10 epochs).
5. Draw **200 fresh uniform candidates**, and score them:
   `score_k = ⟨w(h), cf_k⟩ + b(h)`, then take the argmax as the next query.
6. Choose fidelity by `ℓ ~ Bernoulli(p)` where `p = fidelity_head(h)`.
7. Evaluate the real function, update the data, repeat.

**Step 4 happens every single iteration** — but the DT is **fine-tuned, not
re-initialised**: `self.dt` and its optimizer are constructed once
(`mf_dro.py:1901, 1907`) and every iteration continues training the same weights
on a fresh rollout batch. Knowledge does accumulate across iterations.

**Important**: those rollouts are *simulated* on the GP, so retraining consumes
**compute only — no real function evaluations**. In the intended application
(expensive black-box objective), compute is not the scarce resource, so training
cost is not by itself an objection to the design.

---

## 2. The headline numbers (Hartmann 6D, 10 seeds, matched cost 200)

| method | final simple regret |
|---|---|
| MF-DRO (joint-MES reward) | **0.4007 ± 0.0475** |
| MF-DRO (improvement reward) | 0.5047 ± 0.0395 |
| **MF-MES teacher, no DT at all** | **0.4781 ± 0.0414** |
| MF-MI-Greedy (baseline) | 0.5091 ± 0.1266 |
| MF-GP-UCB (baseline) | 1.7934 ± 0.1223 |

**MF-DRO vs the baseline is not statistically distinguishable.** Paired
difference −0.1085, better on 6/10 seeds, Wilcoxon p = 0.432. Every conventional
test agrees: paired t p = 0.352, Welch p = 0.439, 95% CI on the paired difference
[−0.325, +0.108] contains zero, Cohen's dz = 0.311 (small). **82 seeds** would be
needed for 80% power. *(experiments/h17, h38)*

---

## 3. The chain of evidence — which experiment proves which claim

### Claim A — The DT's decision does not depend on its inputs.

| evidence | experiment |
|---|---|
| Swap `h` for a genuinely different state → argmax unchanged **0/12**, score-vector correlation **1.000000** | h5 AUDIT |
| Same, across the 12 states of a whole real run (states differ by L2 = 1.4968) → **0/12** | h5 AUDIT |
| Sweep RTG across its realised band → **0/12**; across 10⁻³–10⁶ → **1/12** | h8, h26 |
| Sweep BTG in-band and far out-of-band → **0/12** both | h26 |
| Feed the last 7 real queries as context → proposals **bit-identical** (max |Δx| = 0) | h27 |
| Amplify state deviations up to **100×** → **0/12** | h22 |

### Claim B — The reason: it is a fixed linear rule.

| evidence | experiment |
|---|---|
| Decompose `w(h) = w̄ + δ(s)`. Then `‖δ‖ / ‖w̄‖ = **0.00129**` (0.13%) | h23 |
| `w̄` **alone** reproduces the full model's argmax on **12/12** candidate pools | h23 |
| Top-1/top-2 score margin exceeds anything `δ(s)` can contribute by a median factor of **77** | h23 |
| `b(h)` adds the *same constant to every candidate*, so it **cannot** change an argmax — confirmed 0/12 | h23 |
| The fidelity probability `p` spans only **0.5570–0.5577** (sd 2.4×10⁻⁴) across 24 decisions | h33 |

### Claim C — The signal is lost gradually, in two places.

| evidence | experiment |
|---|---|
| Relative spread across real states: state **0.2155** → hidden `h` **0.0745** → coefficients `w` **0.0219** | h5 AUDIT |
| That is 0.346× at the encoder and 0.294× at the head; **~10× end-to-end** | h5 AUDIT |
| A **randomly initialised** encoder contracts almost as much (0.4601× vs 0.3898×) → the encoder's share is **architectural** | h21 |
| A random *head* **amplifies** (1.6039×) where the trained head contracts (0.3348×) → the head's share is **learned** | h21 |

### Claim D' — Continued training does change decisions, but buys no regret.

| evidence | experiment |
|---|---|
| Freeze the DT after iteration 5 vs keep training: paired **+0.0752**, 95% CI **[−0.0105, +0.1608]** (contains zero), Wilcoxon p = 0.0795, n = 30 pre-registered | h6 |
| Yet continued training **does** change decisions: 377 paired decisions, argmax agreement **0.817** → ~18% of decisions differ | h7 |

So the accumulation is real but not useful: training changes roughly a fifth of
the choices and the regret difference is statistically indistinguishable from
zero.

### Claim D — Removing the transformer changes nothing measurable.

| evidence | experiment |
|---|---|
| MF-MES teacher alone, same seeds/budget/pools, DT deleted from the decision: 0.4781 vs MF-DRO 0.4007, paired p = **0.2324** | h31 |
| **The teacher alone also fails the same bar** (0.5195 vs 0.3825) | h31 |
| The DT's location choice is a *faithful* copy: teacher's rank of the DT's pick is **median 2 of 200** (range 1–12) | h32 |

### Claim E — The behaviour it copies is exploitative.

| evidence | experiment |
|---|---|
| Chosen points sit **below the pool median** in σ_H and near the top in μ_H, over 1600 decisions. Magnitude is initialisation-dependent: **2.9th** percentile with the oversized init (h28), **27.9th** with the standard one (h41) | h28, h41 |
| True at **every** operating point swept: cost ratio {2,4,8,16} × y* samples {5,10,50}, never above **5.5%** in 12 cells | h29 |
| Mechanism: MES depends on `γ = (y*−μ)/σ` alone and decreases in γ. Chosen γ **+1.59** vs pool **+2.72** — correct behaviour. γ>0 everywhere, so minimising it resolves through μ (99.5th pct), and μ–σ anti-correlation (−0.4696) pulls σ down as a side effect | h41, h30 |

### Claim F — Cost.

| evidence | experiment |
|---|---|
| MF-DRO takes **~37 s per query** on 2D, 6D and 8D alike; MI-Greedy takes **~0.25 s** | timing run |
| The per-iteration cost is dominated by 200 rollouts × 8 GP-conditioning steps | timing run |

**How to read this**: it is *compute*, not real evaluations. If the objective
takes hours (wet lab, expensive simulation), 37 s of overhead is negligible and
this is not a criticism. It only bites on cheap synthetic benchmarks like ours.

---

## 4. Why the DT fails on Hartmann — start to end

1. **Training data comes from MF-MES rollouts.** The teacher picks every rollout
   step, so the DT can only learn what the teacher demonstrates. *(h28)*

2. **What the teacher demonstrates is high-mean, below-median-uncertainty
   points** — and this is MES behaving correctly, not malfunctioning.

   MES is a function of `gamma = (y* - mu_H)/sigma_H` **only** — `log sigma`
   cancels out of the entropy difference — and it *decreases* in `gamma`.
   Measured: `gamma` at the chosen point is **+1.59** vs **+2.72** for the pool,
   so the acquisition is correctly picking the highest-information candidate.

   Since `mu` never exceeded the sampled `y*` (0% of 40 pools), `gamma > 0`
   throughout, and minimising it pulls two ways: raise `mu` **or** raise `sigma`.
   It resolves through `mu` — chosen points sit at the **99.5th percentile of
   mu** — and because `mu` and `sigma` are **anti-correlated in a GP**
   (`corr = -0.4696`), that drags `sigma` down to the **27.9th percentile** as a
   side effect. *(h41; anti-correlation from h30)*

   **Correction:** an earlier version of this document said "the posterior-mean
   term dominates the argmax." **MES has no such term** — that was a UCB-shaped
   description of an acquisition that does not have that form.

   **Caveat on a number:** h28 reported the chosen-`sigma` percentile as **2.9%**,
   but that used the oversized initial design. With the literature-standard
   design it is **27.9%**. The direction survives; the magnitude is
   initialisation-specific and should not be quoted as a property of MF-MES.

3. **The DT copies this faithfully.** Its chosen point is the teacher's 2nd best
   of 200, median. So the DT is not adding error. *(h32)*

4. **But the architecture only lets the state matter through 11 numbers.** The
   score is `⟨w(h), cf_k⟩ + b(h)` — linear in the candidate features. `b(h)`
   cannot change an argmax at all. So `w(h)` is the entire channel. *(h23)*

5. **And `w(h)` barely moves.** 0.13% of its own length across states, because
   state variation is contracted ~10× on the way (≈3× by the encoder, ≈3× by the
   head). *(h5 AUDIT, h21, h23)*

6. **Therefore the policy is a fixed acquisition rule.** `w̄` alone reproduces
   every decision. The margin between the top two candidates is ~77× larger than
   anything the state can contribute. *(h23)*

7. **The queries still move between iterations** — but only because the 200
   candidates are redrawn each iteration and their GP features update as data
   accumulates. That makes the policy *look* adaptive while conditioning on
   nothing. *(h5 AUDIT, h23)*

8. **So deleting the DT costs nothing** (p = 0.23), and the ceiling is set by the
   acquisition, not the transformer. **The teacher alone also fails the bar.**
   *(h31)*

---

## 5. Claims we retracted (state these to your advisor — they matter)

| retracted claim | what actually happened |
|---|---|
| "The student inverted the sign of its teacher" | The teacher is itself exploitative; the student imitates it. *(h28)* |
| "`w[σ_H] < 0` means the rule avoids uncertainty" | It is a *partial* coefficient under `corr(μ_H, σ_H) = −0.4696`. The **teacher's own** partial coefficient is also negative. Statistical suppression, not aversion. *(h30)* |
| "The distillation is lossy" | It is faithful (median teacher-rank 2/200). The "lossy" reading came from argmax agreement, a bad metric when top candidates are near-ties. *(h32)* |
| "The fidelity head's level is harmful" | MF-DRO's higher HF rate coincides with *slightly lower* regret. Only its *uninformativeness* stands. *(h31, h33)* |
| "ε and α_f are coupled" | Rested on a diversity metric that omitted query locations. *(h18/h19)* |
| "Cost-weighted regret is significant (p = 0.0371)" | Post-hoc metric, computed after the pre-registered one failed. Over the 7 tests reported, Bonferroni threshold 0.00714, **none survive**. *(h37)* |

---

## 6. What is NOT established

- **Why the LF branch loses value inside rollouts.** `LF/c_L` falls −37% across a
  rollout while `σ_L` stays flat. Three mechanism guesses (the HF floor, σ_L
  shrinking, y* drift) were each tested and each **failed**. Recorded as measured
  but unexplained. *(h34, h35, h36)*
- **Whether any of this generalises beyond Hartmann 6D.** One benchmark.
- **Whether the regression head would re-freeze — tested, answer is NO.**
  Both heads on Currin (n=1, 14 iterations): regression **14/14 distinct
  proposals, 2 incumbent improvements**, regret 0.0073; candidate scoring
  **14/14, 1 improvement**, regret 0.0013. Query spread nearly identical
  (0.050 vs 0.048). *(h39)*

  This matters for a confound: `use_candidate_scoring=True` was the default when
  the freeze was declared resolved (h1, 0/10 frozen), and candidate scoring
  landed **before** the leak fix — so h1 could not tell which change fixed the
  freeze. h39 separates them: **the leak fix did it**, not the scoring rewrite.
  Still n=1 and one benchmark; needs replication.
- **How MF-DRO compares to random search at matched budget.** Random search at
  budget 300 on Currin gives 0.5579 ± 0.1686; MF-DRO reached 0.0053 at budget 18,
  but those budgets are *not* matched, so this is a floor, not a comparison.

---

## 7. A protocol problem worth raising first

Our initial design was **6.4× larger than the literature guideline**:

| source | initial design |
|---|---|
| DRO paper (the method's own paper) | `Ninit = 5` Sobol points |
| Takeno MF-MES | 5d LF + 4d HF |
| Best Practices (Nat Comput Sci 2025) | **10% of total cost** |
| **ours** | **348 cost against a 200 budget = 64% of total** |

**We spent 64% of the budget before the optimizer acted.** Direct evidence this
mattered: with a properly sized initial design, Hartmann regret is **2.08** after
8 iterations; with the oversized one it was 0.40. And Currin reaches regret 0.005
by iteration 6 — the benchmark is effectively solved by initialisation alone.

**Every comparison in this report may be compressed by this.** It does not change
the mechanistic findings (those hold weights fixed and vary inputs), but it could
well change the performance comparison.

---

## 8. Questions for your advisor

**On the protocol (ask first — it may invalidate the comparison):**
1. Our initial design was 64% of total budget vs the 10% guideline. Should every
   performance number be re-run before we conclude anything about ranking?
2. The DRO paper uses `Ninit = 5`. Was the large initial design deliberate?

**On the method's premise:**
3. The DT is fine-tuned every iteration on *simulated* rollouts — compute only,
   no real evaluations — and it does accumulate. But freezing it after iteration
   5 is statistically indistinguishable from continuing to train (h6, CI
   contains zero), even though continued training changes ~18% of decisions
   (h7). **So what is the online training supposed to be buying?** Would
   training once across many functions and deploying frozen (true meta-learning
   across tasks, which we have never tested) be the intended design?
4. The score is `⟨w(h), cf_k⟩ + b(h)` — linear in candidate features, so the
   state's entire influence is 11 numbers, and the bias term provably cannot
   matter. **Was this bottleneck intended?**

**On what to do next:**
5. Given the teacher alone also fails the bar, is the interesting question
   "why does DT-as-BO-policy fail" (a negative-results paper we can support), or
   should we change the teacher/acquisition instead?
6. If we pursue the negative result: is one benchmark enough, or do we need
   Currin/Borehole/etc. before it is publishable?
7. Is there a version of the conditioning we should try where the state can
   express more than a linear reweighting of fixed features?

**The honest summary to open with:**
> "The pipeline works — GP + MF-MES substantially beats random. What does not
> work is the Decision Transformer specifically: it collapses to a fixed
> acquisition rule, and deleting it entirely changes nothing measurable
> (p = 0.23). Also, our initial design was 6.4× larger than the standard, so the
> performance comparisons may need re-running."
