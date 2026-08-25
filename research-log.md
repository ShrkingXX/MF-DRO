# Research Log

## Bootstrap — inventory of prior work

Inventoried 1280 result JSONs across 86 experiment directories into
`data/results_inventory.csv` (`src/analysis/inventory.py`).

**Decision: do not run the pre-registered init comparison first.**
The inventory refutes the leading hypothesis before any new compute was spent.
Freeze rate is 0/10 for MF-MI-Greedy and 100% for SF-DRO on the same benchmark,
which is incompatible with initialization coverage being the primary cause.
Redirecting the first inner-loop experiment toward isolating the DRO/DT
mechanism (H1) instead. PROTOCOL.md's evaluation section is untouched; its
"leading hypothesis" section needs a user amendment.

## Session — target-leakage bug found; prior numbers invalidated

**The single most consequential finding so far.** The DT readout was taking the
*action* token (`h[:, 3::4]` in `forward_mf`, `[:, 2::3]` in the SF path). The
causal mask `triu(diagonal=1)` lets a position attend to itself, so `a_emb` —
which embeds the true `actions_x`/`actions_ell` — was visible to the very heads
predicting it. `propose_mf` then zeroes that slot at inference, because the
action is what we are computing. Train/inference shift on the most predictive
input, plus a free copy of the label during training.

Verified by ablation, not asserted: zeroing `a_emb` after training cost the
fidelity head 15.6% of its loss pre-fix, 0.7% post-fix.

**Why this resolves the standing "one mechanism or two?" question.** The SF path
had the identical bug. SF-DRO freezes 12/12 and has no fidelity head, so the
fidelity-threshold bug could never have explained it — but target leakage
explains both, and matches the observed SF-DRO >= MF-DRO freeze ordering.

**Consequence: every pre-fix DRO number in the inventory is suspect**, including
MF-DRO's 1.31 mean regret and the freeze-rate table. Recorded prominently at the
top of findings.md. `data/results_inventory.csv` is retained as-is (it is an
accurate record of what those runs produced) but must not be compared against
post-fix results.

**A second, unrelated mechanism was also separated out.** MF-GP-UCB's 100%
freeze is cost-ratio degeneration to all-LF, already documented in
`src/baselines/mf_baselines.py:206-213`, confirmed by a smoke run
(`n_hf_queries=0`). It has no DT and never had the leak. So the old "100%
freeze" column pooled two unrelated causes. This closes a standing open
question in findings.md.

Several compounding defects were fixed in the same commit (tau=0 state
collapse; BTG pinned at the cost floor; RTG cap giving 2 distinct values across
200 trajectories; a soft teacher target with entropy at exactly 100% of log K;
MES columns reconstructing the label at corr 0.80). The reward was switched from
`mes_entropy` to regret-based `improvement` after a gate measurement moved from
p ~ 0.32 (5/10 groups negative) to z = 2.63, p = 0.0085 (9/10 positive).

**Decision: validate before continuing to search.** Rather than keep hunting
DT-internal pathologies, run the first post-fix numbers under the frozen
protocol. Protocol locked in d1f557f before launch; prediction (MF-DRO mean
< 1.0) recorded there. Measured 41 s/iter for MF-DRO after
`rollouts_per_model` 7 -> 20, projecting ~2.3 h worst case per run;
15 workers x 1 thread = 15, 92% user CPU at launch.

## Tick — literature resolves the RTG puzzle; interim baseline surprise

**Returned to literature rather than running more experiments**, because the
RTG-insensitivity had been measured exhaustively with no theory: sweeping the
target 0.1x-10x never moved the argmax, in any configuration, and a degenerate
reward embedding had been ruled out.

It is a **named, published DT failure mode.** RADT (arXiv:2402.03923): DT
"struggles to align the actual return with the target return due to the
under-allocation of attention scores to the return-to-go tokens," and the fix
must be *structural, not parametric* — which retroactively explains why nothing
we tuned ever helped. DDT (arXiv:2601.15953) supplies a concrete mechanism:
drop RTG from the input, condition via AdaLN-Zero on the last RTG.

Our architecture is the worse case: 4 tokens per step means **two of four** are
scalar conditioning signals competing for attention, vs one of three in standard
DT. Orthogonal to the 7bcc3b8 leak fix — that was heads reading their own
labels; this is conditioning strength once the leak is gone.

H4 protocol locked (350c0a3) with a **mechanism-level** prediction (argmax moves
on >30% of RTG sweeps vs a measured 0%) that is falsifiable in minutes without a
full BO run. Deliberately not launched while the grid owns the pool.

**Interim result from the running grid, worth flagging now.** All 10 baseline
seeds are done at matched cost=200:

  MF-MI-Greedy: mean 0.509 (min 0.188, max 1.540), mean realized cost 207.2
  MF-GP-UCB:    mean 1.933 (min 1.471, max 2.375), mean realized cost 200.0

The inherited MI-Greedy figure this project has been calibrating against was
**0.279**. At genuinely matched cost it is **0.509**, with large seed spread.
This is direct evidence for the standing suspicion that prior cross-method
numbers were never cost-matched (MI-Greedy median 53 iters vs MF-DRO 100 vs
MF-GP-UCB 800), and it roughly halves the apparent gap MF-DRO must close.
MI-Greedy's realized cost overshoots by ~3.6% because its run() checks the
budget at round start and a round costs up to 2*c_H; small, but it is a real
asymmetry and the analysis flags it.

## Tick — H4 refuted; the bottleneck is downstream of conditioning

Implemented AdaLN-Zero return conditioning (DDT mechanism) behind
`rtg_conditioning="adaln"` and ran the locked mechanism probe on 1 core
alongside the grid (10 + 1 = 11 <= 15).

**Refuted, in the opposite direction.** Sweeping RTG 0.1x-10x over 12 resampled
candidate pools: `token` moved the argmax on 25% of sweeps, `adaln` on **0%**.
The locked prediction was >30% for adaln.

Two corrections I am recording against myself rather than burying:
1. The prediction asserted "a measured 0% under token." Under this probe's
   protocol token measures **25%** — the earlier 0% came from a differently
   trained model and a single pool. The valid comparison is the within-probe
   one: 25% vs 0%.
2. AdaLN-Zero initializes to identity by design, and the probe trains 10 epochs
   on one batch. That is plausibly too little for the modulation layer to leave
   its zero init, which would produce exactly 0%. So this refutes "AdaLN is a
   cheap win at this scale", not the RADT/DDT mechanism itself.

**The valuable part is the sharper diagnosis.** Both arms show
argmax(score)==argmax(mu_H) on 67-75% of pools. Combined with the earlier result
that a batch-mean or *shuffled* h changes the argmax only ~8% of the time:

    rtg -> h     works    (embeddings differ, score vectors shift)
    h   -> score nearly severed

The bottleneck is **downstream of the conditioning**. That retroactively
explains why every conditioning-side intervention has failed — including this
one — and predicts that further conditioning work is wasted effort. Logged as
H5. Next intervention should target the score head's collapse onto mu_H, e.g.
by removing mu_H from candidate features the way removing the MES columns
already took argmax(mu_H) agreement from 100% to 70%.

## Tick — H5 refuted; the finding escalates to the frame itself

H4 (AdaLN conditioning) and H5 (deny the score head its GP features) were both
locked, run, and **both refuted**. H5's refutation is the important one because
the manipulation demonstrably worked: stripping the GP features drove
argmax(mu_H) agreement 66.7% -> 0.0%, so the shortcut really was removed — and
the head *still* ignored `h` entirely (0/12), while RTG movement FELL
(16.7% -> 8.3%). That is pre-registered outcome #2, written into the protocol
before the run.

Five independent probes now agree that within a single trained model the
proposal is near-independent of the conditioning. Since `x_t_trace` std is
0.166-0.213, queries plainly do move, so the reconciliation is that the DT is
**retrained every iteration**: it changes because its weights are re-fit, not
because it conditions. MF-DRO is behaving as a per-iteration acquisition
function parameterised by a transformer.

That claim explains every failed conditioning-side intervention at once, and it
escalates the question from "which component is broken" to "does the learned
policy do any work at all" — which is PROTOCOL.md's actual question about the
DRO *frame*. Hence H6 (freeze the DT after k=5) rather than a third component
patch.

`freeze_dt_after` implemented and **verified**, not assumed: a 9-iteration smoke
run gives L_loc = [0.2989, 0.2642, 0.2566, 0.2475, 0.2530, 0.2530, 0.2530,
0.2530, 0.2530] — constant from index 5, so training genuinely stops.

H6 FROZEN arm launched on spare capacity (6 grid workers still live + 6 H6
workers = 12 <= 15). Its control arm is h1's MF-DRO arm, already running, so no
duplicate compute.

**h1 partial (4/10 MF-DRO):** regrets 0.4073, 0.4239, 0.4412, 0.5218, partial
mean 0.4486 vs MI-Greedy 0.5091 +/- 0.1266. Encouraging but NOT reportable: the
completion-order bias documented earlier means finished runs over-represent
HF-heavy behaviour, and n=4 of 10. No claim until all seeds land.

## Tick — h1 complete (FAIL), H6 inconclusive, extension pre-registered

**The frozen evaluation is answered.** MF-DRO 0.5047 +/- 0.0395 vs MF-MI-Greedy
0.5091 +/- 0.1266 at matched cost. Success test: **FAIL** (0.5442 >= 0.3825),
reported exactly as pre-registered. PROTOCOL.md permits this and it is the
honest headline: the fixes do not produce a method that strictly beats
MI-Greedy.

What DID change: **0/10 runs frozen** vs a pre-fix 9/12. The named pathology is
resolved. But the paired analysis corrects the naive reading — the near-zero
mean difference (-0.0045) is carried almost entirely by seed45, where MI-Greedy
failed badly; on the MEDIAN seed MF-DRO is slightly worse (+0.09).

**H6 established nothing at n=10 and I am not pretending otherwise.** The paired
estimate changed sign across the run of the experiment: n=1 -0.208, n=5 -0.103,
n=7 -0.010, n=9 +0.062, n=10 +0.098. Final CI [-0.097, +0.292] straddles zero,
Wilcoxon p=0.322. The variance ratio looked strong (4.79x) but the tests
disagree: F p=0.029 and Bartlett p=0.029 versus **Levene p=0.209**. Regret is
right-skewed and bounded below, so the robust test is the one to believe, and
last tick's "variance is the robust finding" claim does not survive. Both arms
are also paired, which independent-sample variance tests do not respect.

A post-hoc power analysis explains all of it: paired sd is 0.3138, LARGER than
the effect being chased. The design resolves only effects >= ~0.3 regret units
against arms sitting near 0.5. Roughly 80 seeds would be needed for the observed
+0.098.

**Extension pre-registered (b00dacd) with explicit anti-p-hacking guards**: final
n fixed at 30 in advance, n=30 analysis is primary AND final, no further
extension regardless of outcome, and the primary prediction is a NULL (CI still
contains zero) precisely so a null cannot later be spun as a finding. It is also
stated up front that n=30 cannot resolve the mean. This does not touch
PROTOCOL.md — H6 is an MF-DRO-vs-MF-DRO internal comparison, not the frozen
evaluation.

**Same power caveat now propagated to the headline**: MF-DRO and MI-Greedy
differ by 0.0045 with MI-Greedy sd 0.400. That is "underpowered to distinguish",
not "demonstrated equivalence", and findings.md/research-state.yaml both say so.

Self-inflicted error worth logging: I wrote invalid YAML into research-state.yaml
(an unquoted list item beginning with a quote) and the commit went through
because I ran the validator on a separate line instead of chaining it with &&.
Caught on the next validation, fixed, commit amended. Chain validators to the
commit.

## Tick — ETA correction, and a decision not to abandon a pre-registration

**Corrected ETA.** I previously projected ~100 min for the H6 extension. Measured
throughput after 99 min: LIVE 5/20 (7 workers), FROZEN 8/20 (8 workers). That
projects **~4.9 h remaining on the LIVE arm**, ~2.5 h on FROZEN. My earlier
estimate was wrong by ~3x because I costed a run at ~50 min when the h1 MF-DRO
arm had actually averaged 76 min, and did not account for wave structure
(20 jobs / 7 workers = 3 waves, not 2).

**The decision this forces.** H7 is implemented and is a strictly better
instrument for H6's question (~50-200 paired decisions per run vs 1 noisy regret
scalar). It is tempting to kill the H6 extension and run H7 now.

I am NOT doing that. The extension was pre-registered (b00dacd) with n=30 as
"primary AND final" precisely to stop me making sample-size decisions on the fly.
Abandoning it at n~15 because a nicer experiment appeared would be a protocol
deviation, and — worse — it would be indistinguishable from abandoning it because
the interim numbers were not going my way. The interim (paired n=13, +0.0707, CI
[-0.080, +0.221]) is consistent with the pre-registered null, so there is no
result-driven motive here; but the whole value of a pre-registration is that it
binds regardless of motive.

The cost of honouring it is ~5 h of unattended wall-clock, which for an
autonomous overnight loop is cheap. H7 launches when the FROZEN arm frees its
8 workers in ~2.5 h.

**Integrity guard added this tick.** The extension writes LIVE seeds 52-71 into
the SAME results directory as the frozen 10-seed evaluation. Verified
`analyze.py` still restricts to seeds 42-51 and that the frozen-protocol numbers
are byte-identical with seeds 53-55 present. Wrote
`results/README-SEEDS.md` documenting the split, because the realistic failure
mode is not misconduct — it is a future reader seeing 30 seed files and
"tidying up" by widening SEEDS, which would silently convert the frozen n=10
evaluation into a post-hoc n=30 one.

---

## 2026-08-24 — The reward phase: a confound resolved, a finding retracted, a theory found

**Where it started.** Three experiments (H9 floor, H10 normalisation, H11 arm C
DT-style RTG decrement) had each voided on their own manipulation check. Two
structural facts, both provable without compute, explained why:

1. `target = max(batch_max, alpha*running_max)` with `batch_max <= running_max`
   caps the RTG band at `1/alpha` — 2x at `alpha=0.5`, **always**. H10 measured
   1.76x and 2.59x: at the ceiling. Two lines of algebra would have prevented
   both experiments.
2. Under `rollout_reward="improvement"` an LF step earns **exactly 0.0**, so
   **63.0%** of trajectories carried `rtg[0]=0`. A signal that is identically
   zero cannot be made to vary by rescaling it.

The long-open "inert vs starved" confound resolved to **starved**.

**A wrong turn, and the lesson.** I built two new rewards (H12 `kg_incumbent`,
H13 `kg_signed`/`kg_topk`) before specing the codebase. `mes_entropy` already
computed `log(b_0) - log(b_T)` = `H[y*|D_0] - H[y*|D_T]` = the **joint
set-level information gain** — better-motivated than anything I built, and it
dominated all my variants on signal health (0.0% vs 63.0% dead; CV 0.66 vs
1.96). **Spec before building.**

**A retraction.** `mes_entropy` had been abandoned on a gate reading
+0.191 (z=2.63, p=0.0085) for `improvement` vs +0.129 (p~0.32). H15 showed the
gate is **within-group Spearman(rtg[0], f_hf(x_0))** — *step-0 greediness*, which
`improvement` satisfies by construction — and that it **does not reproduce and
reverses** on current code. Cause: it was measured in the same commit as the
RTG-cap fix, a code state that no longer exists. H16 confirmed at 10 seeds:
`mes_entropy` leads on all three axes, PRIMARY +0.0962 paired, Wilcoxon
p=0.0195, 8/10 seeds. `improvement`'s return is *negatively* correlated
(-0.0246) with the best point its own trajectory visited.

**The theory.** Brandfonbrener et al. (NeurIPS 2022, arXiv:2206.01079) give
RCSL's necessary conditions. MF-DRO violates them by construction: the rollout
transition is a **draw from the GP posterior**, so their Corollary 1
near-determinism condition fails maximally, and their Figure 1c shows the bias
then survives **regardless of the conditioning function**. That is precisely
what seven independent pre-registered interventions found (H4, H5, H8, H9, H10,
H11, H12-H16). Their return-coverage condition `P_beta(g=f(s)|s) >= alpha_f` is
our 63%-dead-signal result in the theory's own language.

**A conflict inside the theory's own conditions.** H18 tested the one
intervention the theory predicts should work — deterministic transitions
(`fantasy_mode="mean"`). Determinism verified (G1, repeat-diff 0.000e+00) but
the diversity gate failed: distinct trajectories collapsed 131 -> 62. My
protocol's reasoning was wrong — I claimed diversity would survive because the
behaviour policy stays stochastic, but the teacher is deterministic given the KO
model, so the fantasy draw was the *dominant* source of trajectory diversity.
**RETRACTED the same day.** That diversity measurement was broken: the signature
was `(rtg, ell)` with no query locations, so rollouts with a dead reward (63%)
and matching fidelity pattern counted as identical. With locations included,
every arm is **200/200 distinct** — diversity does not collapse at all, and the
`eps`/`alpha_f` coupling claim is withdrawn. H18's G1 (determinism verified)
stands; its G2 does not.

**H19 then delivered the strongest negative result of the project.** With
deterministic dynamics *and* full diversity — the regime RCSL theory says should
work — the DT still moves the argmax **0/12**. Near-determinism is therefore not
the binding constraint. The proximate cause is the score-head bottleneck H5
already isolated: swapping `h` changes the argmax 0/12, so nothing reaching `h`
can move the decision. This also tempers the RCSL framing above: the theory
explains why the conditioning-side remedies failed, not why the architecture
does.

**Running:** H17 (frozen evaluation on the joint-MES reward, 10 jobs) and H19.

**Process lessons added:** derive a formula's reachable range before
experimenting on it; put the manipulation check first as a standalone gate and
commit to stopping; condition null-guards on the manipulation passing; never
choose a discriminating metric that can saturate; spec the codebase before
building the fix.

---

## 2026-08-24 (evening) — the programme closes

**H17 answered the research question.** Frozen evaluation on the joint-MES
reward, 10/10 runs, 5876.6 s. MF-DRO/`mes_entropy` **0.4007 ± 0.0475** — the best
MF-DRO number on record, lower in mean than *both* baselines (MI-Greedy
0.5091 ± 0.1266, GP-UCB 1.7934 ± 0.1223), with 2.7× smaller sd than MI-Greedy.
And **not significant**: paired −0.1085 vs MI-Greedy, 6/10 seeds, Wilcoxon
p = 0.432. The frozen success test **FAILS** (0.4481 ≥ 0.3825), as it did for the
first reward (0.5442). Paired sd 0.2339 implies ~40 seeds for 80% power;
`PROTOCOL.md` fixes 10 and was not extended — extending to chase significance is
the exact optional stopping a frozen protocol exists to prevent.

**The mechanism, measured end-to-end.** Relative spread across the 12 real-
iteration states: state 0.2155 → hidden 0.0745 (0.346×) → coefficients 0.0219
(0.294×). A ~10× attenuation, split evenly between encoder and head. A full run's
worth of state change rotates the coefficient vector 2.04° and changes the
decision on **0 of 12** pools — both across ensemble members and across real
iterations. The fidelity head is equally inert (p spans 0.1248–0.1286).

**H20 closed the last escape route.** An MLP score head over `[h;cf]` — no
factorisation through a coefficient vector, manipulation verified (affine
residual 0.0863 vs 0.000000) — still moves the argmax 0/12. The failure is
architecture-independent, and the encoder half of the attenuation is sufficient
on its own.

**Three self-audits this phase, all of which changed what I would have written.**
H5's h-swap compared a state with itself (12/12 identical) — conclusion survived
re-measurement but the evidence had been worthless. H18's diversity signature
omitted query locations, so its ε/α_f coupling claim was withdrawn. H20's first
run failed its own assert, exposing `use_linear_score_head` as a flag that had
never been reachable. Four auto-printed verdicts were also wrong, each keyed on a
proxy rather than the decisive measurement.

**Verdict: the empirical programme is complete.** Eleven pre-registered
interventions, one measured mechanism, one answered research question, and the
one lever that remains (more seeds) is forbidden by the frozen protocol.
Proceeding to write-up.

---

## 2026-08-24 (late) — the programme reopened, and the causal story changed three times

The previous entry declared the empirical programme complete. That was premature
in a productive way: six more experiments materially changed what the paper
claims, and three of them were corrections to my own conclusions.

**H27 — the user asked for sliding-window inference.** Implemented
(`inference_context_k`): the DT receives the last K−1 real `(state, rtg, btg)`
triples plus the current one. Gate: context genuinely grows (1→6 across
iterations, verified per-iteration), and the proposals are **bit-identical**
(max |Δx| = 0.000e+00), fidelity choices identical, same final regret. Stopped
per protocol before spending the 2-hour frozen run. One loose end recorded and
**not** resolved: a *synthetic* history (mixed ensemble blocks, fabricated
rtg/btg) moves `h` to cosine 0.803 and `w` by 19.9%, so the tokens are
demonstrably processed — yet the real history changes nothing. Flagged rather
than folded into the conclusion.

**H28 — retraction.** I had written into the paper that the student "inverts the
sign of its teacher's defining term". The MF-MES teacher's own choices sit at the
**2.9th percentile** of `sigma_H` (control: 94.2nd percentile of `mu_H`, 1600
decisions). The student inverts nothing; it imitates. Its *score* mildly rewards
uncertainty (+0.1585) but its *argmax* is dominated by the posterior mean
(+0.8517), and in a GP the high-mean region is where data already sits.

**H29 — intrinsic, not tuning.** Cost ratio {2,4,8,16} × y* samples {5,10,50}:
the chosen-`sigma_H` percentile never exceeds **5.5%** in any of 12 cells. Ten
times more Monte Carlo changes nothing; an 8× cost swing moves it ~1%→~5%.

**H30 — correction.** H29's framing ("a student imitating *choices*") was wrong
about the mechanism: the loss is soft-KL to the teacher's *scores* over the full
K=20 set. So why does a student fitting positive-`sigma` scores learn a negative
weight? Collinearity: `corr(mu_H, sigma_H) = −0.4696`, and under it the
**teacher's own** score has a negative partial `sigma` coefficient (−0.0419, on
91.4% of sets). Textbook suppression. The paper's claim that the rule "penalises
exactly what UCB/EI/MES reward" was withdrawn — MES's own partial coef is
negative too.

**H32 — my locked prediction failed.** I predicted a lossy distillation (median
teacher-rank of the student's pick worse than 10 of 200). Measured: **median 2**,
range [1,12], with the student choosing the teacher's top candidate on 5 of 12
pools. The distillation is *faithful*. My "lossy" read traced to H24's argmax
agreement — a poor metric when the top candidates are near-ties.

**H33 — the one genuine divergence.** The fidelity head outputs `p` spanning
**0.5570–0.5577** (sd 2.4e-4) while the teacher's `ell` varies;
`corr(p, teacher ell) = +0.155`. Uninformative, and 0.557 against the teacher's
4.2% on identical pools. I wrote a caveat blaming `minimum_hf_fraction` and
withdrew it minutes later — that flag never touches the real proposal path.

**Where it leaves the argument.** MF-DRO reproduces its teacher's *location*
choices closely, fails to reproduce its *fidelity* choices, adds no conditioning,
and inherits an exploitative demonstrator. Redundant on location, uninformative
on fidelity. H31 (teacher-only under the frozen evaluation) is running to test
the expectation recorded in H32 before it lands: near-parity.

**Process note.** Four errors this arc were caught by the same move: checking
what a component *does* rather than what it is *specified* to do. Two more came
from using a proxy metric instead of the decisive one.

---

## 2026-08-24 (night) — H31 lands and moves the thesis up a level

**The teacher-only control.** Ran `compute_joint_mf_mes`'s argmax under the
frozen evaluation with no DT deciding — DT still trained so the RNG stream and
therefore every candidate pool matched exactly. 10/10 in 8900 s.

    MF-MES teacher, no DT : 0.4781 +/- 0.0414
    MF-DRO / joint MES    : 0.4007 +/- 0.0475
    paired +0.0774, teacher better on 3/10, Wilcoxon p = 0.2324

**H32's expectation, locked before H31 could land, is confirmed**: near-parity,
predicted from the faithful location distillation (teacher-rank median 2 of 200).
The "transformer is a net negative" hypothesis is **refuted**.

**The consequential number: the teacher alone also fails the frozen bar**
(mean+SE 0.5195 vs 0.3825). It is not the transformer that cannot clear it — the
whole MF-MES-based approach cannot, distilled or not. That relocates the negative
result one level up and explains why eleven interventions on the DT could not
have helped: all of them operated below the binding constraint. The paper's
thesis was restructured around this.

**A correction to H33.** Fidelity mix over full runs: teacher 11.4% HF over 112
iterations, MF-DRO 26.7% over 70.7 — 2.3x the rate, 37% fewer queries. H33 called
the head's *level* a defect; here that mismatch coincides with slightly *lower*
regret, so the implication of harm is withdrawn. Only the uninformativeness
stands (sd 2.4e-4, corr 0.155).

**A statistics error of mine, caught by the user.** I reported the cost-weighted
reward comparison (p = 0.0371) as "significant". It is not usable: cost-weighted
regret is not the pre-registered metric and I computed it only after the
pre-registered one failed. Bonferroni over the two metrics gives 0.025 — it does
not survive; over the seven tests actually printed in that table the threshold is
0.00714 and **none** survive. Retracted from the paper, findings.md and the H37
analysis; what remains is the descriptive fact (lower on 9/10 seeds, lower at
every checkpoint), stated as descriptive.

**And a check on the standard itself.** Every conventional test also fails
against MF-MI-Greedy: paired Wilcoxon 0.4316, paired t 0.3518, Welch 0.4387, 95%
CI on the paired difference [-0.3250, +0.1080] contains zero, Cohen's dz 0.311.
82 seeds would be needed for 80% power. The frozen criterion is not the obstacle.

**ETA record.** H31 estimated at 15--40 min, actual 148 min. Third miss on this
class; the binding constraint is the LF-heavy seed, since cost-budgeted
wall-times are set by the fidelity mix.

---

# 2026-08-25 — H57 through H61: the baseline comparison, and five retractions

## Why each experiment was chosen

**H57 (baseline comparison, 36 runs).** Requested directly: MF-DRO vs Takeno
MF-MES, MF-MI-Greedy, MF-GP-UCB on Currin/Hartmann/Borehole, budget 200, seeds
44/46/48. Chose h48's standalone Takeno MF-MES over `GreedyMFMESOptimizer`
because h47-variant-d had measured the latter's 200-point random pool finding an
acquisition value 4.3x worse than a 4000-point one. Pinned to one commit with
the hash in every result file after the peer session pointed out that 36 jobs
across a moving `src/` would be uninterpretable later.

**H58 (HF floor).** `minimum_hf_fraction=0.25` is applied at exactly one site,
inside `simulate_mf_trajectory` — the comment says "forced HF to ensure training
diversity". Nothing enforces it on real queries. The DT is trained under a floor
and run without one. Ran it because the incumbent moves only on HF and two of
h57's stalls had zero HF queries.

**H59 (SF-DRO baseline).** Chosen after noticing the north star is defined
against SF-DRO and `grep -rl DirectRegretOptimization experiments/` returned
nothing — h1..h58 all measure the multi-fidelity extension. You cannot claim a
novel method improves on SF-DRO without knowing where SF-DRO sits.

**H60 (confound decomposition).** Forced by a question I could not answer: on
Borehole MF-DRO is 99-100% HF post-init, so it should behave like SF-DRO — why
is it worse? Enumerating the configs showed four simultaneous differences.
Borehole chosen as the test bed precisely because fidelity is inert there.

**H61 (teacher optimiser).** h60 excluded the reward schema and the LF initial
design, and showed the teacher is load-bearing (20pp swing). Reading the code
found SF-DRO's `_optimize_acquisition` doing 1000 broad samples + 100-sample
local refinement where MF-DRO's `compute_joint_mf_mes` does a flat argmax over
200 points with no refinement at all.

## Decisions that cost something

- **Killed h57's launcher expecting its `subprocess.run` children to survive.**
  They did not — six MF-DRO workers died with the parent, ~20 min lost. Asserted
  the behaviour in a comment instead of checking it.
- **Relaunched h59 with the full job list after fixing one arm.** The
  results-file guard skips only FINISHED jobs, so nine running jobs were
  duplicated — 20 workers against a 15 cap, and the duplicates swallowed SIGTERM.
  Third occurrence of this defect (h56, h57, h59); every runner now takes arms
  from argv.
- **Wrote H61's arms at 1000 points without costing them.** `compute_joint_mf_mes`
  is O(N) per rollout step, so the arms would have been ~7.7 h and ~15 h per
  seed. Amended to 200+100 and 600 before any run, with the weakened inference
  recorded.

## Five retractions, and what caused each

| retracted | cause |
|---|---|
| h45 reported at 5/6 then 7/8 as "regression head winning" | at 10/10 it is worst-on-mean, p=1.0000. Reported a subset while the rest was running. Became lesson 19. |
| "MF-DRO searches misdirected on Hartmann" | distance to x* is not a value proxy: best value within 0.3 of x* equals best beyond 1.0. Lesson 20. |
| "Hartmann's LF is a more attractive objective" | the MES reward scores LF by information about the **HF** optimum; LF's own value enters nothing. corr(f_LF,f_HF)=0.925 and the LF-heaviest seed is the BEST. Lesson 21. |
| "dropping multi-fidelity helps DRO" | confounded — the two configs differ in surrogate, teacher, reward and init, and on Borehole fidelity provably did not act. Lesson 22. |
| three geometric accounts of Borehole | each refuted by checking against value rather than geometry. Lesson 23. |

The common shape: a statistic that is real and significant, attached to a
mechanism that was never checked for whether it *could* act.

## Standing state

All of h57 (36), h58 (12), h59 (18), h60 (9) complete. H61 running.
Neither DRO variant meets the north star on any benchmark. Two candidate causes
remain for the Borehole gap: teacher optimisation quality (H61, running) and the
surrogate class (KO-GP vs SingleTaskGP, not a flag, untested).
