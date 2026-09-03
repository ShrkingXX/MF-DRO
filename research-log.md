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

---

# 2026-08-25 later — H62 through H65: elimination, one positive, and a user hypothesis

## Why each was chosen

**H62 (SF-DRO traces).** Not a hypothesis — a capability gap. h59's SF-DRO cells
carried `n_queries: 0` from a silent instrumentation bug, so SF-DRO was the only
method that could not go through `freeze_watch` or `stall_diagnose`, while being
the method the north star is defined against. Every diagnostic conclusion to that
point was MF-DRO-only. Gated on reproducing h59 to 1e-9; it reproduced to
**0.00e+00** on all 9 cells.

**H63 (KO misspecification).** The user's hypothesis, and the sharpest measurement
in the session: KO assumes `f_H = rho*f_L + delta` with `rho = sigmoid(...)`
confined to (0,1), but Borehole's true OLS slope is **1.2566** with residual sd
**0.0001** — the relation is essentially exact and the model cannot represent it.
Hartmann's 0.9792 is inside the range, making it a control.

**H64 (pool generalisation).** h61's POOL600 improved Borehole 3/3. Whether that
is a general teacher defect or Borehole-specific decides whether it is a
contribution or a footnote.

**H65 (refine generalisation).** Added after noticing **h64 tests the wrong arm**
for the question h61 actually raised. h64 generalises POOL600 (pool size), whose
null prediction rests on a coverage argument. REFINE is not a pool-size change —
its liveness check showed it *widening* query spread while matching POOL600's
mean — so h64's prediction says nothing about it.

## Decisions that cost something, or nearly did

- **Wrote H61's arms at 1000 points without costing the O(N) rollout call.**
  Would have been ~7.7 h and ~15 h per seed. Amended to 200+100 and 600 before
  any run, with the weakened inference recorded rather than glossed.
- **Predicted REFINE would be the cheap arm.** It is the expensive one — 3.3x
  BASE vs POOL600's 1.6x — because `compute_joint_mf_mes` has per-*call*
  overhead and REFINE makes two calls per rollout step. Cost scales with
  acquisition *calls*, not candidate count.
- **Nearly reported h63's Borehole arm as support.** Its own protocol had
  pre-declared that a Borehole win alone is insufficient and only a
  larger-than-Hartmann gain discriminates. Withheld.

## Predictions I registered and then lost

| prediction | outcome |
|---|---|
| rho saturates near 1 on Borehole | **wrong** — fitted rho is 0.8436, short of even the representable range |
| pool-size gain orders 8D > 6D > 2D | **wrong** — it orders 8D > 2D > 6D; Hartmann gains nothing at N=600 |
| refinement concentrates the teacher's demonstrations | **wrong** — it widens query spread (0.2907 vs 0.2534) |
| Currin's blow-up is steep saturation near the optimum | **wrong** — Currin is the flattest; the ratio is a denominator artifact |
| h58's HF floor would be harmful (budget reallocation) | **wrong** — it improved the starved seed 22% |
| **h61: REFINE beats BASE >=2/3, POOL600 between** | **MET** — 3/3, and 73.40 > 60.23 > 59.75 |

One met prediction in six. The met one is also the only durable gain.

## Standing state

h57 (36), h58 (12), h59 (18), h60 (9), h61 (6), h62 (9) complete. h63 4/6,
h64 1/6, h65 0/3.

Best result: h61 REFINE on Borehole — 23.7% -> 19.3%, 3/3, and **seed spread
8.62 -> 2.57**. Still loses to MI-Greedy's 8.3%. The variance collapse is the
finding; the mean gain is matched by POOL600 at 10x the spread.

# 2026-08-26 — H66 through H73: the withdrawal, and the discovery that n=3 was never enough

The session with the most retractions and the least new capability. Two claims
withdrawn, one measurement corrected, one experiment cancelled before launch, and
one methodological finding that calls the project's headline table into question.

## Why each was chosen

**h66** — replicate h64's north-star claim at n=10. Chosen because h64's 2/3 on
Hartmann was the only time anything in this project beat a baseline, and lesson
19 exists precisely because this project has repeatedly reported a favourable
subset. Analysis script committed while POOL600 stood at 0/7.

**h67** — unbounded rho via softplus, the "repair" h63 pointed at. Built, gated,
and **cancelled before launch** when a three-minute pre-flight refuted its premise.

**h68 / h68b** — separate OPTIMIZER failure from MODEL failure on Borehole, using
traces already on disk. h60 had left exactly two candidates and they were
separable offline.

**h69** — split "model failure" into surrogate vs acquisition with an SF-EI arm
differing from SF-MES in the acquisition alone, behind a bit-for-bit gate.

**h70 / h70b** — the last identified differences: candidate pool size and GP
construction. Then n=10 when h70's Hartmann result turned out to be unpredicted.

**h71** — whether MF-DRO's own teacher pool is load-bearing the way the
baselines' inference pool is. Still running.

**h72** — calibrate the h57 standings against their own n=3 noise, by enumerating
all C(10,3)=120 three-seed subsets of a 10-seed run. Chosen because lesson 26's
corollary is that the headline table is itself n=3.

**h73** — whether SF-DRO's Hartmann advantage over SF-MES, the strongest pro-DRO
result in the project, survives n=10. Still running.

## What was withdrawn

| claim | how it died |
|---|---|
| **h64's north-star arrival** (POOL600 beats MF-MES on Hartmann) | h66 at n=10: 5/10 wins, p=1.0000. The +0.0530 mean advantage was **one seed** — drop s49 and it reverses to −0.0270 |
| **h68's k=10 policy/acquisition mismatch** | 12 replications: the reported 8.7th percentile has a median of 79.4, range 41.2–91.3. A single Monte-Carlo draw reported as a measurement |
| **h70's "KO-style GP costs 6.41 pts on Hartmann"** | h70b at n=10: KO-style is **better** by 3.03, 8/10. Sign reversed |

## What was corrected

**"MF-DRO ran at 5-10x less acquisition-optimisation effort."** Published one
tick, corrected the next: `n_roi_candidates` is the *rollout teacher's* pool.
`_propose_next_query` builds no candidate set, and the regression head emits
`action_head(h).clamp(0,1)` directly — MF-DRO does **zero** inference-time
acquisition search. Different mechanisms, not different sizes of one.

## Decisions that cost something, or nearly did

- **Committed source edits while a script was swapping HEAD files into the tree.**
  `f50bc0b` claims to implement softplus and contains none of it. Left in place
  rather than amended, with the real edits in a follow-up.
- **Launched h71 with `results/ckpt/` absent.** `_atomic` doesn't mkdir, so every
  checkpoint raised inside a daemon thread and died silently. Runs were healthy
  and finals unaffected; 19 minutes of progress kept rather than restarting.
- **h68 scored `x_dro` and `x_mig` in one acquisition call**, letting MI-Greedy's
  strong point raise the shared `y*` and depress `x_dro` — biasing the exact
  comparison the analysis rested on.
- **Two protocols whose bars could not fail.** h68's PRIMARY was necessary but not
  sufficient for its conclusion; h65's "spread < spread" passed on a 1.1%
  contraction. Both passed while the substantive claim failed.

## Predictions registered and lost

| prediction | outcome |
|---|---|
| h66 PRIMARY: POOL600 beats MF-MES >=6/10 | **NOT MET** — 5/10 |
| h68 SECONDARY: x_mig outranks x_dro >=5/9 | **NOT MET** — 1/9 |
| h68b: mismatch tracks per-benchmark standing | **MISSED** — ordering was Currin > Borehole > Hartmann |
| h69 PRIMARY: EI beats MES on Borehole by >=2 pts | **NOT MET** — +0.29 |
| h69 CONTROL: EI worse on Hartmann | **VIOLATED** — EI better on both |
| h70b PRIMARY: plain GP better on Hartmann | **NOT MET** — reversed |
| h70 all three bars | **MET** — the only clean sweep |
| h72 both bars | **MET** |

## Standing state

h66, h68, h69, h70, h70b, h72 complete. h67 cancelled pre-launch. **h71 (6 jobs)
and h73 (7 jobs) running**, 12.6 of 15 cores.

**North star: not met, and never was.** The only claim that ever cleared it is
withdrawn. Nothing in this project beats the baselines on any benchmark that
discriminates.

**Borehole is explained — for the baselines.** MI-Greedy's 5-point advantage over
SF-MES is *entirely* candidate pool size: SF-EI at 1000 candidates reproduces
MI-Greedy **exactly, seed for seed**, residual +0.00. The GP construction
contributes 0.00 there despite the builders differing materially.

**Hartmann is not resolved.** h72: a three-seed draw of MF-GP-UCB could have
landed anywhere in [36.7, 88.5]; MI-Greedy in [22.8, 48.8]. Published entries
were optimistic by +12.7 and +21.5 points. MF-DRO and SF-DRO remain uncalibrated
at 82–473 min per run — and Hartmann is the column the withdrawn claim lived in.

# 2026-08-26 (later) — H74 through H77: the scope of the one positive, and closing calibration

## Why each was chosen

**h74** — h73 established SF-DRO beating SF-MES on Hartmann at n=10. That is one
benchmark, and the n=3 losses on the other two were exactly as unreliable as the
n=3 win had been. Generalisation was untested in both directions.

**h75** — MF-DRO was the last uncalibrated method, and h71's PRIMARY compares
POOL1000 to a BASE that is also n=3. A properly estimated BASE makes h71
interpretable rather than suggestive.

**h76** — *when* the Hartmann advantage appears, measured rather than proposed.
Six mechanisms had already been proposed and refuted, so this one asks only for a
descriptive fact that constrains future mechanisms without asserting one.

**h77** — MF-DRO's Hartmann cell, the last uncalibrated entry, sitting in the
column h72 showed to be least resolved.

## Results

| experiment | verdict |
|---|---|
| **h74** | **NULL fired.** SF-DRO loses to SF-MES on Borehole (2/10) and Currin (2/10). One win, two losses — h73 does not generalise, and the n=3 losses were not noise |
| **h75** | **Both bars MET.** MF-DRO Borehole n=10 = 22.89% vs published 23.71%, shift −0.82, three-seed span 5.89. The published entry held up |
| **h76** | **NULL fired.** Crossing at iteration 18, not <=12. SF-DRO is *behind* early on all three benchmarks; what differs on Hartmann is that SF-MES plateaus and SF-DRO does not |
| **h77** | running |

## Decisions that cost something, or nearly did

- **h73's reproduction control was vacuous** — it reused seeds 44/46/48 from h59
  and so compared h59's files to themselves. Caught, withdrawn, and redone
  directly (PASS, 0.000e+00). h74, h75 and h77 all had the same structural risk
  and each got a real control enforced *in the verdict script* rather than left
  to judgement.
- **A verification that failed open.** `find -newermt "-30 minutes"` was rejected
  by this system's `bfs`; it printed `0` from the error and I reported that `0`
  as evidence h57 was untouched. Redone properly — h57 genuinely untouched.
- **Two sed-induced defects in h77's verdict script** — a `NameError` from a
  missed rename, and stale "Borehole"/"h75" labels in the printed messages. The
  second was a repeat: my first fix patched only the strings visible in the
  output. Fixed by grepping every occurrence.
- **h71 launched without its `ckpt` directory**, so every checkpoint raised inside
  a daemon thread and died silently. Runs were unaffected; monitoring was lost.
  The `_atomic` makedirs fix means h77 is fully observable mid-run while h71
  remains a black box — a direct demonstration of what the defect cost.

## Predictions

| met | grounded in |
|---|---|
| h75 (both bars) | h72's measurement that Borehole shifts little at n=10 |

| lost | grounded in |
|---|---|
| h74 PRIMARY | that h73 would generalise |
| h76 PRIMARY + SECONDARY | intuition that the advantage was early search |

This extends the pattern recorded as **lesson 28**: every met prediction in this
project derives from a prior measurement; every lost one is a mechanism intuition.

## Standing state

h74, h75, h76 complete. **h71 (4/6 — all three Borehole cells in, PRIMARY fully
determined, held for the two Hartmann CONTROL cells) and h77 (5/7) running.**

**North star: not met.** SF-DRO beats its own MES counterpart on one benchmark of
three; against the strongest Hartmann baseline it is a tie (4/10, p=0.43, post
hoc). Borehole is fully calibrated at n=10 and MF-DRO is last among the
non-degenerate methods there — not a three-seed artifact.

## h79 / h79b — one claim, three weakenings

h70's sharpest result was checked twice more, cheaply, and narrowed each time.

**h79 (n=10, seconds of compute).** The SF-EI@1000 == MI-Greedy identity holds
bit-for-bit on **8 of 10** Borehole seeds, not all. Seeds 45 and 49 diverge by
1.23 and 3.00 points with the pool-matched baseline **worse**. h70's three seeds
were among the eight that match. PRIMARY and SECONDARY both NOT MET, NULL fired.

**h79b.** The trajectories were never the same. Both methods split at iteration
1-2 on **all four** seeds examined, *including the matched controls* whose finals
agree to ten significant figures. Starting regrets differ (seed 46: 34.109 vs
36.597). Both of h79b's predictions were wrong, and the NULL fired with a stated
conclusion that did not follow — the protocol had assumed matched endpoints
implied matched trajectories.

**Descriptive close-out.** On matching seeds the two arrive at the same value at
different iterations (80 vs 96; 26 vs 31), so they find it independently. On
divergent seeds one plateaus early — seed 45 has MI-Greedy reaching its final
value at **iteration 2** and never improving across the remaining 98, yet still
finishing ahead.

**Final form:** giving single-fidelity EI a 1000-point pool is enough to reach
MI-Greedy's final regret on 8 of 10 Borehole seeds. A claim about **outcomes**.
The 4.72-point gap really is closed by pool size; the methods are not equivalent.

**Recorded as lesson 29:** the strength of the word must match the strength of the
check. A protocol-design corollary from h79b — a control that checks only the
endpoint cannot detect two different paths to the same endpoint, which is exactly
what the original claim rested on.

## 2026-08-27 — H84: ROI strategy for MF-DRO budget waste

DIRECTION: DEEPEN. h83 established MF-DRO does not beat the best baseline
anywhere (PRIMARY met on all four benchmarks) and diagnosed WHY: its mean HF
query is worth 0.336 against MF-MES's 0.747 on Hartmann, with 20.8% of HF
queries landing worse than the initial design. The DT regression head is trained
on rollout-teacher actions drawn from `roi_candidates`, so the ROI is the lever
that shapes the policy's proposal distribution.

TWO BUGS FOUND AND FIXED FIRST (both in src/policy/mf_dro.py):

1. The ROI candidate pool filtered a fixed 2000-point draw and then padded back
   to n_roi_candidates WITH REPLACEMENT, so distinct candidates were
   min(N, raw*accept) rather than N. Hartmann at sqrt(beta)=2: 231 distinct with
   the ROI on vs 600 with it off. Every prior ROI-on-vs-off comparison varied
   region AND resolution together, so "the ROI does not work" did not follow.
   Fixed by rejection-sampling to a fixed distinct count.
2. Latent: the teacher_refine_samples branch concatenated into `roi_candidates`,
   a variable defined OUTSIDE the rollout loop, permanently growing the shared
   pool every step. Now a local.

The recorded reason for deleting ROI ("zero rollout steps within L2=0.2 of the
optimum") also does not hold: the UNFILTERED pool fails it too -- the closest of
2000 uniform points in 6-D is 0.243 from x*. That is dimensionality, not the ROI.

STRATEGY: a constant beta is unusable. Measured acceptance at sqrt(beta)=2
swings 250x on Borehole (100% at n_hf=10, 0.4% at n_hf=35) -- vacuous exactly
when the surrogate is worst, then collapsed once data accumulates. H84 therefore
controls the ACCEPTANCE RATE and solves for beta_t by bisection, keeping the
paper's ROI set exactly as written and setting only the parameter the paper
subscripts by t but never specifies. Hits q=0.10 exactly on all four benchmarks
with beta_t spanning 0.31-12.29.

Arms: ROI-OFF (h83 reuse, bit-identity gated + live reproduction control),
ROI-FIX2, ROI-Q10, ROI-ANN. 34 runs. Four independent pre-registered
predictions incl. one NEGATIVE about the paper's rule as literally parameterised
and an explicit falsification condition for the whole strategy.

HUMAN DIRECTION: if ROI improves MF-DRO, re-run ALL of h83's MF-DRO arm (4
benchmarks x 5 seeds) with the winning configuration.

## 2026-08-27 (later) — H84 outcome, and what it cost to get right

RESULT. The quantile-calibrated ROI improves MF-DRO's final regret on both
benchmarks tested, decisively on one:

  Borehole_8D  ROI-Q10  -4.22 pts (better 5/5)  15.82% -> 11.59%
  Hartmann_6D  ROI-Q10  -1.62 pts (better 3/5)   7.55% ->  5.95%

P1, the pre-registered PRIMARY bar (+0.10 mean query score on >= 4/5 seeds, BOTH
benchmarks), FAILED: Borehole met it (+0.114, 5/5), Hartmann did not (+0.001,
3/5). Reported as failed; the bar was not renegotiated.

The strongest argument for the contribution is NOT the regret delta. It is that
ROI-FIX2 -- the paper's rule at a fixed sqrt(beta)=2 -- realises 24.9%
acceptance on Hartmann purely as a consequence of that benchmark's posterior
scale. Nobody chose 24.9%. Measured acceptance at fixed beta swings 250x across
benchmarks and data sizes (Borehole 100% at n_hf=10, 0.4% at n_hf=35), so a
constant beta cannot express "tight" at all. Calibrating to a target acceptance
turns ROI tightness into a knob, and h84 shows the knob matters: tight (q=0.10)
beats every looser setting on both benchmarks.

WHAT THIS DOES NOT DO. It does not make MF-DRO competitive. MF-MES still wins
Borehole by 5.2 points after the improvement. The h83 headline -- MF-DRO beats
no baseline on any benchmark -- is unchanged by h84.

FOUR CORRECTIONS I HAD TO MAKE DURING THIS EXPERIMENT, all recorded in
findings.md rather than quietly amended:

  1. "An L2 head cannot reach a bound" -- WRONG. It saturates clamp on 2.02% of
     Borehole coordinates.
  2. "The ROI relocates a uniform draw and cannot reach a corner" -- WRONG. All
     three boundary metrics improve monotonically as the ROI tightens. I had
     reasoned about the sampling distribution instead of measuring the filtered
     one.
  3. "Regret improves monotonically with ROI tightness" -- OVERSTATED. Held at
     n=1 on Hartmann's loose arms; at n=3 the ordering broke. What survives is
     that tight beats loose, not a strict ordering.
  4. My twice-registered prediction that P1 was unlikely to be met -- REFUTED on
     Borehole 5/5. I had measured the ROI's effect on the TEACHER at one model
     state (+0.010) and treated it as an upper bound on its effect on the POLICY,
     which trains on the teacher's whole distribution across rollouts.

The common failure mode in 1, 2 and 4: asserting a mechanism from a plausible
argument when a cheap measurement was available the whole time.

TWO PROCESS LESSONS (findings.md 21, 22):
  21. A control that can VOID an experiment must run FIRST. h84 queued its
      reproduction control behind all 30 treatment runs.
  22. The PRIMARY metric must be the statistic the objective depends on. Simple
      regret is a MAX; h84 registered the MEAN as primary, so on Hartmann the
      bar recorded +0.000 while regret improved 1.62 points.

STATUS: 22/34 done, 0 failures. Reproduction control 1/4 passed bit-identical
(Hartmann s43, |dregret| and max|dx| both 0.000e+00); 3 running. h86 (ROI on
Currin + Ackley, 10 runs) is written and GATED on all four controls passing.
h85 (teacher refinement + HF floor) is written and queued behind compute.

## 2026-08-27 — H84 and H86 complete (44 runs, 0 failures). What the ROI actually bought.

RESULT, four benchmarks, h83's own metric and bar:

  benchmark      MF-DRO h83   +ROI-Q10    delta   wins vs baseline   verdict
  Currin_2D            0.01       0.13    +0.11        0/5           HARMED
  Hartmann_6D          7.99       5.93    -2.05        4/5           BEATS MF-MES
  Borehole_8D         15.82      11.59    -4.22        1/5           helps, still behind
  Ackley_10D           3.83       3.74    -0.09        1/5           negligible

The ROI helps two benchmarks, does nothing on one, and HARMS one. h83's PRIMARY
finding -- MF-DRO beats no baseline anywhere -- no longer holds on Hartmann.
"The ROI fixes MF-DRO" over-reads by two benchmarks.

REGISTERED BARS: h84 P1 FAILED, P2 REFUTED, P3 MET, P4 FAILED. h86 P1 FAILED,
P2 MET, P3 FAILED, P4 met only technically (and the bar was badly written -- an
absolute 1-point threshold records "no harm" on a benchmark whose regrets are
~0.01 while the data show harm on 5/5 seeds). Six of eight registered bars
failed or were refuted. The results that survived were mostly NOT the ones
predicted.

THE CLAIM THAT DOES NOT DEPEND ON ANY OF THIS. Fixed beta cannot set ROI
tightness, along three measured axes: across benchmarks (12.6%-100%
acceptance), within a single run (250x on Borehole), and across SEEDS of the
same benchmark (6.9x on Hartmann, 3.6% to 24.9%). Calibration collapses all
three to 1.0x while solving beta_t anywhere in 0.48-12.29. This is an argument
about controllability, not performance, and it holds whichever benchmarks
benefit.

METHODOLOGICAL COST OF THIS PAIR OF EXPERIMENTS -- worth recording because it
was large. Six refuted mechanism claims (Lesson 23), four of them erring by
UNDERESTIMATING the intervention, each from an argument about why something was
impossible rather than a measurement of its size. Two process lessons: controls
must run FIRST (21), and the PRIMARY metric must be the statistic the objective
depends on (22). Three corrections to numbers I had already published or
written down, including ROI-FIX2 on Hartmann moving +6.32 (n=2) -> +0.36 (n=3)
-> -0.26 (n=5) across three separate reports of the same arm.

NEXT: h87 is running -- the clean confirmation of the Hartmann flip at fresh
seeds 47-51 with q=0.10 fixed in advance and one arm only, carrying a falsifier
that withdraws the flip if it fails. Its comparator finished first (Lesson 21)
and revealed that seeds 47-51 are 3.27 pts harder for MF-MES and three times as
dispersed, which was recorded before any treatment run completed.

## 2026-08-27 (evening) — H85 and H88 complete. The lever was never the ROI.

The session's PRIMARY QUESTION was an ROI strategy to stop MF-DRO wasting HF
budget. After four experiments the answer is that the ROI is not the lever, and
two other things are.

### H88 — the surrogate is the limit, not the data

Same KO GP, same 4096-point recommender, three datasets:

  fit on MF-DRO's queries   Hartmann 13.15%   Borehole 22.66%
  fit on MF-MES's queries            19.92%            21.42%
  fit on a Sobol design              29.03%            22.64%
  (Sobol's own best obs              65.51%            31.90%)

Every recommendation lands in 11.96-29.03% while the best OBSERVED points reach
0.67-19.19%. The KO surrogate's global argmax is a poor point regardless of what
it is fit on. This closes off "recommend from the model" and bounds what any
data-side intervention can buy. It also refutes "good model, bad policy" as the
bulk explanation -- with the correction that the model DOES beat its own best
query on 1 of 10 runs under a strong recommender, where the live 512-point one
showed 0 of 20.

### H85 — teacher refinement is the strongest intervention measured

  Borehole  REFINE-100  -5.85 pts, better 5/5   (15.82% -> 9.96%)
  Hartmann  REFINE-100  -1.93 pts, better 3/5   ( 7.99% -> 6.05%)
  Hartmann  HF-FLOOR    -1.37 pts, better 3/5;  sd 5.85 -> 2.00
  Borehole  HF-FLOOR    +0.18 pts, 0/5 -- non-binding, 4 runs bit-identical

Bars: P1 MET (disproportionality), P2 MET (mechanism: near-bound fraction
8.93% -> 16.32%), P3 MET, P4 MET narrowly (1.89-1.97x wall-clock against a 2x
bar), P5 MET (variance), P6 REFUTED, P7 confirmed literally. Reproduction
control PASSED 4/4 bit-identical.

**This is the one mechanism claim of mine that survived the session.** Measured
before the runs, at a matched model state: the ROI moves teacher action quality
+0.010, refinement moves it +0.046. I argued the teacher's flat argmax over
uniform random candidates was the binding constraint rather than the region
those candidates come from. P1 and P2 both confirm it, and P2 was written so it
could fail even if regret improved.

### The scoreboard for my own predictions

Seven mechanism claims made this session. SIX refuted:
  1. An L2 head cannot reach a bound.
  2. The ROI cannot reach a corner.
  3. Regret improves monotonically with ROI tightness (then: tight beats loose).
  4. h84 P1 unlikely to be met (refuted on Borehole).
  5. h84 P2: fixed beta buys nothing on Borehole (it posted the best regret).
  6. h85 P6: the HF floor will not improve the Hartmann mean.
ONE survived: refinement > ROI, and for the registered reason.

Four of the six refuted claims erred by UNDERESTIMATING an intervention, each
from an argument about why a mechanism was impossible rather than a measurement
of its size. The seventh -- the one I got right -- came from a measurement.

### Standing position

h83's headline STANDS: MF-DRO beats no baseline on any of four benchmarks. The
Hartmann flip was announced and withdrawn (h87, 2/5 at fresh seeds). Refinement
narrows Borehole from 9.42 to 3.56 pts behind MF-MES but does not close it.

NOTHING from h85 is a finding yet. h89 is running: both interventions at fresh
seeds 52-56, treatments hardcoded, controls first, falsifiers requiring
withdrawal. That experiment exists because h84's claim carried four correct
caveats, was announced anyway, and did not survive.

## 2026-08-27 (evening) — H91/H92: the founding diagnosis was half seed noise

Two five-run experiments, both pre-registered with their bars and consequences
stated before running, settled the question the whole investigation rested on.

The premise quoted in every task prompt -- "MF-DRO's mean HF query score is 0.336
vs MF-MES's 0.747 on Hartmann, 20.8% of its HF queries land WORSE than the
initial design" -- was measured entirely on seeds 42-46. H91 ran MF-MES on
Hartmann at seeds 52-56; H92 ran it on Borehole. Pooling each benchmark to n=10:

  benchmark      n   MF-DRO   MF-MES   paired d   median d   MES better
  Hartmann_6D   10     5.32     6.84      -1.52      +0.22       5/10
  Borehole_8D   10    15.42     8.24      +7.18      +8.30       8/10

**Hartmann's deficit is not real.** Five of ten, median +0.22. On seeds 52-56
MF-DRO scores 2.66 against 7.99 at 42-46, and wastes 4.2% of its budget rather
than 20.8%.

**Borehole's is.** Eight of ten, and the median gap EXCEEDS the mean, so it is
not one bad run dragging an average. It holds at both seed sets (+9.46, +4.95).

WHAT THIS MEANS FOR h84-h90. The intervention programme was aimed correctly:
Borehole is where MF-DRO genuinely loses, where the ROI helped most (-4.22, 5/5),
where teacher refinement helped most (-5.85, 5/5 with its mechanism bar
independently met), and where the boundary explanation applies. But the
DIAGNOSIS that motivated the programme was stated on the benchmark whose deficit
does not survive replication.

The corrected problem statement: on benchmarks whose optimum lies on the domain
boundary, MF-DRO's rollout teacher cannot generate training targets there, so the
policy never learns to propose them. Benchmark-intrinsic, which is why it
replicated where Hartmann's seed-specific effect did not.

TWO CLAIMS WITHDRAWN TODAY, both by the same route -- met their bar on five
seeds, carried caveats, died on fresh seeds: the Hartmann ROI flip (h87, 2/5) and
the HF floor's variance result (h89). What separated the surviving results from
these was re-running them, not judgement about which would hold.

STILL OPEN: Currin and Ackley have never been measured at a second seed set.
H90 (Borehole ROI + refinement at seeds 47-51, 15 runs) launched 17:45.

## 2026-08-27 (close) — H89 complete. The session's full arc, and what it cost.

### The question and the answer

PRIMARY QUESTION: find an ROI strategy, using the DRO paper's own heuristic,
that stops MF-DRO wasting HF budget on low-value regions.

ANSWER: the ROI is a lever but the weakest of the three tried, and none of them
makes MF-DRO competitive. h83's headline stands unchanged after 104 runs across
six experiments: **MF-DRO beats no baseline on any of four benchmarks.**

### Three interventions, one survivor

  intervention          first measurement        fresh-seed re-test        verdict
  calibrated ROI        Hartmann flip, 4/5       h87: 2/5                  WITHDRAWN
  HF floor              Hartmann sd 5.85->2.00   h89: sd 2.08->3.17        WITHDRAWN
  teacher refinement    Borehole -5.85, 5/5      h89: -2.11, 4/5           SURVIVES, 36% of size

The ROI's own per-benchmark effect (n=5, control verified bit-identical 4/4):
Borehole -4.22 (5/5), Hartmann -1.62 (3/5), Ackley -0.09 (1/5), Currin +0.11
(0/5, HARMED). It helps two benchmarks, does nothing on one, harms one.

### What is solid

1. **Teacher refinement is real**: -2.11 pts on Borehole, 4/5 fresh seeds, at
   1.25x wall-clock. It does NOT close the gap to MF-MES (12.91% vs 10.07% at
   matched seeds), as pre-registered.
2. **A constant beta cannot set ROI tightness.** Acceptance varies 12.6%-100%
   across benchmarks, 250x within one run, and 6.9x across seeds of one
   benchmark. Quantile-calibrated beta_t collapses all three to 1.0x. This is a
   controllability result and holds regardless of performance.
3. **The surrogate, not the data, limits recommendation quality** (h88). Every
   recommendation lands in 11.96-29.03% while best observed points reach
   0.67-19.19%, whichever dataset the GP is fit on.

### The methodological result, which may outlast the empirical ones

Eight mechanism claims were made this session. SEVEN were refuted. The one that
held -- refinement moves the teacher 4.6x more than the ROI, because the flat
argmax over random candidates is the binding constraint -- was the only one that
came from a measurement taken BEFORE the claim rather than an argument about
why something was impossible.

TWO claims were announced and then withdrawn. Both:
  - met their pre-registered bars at n=5 on one seed set
  - were recorded as PROVISIONAL with accurate caveats attached
  - died when re-run on seeds never used before

**The caveats were correct and did not prevent either announcement.** What
separated the surviving result from them was re-running it. That is the
session's most transferable finding, and it is why h89 and h87 existed at all.

Supporting rules now standing (findings.md Lessons 21-23):
  - a control that can void an experiment runs FIRST
  - the PRIMARY metric must be the statistic the objective depends on
  - measure the quantity a mechanism operates on, under its own conditions
  - n=5 cannot characterise a paired difference here (paired sd 0.45 on one
    seed set became 7.45 on another)

### Open, and honestly open

- h90 (session B) is independently re-testing refinement at seeds 47-51. If it
  disagrees with h89's -2.11, the effect size is unsettled at n=5 twice over.
- The Currin harm (+0.11, 0/5) has a hypothesis (nothing to fix on an
  already-solved benchmark) and no test.
- The RTG/reward signal and the state representation remain untouched by any
  measurement this session.

## 2026-08-27 ~20:00 — CODE AUDIT: the ROI has never been applied to a real query

Not an experiment. A fact about the execution path, found by reading it.

`use_roi=True` restricts `simulate_mf_trajectory`'s `roi_candidates` — the pool
the TEACHER draws demonstrations from. The real query comes from one call
(`mf_dro.py:3090`) into `propose_mf`, which with `use_candidate_scoring=False`
— the setting every experiment here used, because pool+argmax is excluded —
returns `action_head(h).clamp(0.0, 1.0)`. Between that call and the query being
issued, the only modifications are FIDELITY overrides. No ROI symbol appears in
that path at all.

So every ROI result in this project measured the ROI through an imitation
channel: teacher proposes in-ROI → `actions_x` → `L_loc = MSE(x_pred, actions_x)`
→ weights shift → the emitted x moves *somewhat* toward where the ROI was. DRO
Sec 4.2 defines X_hat as a constraint on the QUERY. What was implemented is a
constraint on the DEMONSTRATIONS.

This is the best structural account I have of the ROI's signature — real but
small on Borehole, negligible on Ackley, harmful on Currin, withdrawn on
Hartmann. An intervention filtered through imitation should be weak and
inconsistent across problems, which is what was measured. It also fits h88: MF-DRO
builds a *better* global surrogate than MF-MES and fails to convert it into
queries. The ROI targets exactly that conversion step, and then never touches it.

Three things I am NOT claiming. This does not rescue any withdrawn result. It
does not predict the SIGN of an inference-time ROI — a constraint that excludes
the DT's preferred point can easily hurt, and Currin is already harmed by the
training-only version. And it does not mean the imitation channel carries no
signal; ROI-on and ROI-off runs really do differ.

H94 tests the paper's version. Its arm D — snap to an UNFILTERED pool — exists
because "enforce the ROI at inference" is one bad step from the excluded
pool+argmax mechanism, and I would rather detect that empirically than trust my
own argument that I avoided it.

Implementation written, then deliberately REVERTED out of `src/`: a concurrent
session has workers up and jobs still to launch, and a new worker imports
`mf_dro.py` at process start, so a working-tree edit would reach another
experiment's runs. It lives as a patch verified to re-apply byte-exactly. The
bit-identity gate has NOT been run yet — it needs cores that do not exist right
now, and no H94 result may be reported before it passes.

Also this tick: **withdrew my own compute-cap alarm.** I reported 27 workers on
15 cores and framed it as another session breaching the rule. There were 15 real
workers; the launcher's argv carries sixteen job strings and my per-PID grep
attributed it many times over. Load average 21.7 was consistent with 15 and I
did not reconcile it. Session B's deviation note was accurate as written.
`src/analysis/worker_count.sh` now counts only worker processes WITH job args.

## 2026-08-27 ~21:00 — H95: the founding diagnosis's implied fix points the wrong way

h90 confirmed the ROI lowers final regret on Borehole. That is not what this
investigation was commissioned to test. The commission was "stop MF-DRO wasting
HF budget", and simple regret is a MAX statistic — a method can lower it by
landing one good query while wasting as many as before. So I registered h95's
bars before computing anything and read h90's completed runs.

The average query really does improve: mean HF query regret falls 4.15 points on
5/5 seeds. Waste falls wherever waste exists, roughly halving in all three seeds
that had any; the other two had none in either arm, which is why my M1 bar
failed — I demanded ">=4/5 strictly better" on a measure with a floor that 2/5
seeds already sat on. That is a bar-design failure, recorded as FAILED rather
than rewritten, with a standing rule added.

The finding is M2, which failed in the opposite direction to my prediction. **The
ROI increases dispersion while improving both regret and query quality.** The
concurrent session had measured the same thing independently under a different
statistic and got the same sign; its message arrived in the same tool result as
my registration commit, so my bar was blind but I knew the likely answer before
running my script — recorded, because "registered before computing" and
"registered before knowing" came apart and only the first is strictly true.

This matters beyond one benchmark. The diagnosis I was handed says "proposals
are 3x more dispersed", which invites "concentrate them". h88 already found
MF-DRO's wider queries produce a *better* global surrogate than the method that
beats it. Now the one intervention that demonstrably works increases dispersion.
Dispersion looks like a correlate of the failure rather than its cause, and any
future fix argued from "reduce dispersion" has to explain this first.

h94 is unaffected: it constrains the query to a plausibility region, which is
not a dispersion argument. If anything this raises the value of its SNAP-CONTROL
arm, since snapping to a finite pool is a concentration mechanism and now has a
concrete reason to be viewed with suspicion rather than hope.

## 2026-08-27 ~21:20 — the mechanism, and two corrections from the concurrent session

The concurrent session caught two errors in how I argued h95. I had written "two
independent measurements" for what were two statistics on the same runs, and
more seriously I had refuted a HARTMANN-derived diagnosis using BOREHOLE data.
Both corrected; its neither-necessary-nor-sufficient framing replaces my
"correlate, not cause" throughout, because it is checkable per benchmark and
does not lean on a causal word n=5 cannot support.

Then I applied the same objection to my own h96 relocation account, which was
equally Borehole-only, and registered Hartmann as its falsifier before
computing. It survived, and extending to all four benchmarks gives relocation
present on exactly the one where the ROI works and absent on the three where it
does not — while dispersion moves the opposite way on the two that matter.

The measurement choice decided this. The protocol named the sensitivity-weighted
distance as the deciding measure in advance, on the strength of a prior finding
that unweighted distance reverses the ordering on Borehole. Weighted gives 5/5;
unweighted gives an ambiguous 3/5 and I would have concluded there is no
relocation. That is the h85-P4 rule changing an outcome for the first time
rather than merely being obeyed.

Two limits I am keeping loud. There is exactly one positive case, so the three
negatives are cheap and the whole structure rests on Borehole. And the account
is silent on HARM — Currin got worse with relocation at zero, which no version
of "relocation is the mechanism" predicts.

The useful downstream consequence: on Borehole the ROI improves boundary reach
in dim 0 and barely moves dims 3/5/6, which quantifies why it closes 37% of the
gap and cannot close more. The residual is boundary aversion, and an ROI cannot
fix it because it selects where to look, not what the head can emit. The
output-parameterisation experiment now has a specific target instead of a hope.

## 2026-08-27 ~21:55 — h94 running; and boundary aversion turns out to be a spread problem

h94 launched, but only after failing. Eight jobs, eight crashes, a NameError:
I put the inference hook in `_propose_next_query` and passed `t`, which is
`run()`'s loop variable. The bit-identity gate passed and could not have caught
it — with the flag off, the hook short-circuits and the new code never runs. A
gate on the OFF path is not a gate on the ON path, which is now G3.

G3 then told me something that changes how h94 must be read, and I recorded it
before any result: the DT's raw proposal is never admissible — 11 of 11 snapped,
by more than the student–teacher gap. So h94 tests "the nearest in-ROI pool
member to the DT's output". Not pool+argmax, but not "the DT decides and the ROI
constrains" either.

Then h101 (numbered h97 when written), on existing data. h96 had left the residual gap as "boundary aversion,
which an ROI structurally cannot fix", and I wanted to know which mechanism that
was before spending compute on a fix. I framed it as two options and both were
wrong. The head does NOT fail to find the boundary — it moves its centre toward
the correct bound in every sensitive dimension and sits at the middle in the
irrelevant ones. What it cannot do is span the last fifth of the domain, because
its output cloud is three to six times tighter than uniform. Reach follows
gap/sd almost exactly across the four dimensions.

That is the second time today my hypothesis space excluded the answer — the
first was diagnosing a published statistic as an sd-vs-MAD mix when it was a
pooled-sample mismatch. The rule I wrote then was about numerical coincidences;
it belongs to hypothesis design too.

It also cheapens the next experiment considerably. h96's framing implied an
architecture rewrite. The head is already 83-98% of the way, so the levers are
to sharpen centring (a sensitivity-weighted L_loc, one flag) or to widen the
output — and widening is no longer obviously wrong, because h95 and h96 showed
more dispersion arriving alongside better regret.

## 2026-08-27 ~22:15 — the dose, and a bar that passed on noise

h84 had run three ROI tightness settings on Borehole and their regret outcomes
were already known, so the mediation claim from h96/h101 could be tested against
a four-level dose instead of a single on/off contrast. Registered first, because
the interesting feature was that regret is NOT monotone in tightness — the
fixed-beta arm is looser than q=0.10 and scores better — so if centring mediates
it had to inherit that shape.

It appeared to. GAPSD ranked the four arms exactly as regret does, inversion
included, and the mechanical bar printed MET. Then I checked whether the two
arms at the top are actually separable, because they differ by 0.02. They are
not: paired mean −0.018 against a per-seed sd of 0.169, better on 2 of 5. The
inversion is noise, in the mechanism and probably in the regret too.

So T2 is recorded as MET, as registered, and as uninformative at the position
that motivated it. An ordering bar over four items always produces an ordering;
passing it means nothing unless the adjacent pairs are separable. That is the
same defect as the M1 bar on a floored measure, wearing different clothes, and
it is now a standing rule.

What survives is still worth having. Over the three levels that ARE resolvable
— no ROI, annealed, tight — centring improves monotonically and unanimously
(5/5 at both steps) and regret follows in the same order. Centring mediates the
dose where the dose can be measured at all.

It also quietly removes a claim: q=0.10 was never shown to beat q≈0.21. They are
tied on regret and on mechanism. Calibration's case is controllability across
benchmarks, which is what it always was, and not superiority on Borehole.

---

## H93 — the four-benchmark headline, measured twice

This entry backfills a gap: h90, h91, h92 and h97 are all logged above, and h93
is not, despite being the experiment that changed what the project's headline
says.

The setup was that h83 concluded MF-DRO beats no baseline on any of four
benchmarks, and only two of those four had ever been measured at a second seed
set. Hartmann's deficit had already evaporated at n=10 and Borehole's had held.
Currin and Ackley had never been re-measured at all, and their h83 margins were
0.01 and 0.40 — far tighter than the 1.37 that did not survive on Hartmann. That
made them worth checking rather than assuming, which is the whole reason the
experiment was registered.

Two design choices mattered. The comparator is each benchmark's OWN best baseline
— MI-Greedy for Currin, SF-DRO for Ackley — not MF-MES throughout, because
MF-MES wins neither and using it would have tested an easier question. And P2,
the Ackley direction, was registered with **no prediction at all**. That was not
a hedge: six mechanism predictions had been refuted by then, and Ackley's margin
sat well inside its own per-method spread. Declining to guess turned out to be
right — Ackley's deficit reversed sign.

The result: Ackley does not replicate (−0.07, better 3/5, against +0.40).
Currin replicates in sign (+0.01%, 1/5) and is meaningless in magnitude — 0.0155
in the function's own units against an optimum of 13.80, with four of five runs
finishing at exactly zero. Both methods have solved Currin.

I recorded REPLICATES for Currin because that is what the rule written in advance
says, and rewriting a scoring rule after seeing which side it lands on is the
failure this project's discipline exists to prevent. But the honest reading is
stated alongside it: a deficit that survives only because a solved benchmark
cannot score lower is not evidence about the method.

So the headline becomes **one substantive deficit of four, not four**, and the
boundary-optimum explanation stops being one benchmark's excuse and becomes the
whole account. Two things that does not license, both now on the record: h83 was
not wrong when written — its within-experiment comparisons were sound and its
prediction was met at n=5, and what failed is the durability of three margins
that only a second seed set could test. And Borehole's remaining deficit is
large, real, and untouched by anything that worked — the calibrated ROI closes
37% of it, refinement does not close it. Narrowing where the problem lives is
useful because it says where to aim, not because the problem shrank.

## 2026-08-27 ~22:55 — two mechanisms tested, two eliminated, and a prescription withdrawn

h96 left a real pattern with no mechanism: the ROI relocates the query cloud on
exactly the one benchmark where it helps. I tested the two obvious gates.

Headroom went first, and it failed in the direction I had registered as likely.
The ROI works on the benchmark with the second LEAST left to gain and fails on
the one with ten times more. Weighted headroom is nearly anti-correlated with
where the ROI helps. That test also killed a prescription of my own: Borehole's
dominant dimension carries 86% of the variance and has zero headroom, so the
sensitivity-weighted loss I had recommended would optimise hardest where nothing
can be gained. Withdrawn outright — it came from a cloud statistic and did not
survive the incumbent correction.

Containment went second and produced the more interesting failure. The ROI on
Borehole sits farther from the optimum than an unfiltered pool would, contains
nothing at all within 0.2 of it, and still delivers the only confirmed gain in
the project. My script printed "the hypothesis is DEAD" and I overrode it —
because the diagnostic is an unweighted full-dimensional distance, the exact
metric this project established reverses orderings on this benchmark. The honest
verdict is untested rather than refuted.

That override is post-hoc and I have labelled it so. My protocol normalized out
the dimensionality confound and missed the weighting one, which I have now been
caught by four separate times today. Noticing a defect only when the result is
unwelcome is how any failed prediction gets rescued, so the registered FAILED
stands alongside it.

One thing survives the whole mess independent of the weighting: the rationale
that originally justified this diagnostic — that a region containing nothing
near the optimum starves the model of good training examples, which is what got
the ROI deleted before it was reinstated — is false as stated. Borehole scores
exactly zero on it and works.

So two candidate gates are gone and the pattern stands unexplained. That is
where h94 matters: it tests the region as a constraint on the query rather than
on the demonstrations, which is a different mechanism entirely.

## 2026-08-28 ~00:10 — every effect the ROI has is on one benchmark

The brief that opened this investigation grounds its diagnosis in Hartmann
numbers, and every waste measurement I had made was on Borehole — which is also
the only benchmark where the ROI improves regret. So the two things had never
been seen apart, and I could not tell whether "the ROI reduces waste" and "the
ROI improves regret" were two descriptions of one phenomenon or two phenomena.

Hartmann separates them, because the regret result there failed and was
withdrawn. The answer turned out to be neither: on Hartmann the ROI reduces no
waste, improves query quality by −0.001, and moves dispersion *down* — the
direction that accompanies its failures elsewhere.

So the picture is now uniform. Regret, relocation, waste reduction, query
quality: all four appear on Borehole and none appear on Hartmann. Whatever the
region of interest is doing, it does it in one place.

The tick also produced my third badly-written bar of the session, and the three
rhyme in a way worth naming. h95's bar demanded a seed count *and* an effect
size; when I wrote today's I kept the count and dropped the size, so it passed at
a paired mean of −0.001 — and my own script drew a real conclusion from that
pass. I declined the conclusion, recorded the bar as met-as-registered and
meaningless, and wrote the general rule: a bar has to say how big an effect it
requires, not just how many seeds must show it, and what happens when a seed has
nowhere to move.

That is now three bars this session that turned on their own construction rather
than on the effect they were testing. It is a cheaper failure than a wrong
result, but only because someone reads the number underneath the verdict.

## 2026-08-28 ~02:00 — the answer takes shape, and one thing turns out to be unanswerable

The commissioned question now has an answer worth writing down, so I wrote it as
a seven-part synthesis a paper could lift: what the strategy is (quantile
calibration on the teacher's pool), what it delivers, whether it stops waste,
the mechanism, the scope, why the paper's own query-side formulation adds
nothing, and an explicit list of what is not established.

Two things then arrived that changed it within the hour. The peer's tightness
experiment showed the setting I had just recommended is not the right one —
halving the acceptance rate improves the result by more than its own spread, and
every region number in this project used the first calibrated value anyone tried.
So the synthesis was amended: the strategy is q=0.05, and the well-replicated
figure is no longer the best figure. Reassembling the dose curve with that new
point also corrected an earlier experiment of mine, which had treated the
fixed-beta arm as a tightness level when its acceptance drifts 250x within a run.
Among the arms that actually hold a set point, tightness is monotone.

The second thing is a limitation I had been circling without naming. There is
exactly one benchmark where the region does anything, and it is simultaneously
the most concentrated, the only one with a boundary optimum nearly everywhere,
and the only real deficit. Those three co-occur perfectly, so any property unique
to that benchmark "explains" the pattern. That is why the two mechanism gates I
tested today died so cleanly — passing a test that requires failing on three
benchmarks and holding on one is nearly free, and is not evidence. The honest
statement is that the mechanism is unanswerable from these four benchmarks, not
merely unanswered, and what would settle it is a synthetic family varying one
property at a time rather than a fifth benchmark varying all of them.

Also corrected a withdrawn claim that was still live on the published page,
stated as established, with a conclusion drawn from it. It had been withdrawn in
the findings file hours earlier and the withdrawal never propagated to the
outward-facing deliverable. That is the same defect the peer had fixed in the
other direction earlier tonight, with inline markers at the point of claim.

---

## H105, H97, H102 — a pass, a tuning result, and a refuted mechanism

Three results landed tonight and two are already back under test. Logging them
together because what connects them matters more than any one.

**H105 discharged the last qualification on the project's only passing
pre-registered claim.** A peer session established that PROTOCOL.md registers two
baselines on one benchmark at ten seeds, never amended, and that h83 ran five. I
verified the file myself rather than take it on report, then supplied the missing
five. Cost: about two minutes, because the registered baselines are the cheapest
methods in the comparison and MF-DRO and MF-MES already existed at those seeds.
The test passes at the full sample by a factor of six.

The gate nearly voided it and the gate was wrong, which is the part worth
keeping. My code-drift check failed all four methods because `mf_dro.py` changed
between the contributing commits — but that change lives entirely inside the
`use_roi=True` branch, and a control run never enters it. The criterion that
matters is whether the *executed* path changed; hashing it showed byte-identity
across every commit involved. That also retroactively validated three earlier
n=10 conclusions that had pooled across the same commits, which neither session
had checked. **A gate that fires on the wrong criterion is worse than no gate: it
would have discarded a valid experiment and taught us nothing.**

**H97 answered a question that had sat open all day.** Every ROI result this
project reports uses an acceptance rate of 10%, chosen because it was the first
calibrated value tried. Halving it beats it — −1.52 paired, 4/5 — and the margin
clears a bar I fixed before the runs at the level two existing settings 2.1x
apart had *failed* to clear. So the setting was never tuned and every ROI figure
here understates the effect.

**H102 refuted its own mechanism, and that is the most useful thing it did.** I
predicted an L1 loss would make the head reach domain boundaries more often,
calling it "close to definitional" from the median-versus-mean argument. It
reaches them *less*, on four of five seeds. The argument depends on more than
half the teacher's target mass sitting at a bound, and I never checked that
clause — targets are overwhelmingly interior, so the median is interior too and
L1's robustness suppresses the very outliers that occasionally pulled an L2
prediction outward.

The registered consequence is that P2 says nothing about boundary aversion in
either direction, and I applied it rather than reaching for the reading where a
−2.08 regret gain becomes evidence for the hypothesis it was chosen to test. The
gain is real, clears the bar, and is **unexplained**.

**What connects them.** H97 and H102 are both single-seed-set, and this session
has watched −5.85 become −2.11, a 4/5 become 2/5, and two claims withdrawn
outright on exactly that re-test. Both are now running at the original seeds
(H107, H108) before anything is built on either. H108 carries one prediction
aimed at me rather than the method: it forecasts the boundary direction from
H102's *measurement* where H102 forecast it from an *assumption*. If that
succeeds where the assumption failed, Lesson 23 is working rather than being
recited; if it fails too, the lesson is insufficient and I will say so.

## 2026-08-28 ~05:30 — the gate held, and buying the measurement was the right call

The comparability question that ran through the night is settled. Both control
seeds came back bit-identical to the stored originals — same query counts, same
locations to the last digit, same fidelity at every step, final scores matching
to ten decimal places, and the region's own calibration statistics identical
too. The working-tree changes are inert on the branch that matters, measured
rather than argued.

The path there is worth keeping because I got it wrong first. I claimed the
question was already settled, citing an agreement between two halves of an
experiment — but only one half had been re-run, and the other was the stored
data being compared against itself. An identity presented as a check. The peer
said plainly that the design could not support the conclusion, and then spent
two runs rather than a third argument.

Both of the arguments we had were correct. That is exactly why spending the runs
was right rather than redundant: neither of us could know they were correct
beforehand, and being wrong meant re-running four experiments and retracting
every conclusion drawn from them. The rule I would carry: when an argument's
failure mode would invalidate work already done, buy the measurement.

Two smaller things from the same stretch. I built a trace-comparison utility for
the verdict and gave it one property that matters — it refuses when both paths
resolve to the same file, because that is precisely the degenerate case that
produced my original error, and a tool that silently returns "identical" there
would reproduce it on demand. And while waiting I nearly filed a contamination
alarm after comparing a partially-completed run against a finished one; index-
matched at the same query, they agreed to the digit. That is the fourth instance
today of the same shape — compare like against like, at the same point in
whatever process generated the two sides.

With the gate passed, the region result stands: on the one benchmark where this
method genuinely loses, the calibrated region removes 57% of the deficit at the
well-replicated setting, on every one of ten seeds. The better setting closes
62% on the seeds where every arm exists. The run testing whether that holds at
the full seed-matched sample is finishing now.

---

## H109, H111, H115 — three experiments that mostly said "no", and were worth it

None of these produced a positive result. Logging them together because the case
for running them is clearer in aggregate than singly, and because two of the
three changed a conclusion that had already been published.

**H109 bought a fact that two arguments could not.** Two independent lines said
the working-tree patches were inert on the ROI path — my sandbox smoke test and a
peer's reading of the diff. Both were right. That is exactly why spending two runs
was correct rather than redundant: we could not have known they were right
beforehand, and three experiments' results rested on it. The runs came back
bit-identical to h84's stored traces, 115 and 103 post-init queries, |dregret| = 0.

**H111 closed the investigation's most-repeated limitation.** "The ROI only works
on Borehole" had been said by both sessions for hours and rested entirely on
q=0.10 arms. Two settings spanning a 2x range now fail on both benchmarks where a
difference could be resolved. The limitation survives a real test instead of an
assumption — and the hoped-for outcome, that a better setting would make the
mechanism identifiable from more than one benchmark, did not happen.

Its Hartmann arm also produced something more useful than its verdict: sd 6.32
across five seeds, spanning −9.52 to +8.02. **Hartmann cannot resolve an ROI
effect of any plausible size at n=5.** Both its q=0.10 and q=0.05 numbers should
therefore be dropped from arguments in either direction — including from my own
protocol's reasoning, which had treated the q=0.10 figure as evidence of a weak
effect rather than of an unresolvable one.

**H115 was a comparator fill that refuted the arithmetic that motivated it.**
Borehole MF-MES had never been run at seeds 47-51 — half of h113's design. Five
runs later: MF-MES scores 5.59 there, not the 8.24 available from a different
pairing. Even perfect additivity now leaves the combined arm 3.71 points behind.
The reading that h113 might tie the strongest comparator was an artefact of
quoting a comparator from seeds h113 does not use.

That is the seed-matching error's third appearance in one night, and **the first
that neither session could have caught by reasoning.** The missing cell had to be
run. It is the strongest argument I have for filling comparators rather than
quoting the nearest available number, and it cost five cheap runs.

**What connects them.** Each replaced an assumption with a measurement, and in
two cases the measurement contradicted what careful reasoning had concluded. The
pattern across tonight is that our arguments have been right more often than not
— and the two times they were not, only running something revealed it.

H113 is in flight: whether the two surviving interventions compose. Its reading is
already bounded in advance by H115 — it cannot show the method reaching MF-MES on
Borehole — and by the weighted-dispersion work, which rules out dispersion as the
shared channel if it comes out shared. Both bounds were established before the
numbers, which is the only time such bounds are worth anything.

## 2026-08-28 — h116 (zero compute) and h117 (launched)

Chased down whether the surrogate is ARD. It is (`ko_gp.py:312`); `exactGP.py`
is isotropic but is not on the KO path, and I nearly filed that as a finding
from the wrong file. What IS true is that `mf_dro.py:251` averages the ARD
vector into a single state feature, and that is the only place lengthscales
enter the pipeline — the policy never sees per-dimension relevance.

Pre-registered h116 on that basis (relevance-blindness should show up as an
uncorrelated dispersion profile) and it **failed its gate**: effect 0.17 (SD)
and 0.10 (MAD), with the two declared variants disagreeing in sign.

Also caught a real error in my own measure mid-experiment: stored `x` has no
common coordinate scale across benchmarks, so the first run was reading
Borehole's domain box rather than the policy. Amended (disclosed as
post-hoc), recomputed, kept the originals as superseded.

The exploratory follow-up is the substantive result: the two methods' TOTAL
HF-query dispersion on Borehole is equal (1.06x), but weighted by variance
share MF-DRO's is 3.96x larger, 5/5 seeds, effect 4.84. The problem is
mis-allocation across dimensions, not more scatter. That reworded the founding
diagnosis and refuted h116's own "blind" framing.

Launched h117 to replicate it on fresh seeds 52-56, with a blocking bit-identity
gate because the working tree carries two uncommitted patches that are inert
only by inspection. 15 workers total (10 peer h113 + 5 mine), at cap, not over.

## 2026-08-28 (later) — h118, and a line of work closing

Pursued the primary question directly: does the ROI reduce the wasted HF budget
h116 found? There was a clean mechanism to test — the teacher's 600 candidates
are uniform draws filtered by the ROI, so tightening it packs the same points
into a smaller volume. Resolution amplifier.

**It failed its gate: effect 0.62 against a bar of 1.0 (0.76 at the declared
second cut-off), 4/5 seeds.** Second pre-registered gate miss in two experiments.

A third arm in the same experiment showed why. REFINE-100 clamps 100 local
Gaussians to the box, and clamping deposits mass exactly on the boundary; it
reaches the boundary on 54.2% of HF queries against the control's 14.7%, with
non-overlapping per-seed ranges, and cuts waste at effect 1.90, 5/5. *A filter
cannot create probability mass the proposal distribution never had* — which also
covers MF-MES's box-constrained L-BFGS-B.

Then the result that closes the line: ROI-Q10 and REFINE-100 reach the SAME
final value (271.64 vs 271.03) while differing 2.8x in wasted budget. The
inefficiency is real, reproducibly fixable, and buys nothing. I recorded that
against my own three commits from this morning, and corrected the published
report, whose lead section had been built on the opposite assumption.

Net for the primary question: two mechanistic candidates for the ROI's Borehole
benefit — dispersion and boundary resolution — are now ruled out by tests
written before the numbers were seen. The channel is still unidentified. h117
continues; it can confirm the inefficiency is reproducible but can no longer
make it the explanation.

## 2026-08-28 (later still) — gate passes, two peer corrections adopted

**h117 GATE G0 PASSED: 83 queries, 0 differing.** The working tree, carrying the
uncommitted h94/h102 patches, reproduces h83's stored Ackley MF-DRO seed42 trace
bit-for-bit. The patches are inert by EXECUTION — the thing the gate was added
for after inspection alone missed h94's NameError. h117's ten runs are cleared.

Two corrections from the peer session, both accepted:

1. **One tree.** I wrote a provenance condition ("must be empty-diff against
   af5ec31b1") referring to "my working tree". We share one repo; the condition
   could only be met by reverting patches mid-flight and breaking their h113.
   Withdrawn. I did verify their supporting evidence rather than accept it —
   h109's stored `code.dirty` is True with exactly today's two files — and
   flagged the gap it leaves (same filenames is not same bytes), which G0 then
   closed on current content.
2. **The count.** "Five of seven separated" is now "three of five independent
   quantities". C1-C3 are one fact under a fixed cost budget; my prose said so
   and my headline didn't. 3/5 is materially weaker evidence than 5/7 reads as.

Launched the three ROI-OFF control runs (seeds 44, 45 up; 46 queued) behind a
launcher that polls the global worker count before each spawn rather than
assuming an allocation, since the 15-cap is shared.

Peer also disclosed unprompted that their h113 gate FAILED at seed 43
(L_loc 0.0969 vs a registered >0.10) and that they amended it. The timing
defence holds — written before any regret number was read — and their
first-five-iteration L_loc (L1 0.187/0.219 vs MSE 0.058/0.052) does establish
L1 fired. I'd still class the amended gate as exploratory-grade, since the
statistic was chosen after seeing the failure.

## 2026-08-28 — h121: the framing has a mismatch in it

Checked the founding diagnosis on its own terms. Its 20.8% reproduces EXACTLY —
the mean of [0.0, 0.0, 75.0, 16.7, 12.5]% across Hartmann seeds 42-46. But the
per-seed HF counts are 8, 24, 12, 6, 8: two seeds waste nothing, the 75% seed is
9 of 12 queries, the median is 12.5%, and dropping seed 44 gives 7.3%.

Then the part that matters. Waste by benchmark (MF-DRO median): Hartmann 12.5%,
Borehole 3.2%, Ackley 2.5%, Currin 0.0%. Every ROI benefit in this project is
Borehole-only, and h111 showed it fails on Hartmann and Ackley at two tightness
settings spanning 2x. **The ROI works where the waste is smallest and fails
where it is largest.** It is not addressing the diagnosis it was introduced to
address.

P3 missed its gate (3/5, needed 4/5 — two seeds tie at exactly 0.0% and a tie is
not an exceedance). Recorded rather than rounded up.

Also recorded: the diagnosis's "0.336 vs 0.747" score does NOT reproduce under my
normalisation (11.844 vs 15.276 on the same runs). The ordering reproduces; the
normalisation behind the original pair is not recoverable from what is written
down. Flagged as a definition mismatch, not a claimed error.

Launched all three h120 ROI-OFF control runs (seeds 44, 45, 46) behind the
slot-polling launcher. h117's four Borehole arms are 42 min into ~83.

## 2026-08-28 — the annealing bug, and testing the ROI where the waste is

Went looking for whether the paper's `beta_t` subscript had ever been tested.
h84 has a `ROI-ANN` arm configured `roi_accept_start=0.50, roi_accept_end=0.05`,
so it looked answered. It is not.

**The anneal never ran.** `_prog = n_real_iter / T_real` with `T_real =
bo_iterations = 4000`, against runs that terminate on a cost budget of 200 and
reach ~104 HF observations on Borehole, ~18 on Hartmann. Realized q moved 1.1
points on Borehole and 0.13 on Hartmann, out of a configured 45. ROI-ANN is a
constant q~0.49 arm — the loosest ROI in the project.

findings.md already recorded it as "ROI-ANN (q~0.49)". The realized value had
been measured and written down without anyone asking why a 0.50->0.05 schedule
was sitting at 0.49. The defect and its own evidence have been adjacent in the
file for some time.

Every "annealed ROI" statement is void, and the paper's beta_t is still
untested. The direction also matters and cuts against the project's prior:
acceptance is monotone INCREASING in beta, GP-UCB's beta_t GROWS with t, so a
faithful schedule WIDENS the ROI over the run — the reverse of ROI-ANN's intent
and of the whole q=0.10/q=0.05 tightening programme.

Did NOT edit `mf_dro.py`: it is the one shared tree and 10 runs are live. Asked
the peer to hold off too, and put two design questions to them before I register
the schedule experiment (cost-based progress variable; widening vs tightening).

Launched **h122**: completes h84's Hartmann ROI-OFF control (seeds 44-46 — that
arm holds only 42-43, the same shortfall as its Borehole side) so the primary
question can be tested on the benchmark h121 showed the waste lives on. The
locked prediction is a NULL, with grounds, and pre-commits that any arm clearing
the bar gets reported as prominently and needs separate confirmation.

Also caught a problem I had created: h120's and h122's control runs write to
their own directories while the arms they compare against live in h84 — the
banned cross-experiment pairing. Declared an explicit exception before any of
those runs produced a file: they are h84 arm completions (byte-identical worker,
config, budget, spec; tree passed G0), will be merged into h84's results with a
per-run provenance manifest, and the merge is withdrawn if any run's commit
fails an empty-diff check at merge time.

## 2026-08-28 — h120 confirmed, and I was wrong about the control

The peer's h113 analysis showed h83's plain `MF-DRO` is the no-ROI control under
a different name. I had told them no independent 5-seed set existed. I measured
the substitution rather than accepting it: h83 MF-DRO vs h84 ROI-OFF is
bit-identical at both overlapping seeds, 137 and 132 queries, across three
commits. So h120 became evaluable at its locked seeds.

CONFIRMED: the ROI buys ~9 fewer HF queries and ~18 more LF (effects 1.55/1.58,
5/5), and each HF query it buys is better by +17.05 count-matched (effect 3.54,
5/5) — larger than the unmatched +15.16, so not a convergence artefact.
NOT CONFIRMED: time-to-incumbent, 0.29 at 3/5 against the screen's 1.35 at 5/5.
Limb dropped. The screen produced three quantities; confirmation kept two.

Registered h123: the paper's beta_t as a WIDENING ROI, with both of the peer's
points built in — cost-consumed as the progress variable (the bug was a
denominator the termination condition doesn't use), and the widening direction
recorded explicitly as inverting every ROI experiment either session has run.
Locked prediction is a null, since tightness has been a null axis wherever
measured properly. Not launched; still holding off src/ with 10 runs live.

## 2026-08-28 — h125 refutes a prior both sessions held

Found that `roi_summary.accept_frac` is stored in every final result file (not
the ckpts, which is why it had gone unread). Measured the realized tightness of
every ROI arm instead of trusting arm names — the discipline the ROI-ANN naming
bug demands.

Quantile mode hits 0.100 exactly on every seed of both benchmarks. Fixed beta=2.0
realizes 0.036-0.265, a 7x swing. Corrects findings' recorded "ROI-FIX2 realises
24.9%", which was one seed. And ROI-ANN measures 0.493/0.498 against the
0.494/0.498 I derived this morning — the annealing bug confirmed by measurement.

Then registered h125 predicting a NULL and was refuted. Q10 (q=0.100) vs ANN
(q~0.495) is a 5x contrast: Borehole regret +9.018, effect 5.69, 5/5.

The prior failed because every study behind it was 2x or narrower. Together they
give a shape: flat below 0.10, steep degradation by 0.5. That is the first
affirmative answer to the primary question this project has produced.

Amended h123 before launch: its grounds are gone and it tests the harmful
direction. Original prediction left visible.

## 2026-08-28 — h126 launched, after a launch incident I caused

Registered h126 to probe BELOW the plateau h125 revealed: q=0.02 vs q=0.10, a 5x
contrast — deliberately the range h125 proved has power, rather than the 2x
contrasts that produced the false "tightness is a null axis" prior. Checked
feasibility before locking: roi_raw_pool=2000 and _N_POOL=600 mean q=0.02 needs
~15 draws against MAX_DRAWS=40, so no pool starvation and no silent top-up from
unfiltered draws.

INCIDENT, reported. My first launch went out against a worker whose ROI-Q02 arm
did not exist: the patch asserted a 3-space anchor against a 1-space file and
refused, correctly — and I started the launcher in the same command block
without checking that it had. Four jobs spawned for an unresolvable arm, died on
the ARMS lookup, wrote nothing, killed within a minute. Cost: worker-seconds.

Second instance of the h94 pattern. h94 launched 8/8 runs that died on a
NameError because the ON path had never been executed, and gate G3 was added
afterwards specifically to stop that. I did not apply my own gate.

The lesson is narrower than "run the gate": **a patch that refuses is only a
safety net if something checks whether it refused.** The assertion worked. I
ignored its result by putting the launch beside it.

Relaunched after importing the worker module and asserting ARMS['ROI-Q02'] is
exactly {'use_roi':True,'roi_beta_mode':'quantile','roi_target_accept':0.02},
with ROI-Q10 and ROI-OFF unchanged and BUDGET/SPEC intact. All five runs alive
and checkpointing.

## 2026-08-28 — h129, the mechanism narrows and four of six predictions fail

Opened by confirming the read point the peer had queried: h83's `sr_curve`
computes `cost_cum - init_cost`, verified empirically (axis ends 200.60 Borehole
/ 201.20 Hartmann against raw `cost_cum` 240.60 / 294.20). Every number I have
reported is at full post-init budget. Adopted the peer's sharper rule — name the
READ POINT, not just the statistic — since unit mismatches leave effect sizes
invariant while read-point choice moves effects 3x to 30x.

Registered h129 before h127 had a single result file, then ran it down:

  P3  h128 q=0.493 HF frac 0.839 +- 0.012   PASSES but uninformative
  P4  quality flat on Borehole              FALSIFIED — quality rises, effect 2.66
  P5  quality improves on Hartmann too      FAILS — effect 0.02, no dissociation
  P6  Ackley shifts down                    FAILS and REFUTES P5's reading
  P1/P2  h127 q=0.30                        PENDING

Two conclusions of mine were overturned by my own tests. **"The interventions
that work act elsewhere" is wrong for Borehole query quality** — the ROI moves
the channel the founding diagnosis prescribed, effect 2.66 at 5/5, and the peer's
count-matched h120 P3 agrees from an independent statistic. And **the fidelity
mechanism is Borehole-specific** (effects 1.65 / 0.78 / 0.49 across three
benchmarks), so it belongs beside the benefit it was meant to explain rather than
above it.

Two errors of my own, both worth keeping:

- A units error INSIDE a locked protocol: P4 invoked the 0.59 separability bar,
  which is in regret points, against a normalised score at 0.381. A criterion
  that could not fire. The peer had confessed the identical class an hour before.
- A two-point unification: I proposed the ROI "regularises the fidelity mix
  toward the middle" from two benchmarks whose controls sat on opposite sides of
  0.5. **No arrangement of two such points could have failed to suggest it.** The
  third benchmark was in the repository the whole time and inverts the ordering.

The founding diagnosis reproduces exactly on its own benchmark (0.336, 0.208),
and Hartmann affords 11.6 HF queries per run against Borehole's 94.0 — so its
headline statistics are per-run means over about twelve numbers.

Report merged onto the peer's concurrent republish and updated.

## 2026-08-28 (late morning) — generality closed, negatively

Two results settled the day's direction.

**h125** refuted my own locked null: ROI tightness IS a lever. Across a 5x
acceptance range (q=0.100 vs q=0.493, both realized to three decimals) Borehole
regret moves +9.018, effect 5.69, 5/5 — the largest effect in the project. The
prior "tightness is a null axis" failed because every study behind it was a 2x
contrast or narrower. Together they give a shape: flat below 0.10, steep
degradation by 0.5.

**h128** then failed both its predictions because I mixed units inside my own
locked protocol — adding a rel%-of-optimum benefit to a raw-regret cost. A loose
ROI is not harmful; it forfeits ~69% of the benefit and drops below
separability. Fifth unit mismatch of the day and the first I committed rather
than caught.

**h130** closed generality. Fidelity had already failed to generalise (peer's
h129 P6, verified, plus a fourth benchmark I added). Quality was the survivor —
two independent statistics on Borehole, and the channel the founding diagnosis
named. It fails too: Hartmann 0.33, Currin 0.30, Ackley 0.78.

Four mechanisms, four benchmarks, one positive cell each time, always Borehole.
And the sharpest form of h121's mismatch: **the ROI fails to move the diagnosed
quantity on the benchmark where it was diagnosed.**

**h120's registered invalidation condition discharged.** Its three ROI-OFF runs
reproduced h83's MF-DRO bit-identically at seeds 44-46 — across three different
commits, all with the tree dirty. MF-DRO == ROI-OFF now verified at 5/5 Borehole
seeds, and h117's gate G0 extended by 414 queries as a side effect nobody
designed for.

Published the synthesis. Still in flight: h117 (last run), h126 (q=0.02),
h127 (peer's q=0.30). None can change the generality finding — all Borehole.

## 2026-08-28 (cont.) — h131: the ROI is front-loaded and then stalls

Registered h131 before computing, to ask whether Hartmann is different in KIND or
merely TRUNCATED — since four mechanisms x four benchmarks now have exactly one
positive cell each, always Borehole, which is descriptive and explains nothing.

**P1 INDETERMINATE.** It landed at 0.86, inside a gap I left between my 0.5
threshold and my 1.0 falsifier. A protocol-design flaw, third of its class here.

**The motivating story is refuted anyway**, and the gate would not have told me:
Borehole's benefit does not accumulate. −4.68% of optimum at 12 HF queries,
−4.22% at 94. It peaks at −7.81% around 48 and declines. What improves
monotonically is precision, sd 5.417 → 2.433 — so a rising effect size across a
run is variance reduction, not signal growth.

**EXPLORATORY follow-up, and the most useful thing this round:** the advantage
erodes +3.588% of optimum between cost 100 and 200, effect 1.40, 5/5 seeds. The
mechanism is a stall — over that span the control's regret falls 5.193% while the
ROI's falls 1.605%. Front-loaded gain, then a plateau the control eats into.

**This reframes h125 and the paper's `beta_t`.** h125 varied a CONSTANT tightness
and establishes that uniformly wider is worse; it says nothing about a schedule.
Tight-then-wide is compatible with both. The ROI's value sits where a widening
schedule stays tight, and its stall sits where such a schedule relaxes. Sent to
the peer with a falsifiable prediction BEFORE their h123 launches, so the grounds
are visible in advance rather than assembled afterwards.

**METHODOLOGICAL:** the 0.59-regret-point separability bar is scale-dependent —
0.191% of optimum on Borehole, 17.758% on Hartmann, ~93x stricter. It was set in
a Borehole context and h111 applied it to Hartmann and Ackley. Checked rather than
assumed: it flipped no verdict, because every cross-benchmark null also fails the
unit-free criteria. One near-miss, my own. Rule extended — a bar is portable only
to the statistic AND the benchmark scale it was measured on.

## 2026-08-28 (cont.) — h133: the stall is a step, and pre-registration caught a second false positive

Registered h133 before computing: if the late stall is over-restriction it must
scale with tightness, and if it does not then h123's and h132's premises weaken
before their runs are spent. Four Borehole arms with measured constant realized q
already existed at seeds 42-46.

**P1 failed on the statistic I pre-committed to.** Raw late-gain is monotone in q
— 1.60, 3.65, 4.37, 5.19, a textbook dose-response. It is an artefact: arms ahead
at cost 100 have less regret left to remove, and the ROI arms are ahead.
Normalised by regret remaining at the midpoint it reads 10.51, 24.34, 22.06,
24.45 and is not monotone. The confound was written into the protocol before the
numbers existed, together with the rule that the normalised statistic governs.

**The finding is a STEP.** The extreme pair separates (Q10 vs OFF, +13.94, effect
1.08, 5/5); all three adjacent pairs are ties (0.81, 0.35, 0.19). q=0.10 recovers
10.5% of the regret still available at the midpoint; everything from q=0.21 up
recovers 22-24% and is mutually indistinguishable. So the stall belongs to the
tightest setting, not to tightness as a graded quantity.

**Two false positives today, both caught only by pre-registration**: this one and
the dispersion instrument. Both would have produced clean monotone orderings that
looked like results rather than artefacts. The peer's observation on the second is
the sharper statement of the danger — the raw dispersion column was perfectly
monotone across the three ROI arms, and only the arm one would be least likely to
include broke it.

Predictions handed to the peer before h123 launches: the ramp's shape should
matter far less than whether it escapes q~0.10, and there should be a ceiling
around q~0.21 above which no schedule endpoint buys anything.

## 2026-08-28 (cont.) — h134: why Hartmann is inert, and the bar failures move into tools

**The finding.** The primary question's causal path is that the ROI shapes a
*training distribution*, so it can only matter if training responds. Nobody had
checked. `L_loc` first-third vs last-third, seeds 42-46: Borehole control declines
+0.078; **Hartmann control declines −1.003 — the loss nearly doubles across a run,
4/5 seeds.** Between-benchmark contrast effect 1.10, 4/5, independently reproduced
by the peer.

Since `L_loc` is a loss on a moving target, this cannot separate "the head can't
learn" from "the target moves faster than the head follows". The disjunction
suffices: **on Hartmann the head does not track its training target, so an
intervention that works by reshaping that target has little purchase.** That is
an account of ROI inertness needing **no claim about where the region sits** —
the question h100's unweighted-distance instrument left undecidable. Hartmann is
not iteration-starved (120 vs 107) but HF-query-starved (11.2 vs 93.4).

**P1 failed on a threshold I mis-set** — 0.40 calibrated from one seed's
first-iteration vs last-iteration values, then registered against first-third vs
last-third means, which differ by an order of magnitude.

**A lead correctly died.** I noticed the ROI arm degrading less than control on
Hartmann and flagged it as a lead, not a result. My P4 replicated it across four
arms on the same five seeds and returned an equivocal 3-of-4 with pooled effect
0.47. The peer's test varied the **benchmark** instead and killed it outright: the
Hartmann mean is carried by seed 42 alone (+0.424 → +0.082 without it), and on
Borehole the sign reverses on a more separable contrast. **I replicated along the
axis with the most data rather than the axis the claim was about.**

**Tooling.** Three bar-design failures today and two calibration failures, none
caught in advance. `tools/check_gate.py` now verifies pass/falsifier **partition**
the outcome space (self-tested against both real gaps), and `--calibrated-by`
implements the peer's rule that a bar carries the command computing its
calibrating value, so misapplying it becomes a diff rather than a judgement call.

## 2026-08-28 (late morning) — h117 completes; logging patch applied and gated

h117 finished 10/10. **All six locked predictions pass** at seeds 52-56: the
boundary waste replicates (10.0% of HF queries off-boundary against MF-MES's
0.2%, effect 2.40, 5/5), with the effect smaller than at the exploratory seeds
(wR/uR 3.67 -> 2.50) as expected. h118 had already shown the waste does not
predict regret, so this confirms a real inefficiency that is not the reason
MF-DRO loses.

Patched `mf_dro.py` with a single additive field tagging each `roi_stats` record
with its real-loop iteration. Since the records already carry `beta_sqrt`, one
tag makes both acceptance and beta recoverable per iteration — closing M1's shape
blind spot and the peer's FIX2-as-implicit-schedule question together.

**Did not reuse GATE G0 for it.** G0 and h120's 414-query extension both ran
use_roi=False arms, and the patched line is inside `if roi_stats is not None:` —
neither could reach it. Registered h136 instead: Ackley ROI-Q10 seed42, which
enters the branch 2580 times, against h86's stored trace.

Also found the served-copy wrapper back in the shared report file and stripped
it; content was byte-identical to the peer's live version, which had already
absorbed my pooled-estimate correction.

## 2026-08-28 (cont.) — h137/h138: the competitive picture, and a budget-changing read-point hazard

**h137 — the best configuration against the strongest baseline, n=10, never asked
before.** Borehole, seeds 42-51, rel% @cost_curve 200: control 15.780, ROI+L1
9.822, MF-MES 5.996. `ROI+L1 - MF-MES` = +3.826, sd 5.438, effect 0.70, better
2/10. **The interventions move MF-DRO from 1/10 to 2/10 against MF-MES and close
roughly 6 of a ~10-point gap. Real progress, not a competitive win.**

My registered gate returned **TIED** for that, and I did not report it as parity:
the mean and 8/10 seeds are unambiguous, only the variance-scaled statistic is
not. **New gate lesson: partitioning is necessary but not sufficient — a residual
category can be named misleadingly.** It should have read NOT SEPARABLE. The peer
found the same defect in h126 ("CONFIRMED: no separable difference" for a
predicted null) and renamed it with 0/5 on disk.

**HAZARD, found by the peer, verified by me.** `cost_curve` is POST-INIT in
`mf_dro.py` and CUMULATIVE in `mf_mes_takeno.py`. Reading both at the stored
`cost_curve == 200` gives MF-DRO 200 post-init against MF-MES 160 post-init —
25% more budget for our own method in a competitive comparison, invisible at the
read site. **h137 is unaffected because `sr_curve` rebuilds the axis from the
trace**; verified, MF-MES's sr_curve axis ends 200.00 while its stored field ends
240.00. This ranks above the day's other read-point errors: the rest were units
and effect sizes survived them; this one changes budget and flatters us.
Encoded in `tools/check_axis.py`.

**h138 — the diagnosis's own metric, on the benchmark where the fix works.**
Borehole n=10: control 0.4049, ROI+L1 0.5771, MF-MES 0.7179. P1 PASSES (+0.1722,
effect 3.49, 10/10 — largest effect in the project). P2 STILL BELOW (-0.1408,
effect 1.17, 1/10). **The registered retraction did not trigger: the diagnosed gap
exists on Borehole too** (0.3130, effect 2.15) against Hartmann's 0.411 — so the
Borehole work has been attacking a deficit that is genuinely present there, which
had never been established. "Worse than initial design" runs 6.79% -> 2.16% ->
0.24%.

## 2026-08-28 (cont.) — h139: the best arm may already BE the paper's beta_t

Noticed I held three beliefs about ROI-FIX2 that cannot all be true: the analytic
argument (a fixed beta contracts the set as sigma shrinks, so it should loosen
early and tighten late), h133's measurement that FIX2 does NOT stall late, and the
wall-clock anomaly that FIX2 costs more than a tighter arm. Registered h139 before
any data existed, naming what dies under each outcome.

**P2, an indirect test on existing runs, PASSES with a clean crossover.** FIX2's
early phase tracks the tight arm (1.019 away from Q10, 4.576 from ANN) and its
late phase tracks the loose one (0.727 from ANN, 2.043 from Q10). **Tight early,
loose late — a widening trajectory**, which is the opposite of my own locked P1.

If the logged array confirms it, **ROI-FIX2 — the best-performing arm on record
(-4.814 rel%, effect 4.67, 5/5) — is a de facto widening schedule: the direction
the paper's `beta_t` subscript specifies, implemented by accident in an arm
labelled "fixed".** One mechanism reconciles all three beliefs: tight early
explains both the runtime and the front-loaded gain; loose late explains the
absent stall.

**Registered as a hypothesis, not a finding.** P2 is a proxy inferring acceptance
from behaviour, and that inference holds only if acceptance drives phase
behaviour — which h133 put in doubt. n=5, means not paired contrasts, no effect
size claimed. The caveat was written before looking so a pass could not be quoted
as settling P1.

It also makes the peer's h123 comparator refusal right for a second reason:
comparing an explicit schedule against FIX2 would have been schedule-vs-schedule
with one of them mislabelled a constant.

## 2026-08-28 09:25 — STOPPED BY USER

The user directed: stop chasing the ROI, stop the autoresearch. Actions taken:

- Killed h126 (q=0.02, mid-run at ~58%) and the h136 gate launcher.
- Cancelled the recurring autoresearch cron (job 6646b35a).
- Did NOT launch h123 (the paper's beta_t schedule), h136's gate, or anything
  further.
- Notified the peer session, which still holds h127 (10 workers); killing another
  session's runs was left to them.

**State of the tree, for whoever picks this up.** `src/policy/mf_dro.py` carries
an UNVALIDATED logging patch: one added field, `'n_real_iter'`, tagging each
`roi_stats` record so acceptance and beta become recoverable per iteration. Its
gate (h136) never ran. It is inert by construction on the `use_roi=False` path
but has NOT been proven inert on the ROI path. **Gate it or revert it before
launching anything on this tree.** The h94/h102 patches it sits alongside ARE
validated (h117 GATE G0, plus 414 queries across three commits from h120).

**Unfinished experiments, all registered with locked predictions:**
- h123 — the paper's `beta_t` as a widening schedule. No direction registered,
  sharp M1 gate. Never run. The one lever the paper specifies that nobody pulled.
- h126 — q=0.02, below the plateau. Killed at ~58%; no results.
- h136 — gate for the logging patch. Never ran.
- Peer's h139 — whether ROI-FIX2 is a de facto widening schedule. Rests on a
  proxy; the decisive test is a single logged FIX2 run.

The consolidated position for the primary question is in research-state.yaml
under CONSOLIDATED POSITION.

### CORRECTION to the handover note above (same session, before stopping)

Two errors in what I wrote, both caught after the fact — one by the peer, one by
my own check prompted by theirs. A handover is the worst place for an unchecked
assertion, so both are corrected here.

**1. "One added field" understates the tree.** `git diff src/` is **129
insertions** across two files, not one line. Breakdown by origin:

| change | origin | gated? |
|---|---|---|
| `'n_real_iter'` tag on `roi_stats` records | mine, today | **NO — h136 never ran** |
| `loc_loss` selector | h102 | yes (h105, h109) |
| `actions_x` variance tracking | h117 | yes |
| `_roi_snap()` + hook, ~60 lines incl. docstring | h94 | yes |

Only my one line is ungated, so the warning was right about what matters — but
anyone running `git diff` sees 129 lines with no way to tell which are covered.

**2. I told the peer the h94 hook is "dormant unless `roi_inference_mode` is set,
which nothing does". That is FALSE.** `experiments/h94-roi-at-inference/code/worker.py`
sets it at lines 43 and 49 — `roi_inference_mode='project'` and `'snap_control'`
— for its two arms, whose results exist at seeds 47-51.

The hook is dormant for every OTHER experiment's worker, which is what makes the
gated patches safe for the arms we have been running. But "nothing sets it" was
an assertion I made without grepping, in a note written for someone else to rely
on. The accurate statement: **the h94 hook is exercised only by h94's own
ROI-PROJECT and SNAP-CONTROL arms and is inert for every other worker in the
repository.**

**Patch NOT reverted.** Nothing will launch, so the gate condition does not
trigger, and the peer judged silently reverting another session's uncommitted
work more invasive than recording it. I agree. It stays, documented, ungated.

## 2026-08-28 — CLOSED. h129 P1/P2 both pass at a held-out dose; work stopped by user instruction.

**Stopped.** A peer relayed that the user had halted this line of work. I put it
to the user directly rather than acting on a second-hand instruction in either
direction; their answer was to stop and let h127 finish. Nothing further launched,
five freed slots left unused, no runs killed.

**The last result is the cleanest.** h129 was registered when h127 had zero result
files and its analysis committed before any landed. At the held-out dose q=0.30,
Borehole, seeds 42-46, paired:

    HF fraction   predicted 0.808 +- 0.020   observed 0.7916   INSIDE  (effect 2.58)
    benefit       predicted 2.21%, bracket (1.31,4.22)   observed 2.858   INSIDE (effect 2.02, 5/5)

The band was honestly sized this time — paired sem 0.0158 against a half-width of
0.020, where P3's had been 1.8x too tight. Both falsifiers excluded at 5.8 and 3.3
sems. The point prediction sits ~1.0 sem from the measured value. Dose-response
now monotone at three doses on one read point: -4.22 / -2.86 / -1.31.

**And it does not establish the mechanism.** The model that made the call treats
the benefit as flowing through fidelity reallocation alone; P4 showed query
quality also improves (the largest effect in the project) and P6 showed the
fidelity effect is Borehole-specific. Predictively accurate, mechanistically
incomplete, both true.

**Left registered and unanswered:** h123 (the paper's `beta_t`), h139 P1 (is the
best arm already such a schedule — one run would settle it), h132 (step-off), h126
(q=0.02, killed at 58%, not data). Repo carries an ungated `n_real_iter` logging
tag among 129 uncommitted insertions; gate or revert before launching anything.

**The day's ledger on method.** Nine errors of statistic, scale, or unchecked
assertion between two sessions. **Eight were caught by the other session.** The
one exception was caught by a protocol that named in advance the claim its own
result could retract — the only mechanism here available to a lone worker.

## 2026-08-31 — the question answered: the teacher was already optimal in the rewarded currency

Front: "why does better trajectory quality not improve MF-DRO?" Three experiments,
each with a registered retraction, two of which fired.

**h145** — an oracle teacher (straight line to the true x*) degrades Borehole by
+28.126 rel%, effect 4.49, 0/5. Hartmann pointed the same way but is confounded
(corr(HF share, degradation) = -0.830) and is not quoted.

**h147** — tested the RCSL return-coverage account from our own literature notes
(Brandfonbrener et al. 2022). **P1 FALSIFIED, opposite direction**: the oracle
RAISES between-trajectory return variance (0.611 vs 0.445, effect 4.17). The
variance reading of coverage does not explain it; the literature note now carries
a qualification that it covers the conditioning-side nulls only.

**h148** — P1 not evaluable (fifth registration against unserialised data). P2,
registered without direction, is decisive: **`rtg_target` collapses 0.9761 ->
0.3113, effect 32.11**, the largest in the project, with 55.2% of steps scoring
negative against the control's 25.8%.

**The synthesis.** The reward is information gain, `log(b_tau) - log(b_T)`, with
nothing measuring proximity to x*. An oracle path earns almost none of it. And the
control's teacher already argmaxes cost-normalised MES — *the same quantity the
reward measures*. **Teacher and reward optimise the same thing, so there was no
headroom.** Any improvement on a different axis costs reward, collapses the
conditioning target, and degrades the policy; the more perfect the improvement,
the larger the cost.

This also explains why the ROI's Borehole gain cannot be a teacher-quality effect:
the teacher already argmaxes information gain over the pool, so restricting the
pool can only lower the achievable max.

**Open and registered:** h146's DIVERSE-GOOD arm separates trajectory quality from
endpoint diversity. Prediction locked before the runs land — its `rtg_target`
should sit near 0.976, not 0.311. If it collapses too, the account is wrong and the
mechanism is about *forcing* rather than information gain.

**Instrument:** `tools/check_fields.py`, after five registrations against data the
pipeline does not serialise. The written rule was added after the second instance
and three more followed.

## 2026-08-31 — FRONT ANSWERED: the DT inherits its teacher's POLICY, not its trajectories' outcomes

Five pre-registered experiments, four with retractions that fired.

    h145  oracle teacher degrades          P1 falsified, opposite direction
    h147  RCSL return-coverage (variance)   P1 falsified, opposite direction
    h148  rtg_target collapse               P1 not evaluable; P2 decisive (effect 32.11)
    h146  quality vs endpoint diversity     P1 falsified; neither is the axis
    h149  is it my forced_x hook?           fork resolved: EXONERATED

**The result.** Three teachers spanning the whole quality range — perfect,
good-and-diverse, uniformly random — give **43.94 rel%, +28.13 vs control, effect
4.49, 0/5 improved**, identical to three decimals. The control gives 15.82 and
improves 5/5. Teacher quality does not order the outcomes; the identical figures
are the score for never improving on the initial design.

**The answer.** The DT is a policy distillation of MES. It inherits its teacher's
quality *as an adaptive rule*, not the outcome quality of the trajectories shown.
An oracle path is not a policy (it cannot be followed without knowing x*); a random
path is a worthless one; only MES is both followable and good.

**Process note worth keeping.** Two of the three replacement teachers used a hook I
wrote, and all failed *completely* — a binary outcome over nine runs, which is as
much the signature of a broken tool as of a real effect. h149 was run specifically
on pre-existing code before the conclusion was allowed to stand. Without it the
whole front would have been a measurement of my own bug.

**Corrections made along the way:** +28.13 read as graded degradation when it is a
saturation floor; the information-gain currency described as *ranking* teachers
when it only separates MES from everything else; Hartmann quoted before its
fidelity confound (corr -0.830) was found.

**Not claimed:** everything is Borehole. The mechanism is benchmark-independent in
principle, but that is an argument, not a measurement.

**Consistency check the account passes:** it condemns every intervention that
REPLACES the teacher, and the ROI does not — it restricts the pool the MES argmax
runs over. The project's only positive result belongs to the one class left
standing.

## 2026-08-31 (cont.) — h150 retracts the mechanism; the measured answer stands

Tested the account's sharpest claim: if the DT distils MES, its queries should
resemble MES's. **They do not.** A = nn(DRO->GP-UCB) - nn(DRO->MES) is positive on
only 2 of 5 seeds, mean -0.0388, and DRO->MES distance ~ MES's own internal spread.

**Retracted: "policy distillation of MES"**, from findings.md AND from the report
published to the user, per the find-every-surface rule.

**Survives untouched:** the measured answer. Three teachers, one floor; MES teacher
5/5, every substitute 0/15; rtg_target collapse. Better trajectory quality does not
improve MF-DRO.

**Left behind:** a puzzle I am recording rather than explaining. Teacher choice
decides *whether* the method works without deciding *what* it does. Three mechanism
accounts have now been tried and discarded on this front alone — RCSL return
coverage, information-gain-as-ranking, policy distillation — matching the project's
wider pattern that mechanism stories fit and fail to predict. The next step is not
a fourth account.

## 2026-09-02 — holistic reflection, prompted by the /loop instruction

**Was the last stretch deepening understanding, or just running arms?**

Honestly: **both, and the failure mode was real.** h145 → h146 → h149 → h151
were four teacher variants. Each falsified something specific and each named
its retraction, so none was wasted. But all four varied the same axis (what the
teacher does) and NONE of them varied the axis that turned out to matter (whether
the teacher can adapt). I declared the front ANSWERED at h149 on the strength of
three arms agreeing with each other — and three arms agreeing is exactly what a
shared confound looks like. It took h152, which the USER proposed, to run the
control that exposed it.

The lesson is not "run fewer arms". It is that **agreement between arms was
treated as convergent evidence when it was actually a signature of a common
confound.** Three teachers landing on the identical 43.94 should have been read
as suspicious rather than conclusive: identical outcomes from very different
interventions almost always mean the interventions were not as different as
believed.

Guard adopted going forward: before recording a front as ANSWERED, name the
axis every arm HELD FIXED, and run one arm that varies it. h153 is that arm for
this front.

**Also recorded as a discipline point.** h154 was run specifically to give a
cheap chance to KILL h153 before it consumed 4 more hours, and its gate included
"CONTRADICTED → h153's prior drops sharply, written down BEFORE its result
lands." That is the right shape for a cheap pre-test. Its outcome was MIXED: M1
confirmed with a complete 5-vs-15 separation, M2's registered direction refuted.
The M2 failure is kept as a failure; the post-hoc reading that makes it coherent
is labelled post-hoc and is not counted as support.

**Compute discipline:** 5 workers (h153) + 1 (h154b, finished) ≤ 15 throughout.

**Stale loop prompt:** the /loop text still names h146 and the POOL dose as the
current front. Both are closed — h146 completed, and the POOL dose was
deliberately NOT run (h146/h149 showed the outcome is flat in both quality and
diversity, so every dose point would return 43.94). research-state.yaml now
carries a `current_front` block at the top so a future tick reads the real state
rather than the prompt.

## 2026-09-02 (later) — reflection: the instrument needed auditing, not more arms

This tick ran no new MF-DRO arms. It audited the instrument built last tick, and
that was the right call: the audit found that h156's headline claim — "all four
targets reproduced, three within 8%" — was an over-read of a harness with a 6.1%
noise floor. I would not have found that by running another arm.

**How the noise floor surfaced is worth recording as a method.** h156d was
designed to change only C4/C5. Because its new fidelity call consumes RNG, it
incidentally produced a pure replicate of C1/C2/C3. That accident is the only
reason the noise floor became visible. **A deliberate replicate should have been
part of h156 from the start** — I quoted four agreement percentages without ever
measuring the harness's own reproducibility, which is the most basic control an
instrument owes before its readings are used as evidence.

Guard adopted: any harness whose numbers are going to be compared against real
runs gets a replicate at a different seed BEFORE its agreements are reported.

**Gate misses this tick, all pre-registered and all honoured:**
- h156c gate: R2 PARTIAL. Scale generalised to Hartmann; the required test
  (tracking the narrowing between benchmarks) failed.
- h156d gate: FAIL. Direction corrected, magnitude unmoved. The AND was
  pre-stated precisely so this could not be read as a pass, and it is not.

**What did not change:** the h153 forecast. C2/C1 is 90.9 / 95.5 / 90.9 / 93.2%
across two benchmarks and two replicates — tight against a 6% floor, and
untouched by the C4/C5 misfit.

**Compute:** 10 workers (h153 ×5, h155 ×5) plus at most 2 short offline jobs,
never above 12 of 15.

**Runtime honesty:** h153 and h155 are progressing at roughly 1-2 iterations per
15 minutes per worker and are at 111/240 and 82/240 cost. Per-iteration cost
grows with the accumulated data, so the remaining time is longer than a linear
extrapolation suggests. They are not stalled — the workers hold ~99% CPU each.

## 2026-09-02 (tick 3) — the instrument audit converged, on a less flattering answer

Two ticks ago I built an offline harness and reported that it "reproduces all
four observed rtg_targets, three within 8%". Two audits later the honest
statement is: **it reproduces their SEPARATION (3-4x), not their values**
(per-arm errors 0.5-31%, noise floor 8-13%). Both intermediate claims —
"within 8%" and then "the interpolating-condition misfit is real and
unexplained" — have been superseded by the next audit.

That sequence is not a failure of the audits; it is what auditing an instrument
looks like. But it does mean **the harness should have been characterised
before its readings were quoted**, not after they had already been written into
findings.md and published twice.

**The mechanism question did resolve.** The one-sided under-prediction was the
missing between-model variance (gp_num_models=10 fit on identical data, my
harness used one), and the prediction that it would hurt the low-spread
conditions most was confirmed exactly: C5 −22.0% → −0.8%, C4 −13.3% → −7.1%.
The fix nonetheless broke C3 (+29.4%) and raised the noise floor, so the
ensemble harness is not adopted.

**Discipline misses this tick, both recorded:**
- h156e's gate had a HOLE. PASS was an AND; FAIL and PARTIAL both required
  C4/C5 to stay outside 15%, and they came inside. No clause matched the
  outcome. `tools/check_gate.py` exists precisely for this and I did not run it
  on the protocol. Guard: run check_gate.py on every protocol before locking,
  not only when a gate feels complicated.
- The replicate guard adopted last tick WAS honoured (two replicates, and the
  noise floor was measured before any agreement was quoted). That is the one
  process improvement that held.

**What is untouched by all of this:** the 3-4x separation, h149's reinstated
information-gain mechanism, and the h153 forecast (now stated as 85-96%, from
four measurements rather than one).

**Compute:** 12 workers peak (h153 ×5, h155 ×5, 2 offline replicates), never
above 12 of 15.

## 2026-09-02 (tick 4) — stopped auditing the instrument; used it to forecast instead

**The holistic call this tick was to STOP.** Three consecutive ticks audited the
offline harness. Each audit was honest and each superseded the last claim, but
the returns were shrinking: tick 2 found a real bridge, tick 3 found a real
noise floor, tick 4's marginal audit would have found a smaller correction to a
tool already characterised as scale-only. Continuing would have been running
arms, in the sense the loop prompt warns about — motion inside a question
already answered well enough for the use it is being put to.

So: closed the audit, pre-committed the readout, and spent the tick producing a
**blind forecast for the second running arm**.

**What went right.** The forecast for h155 was produced and committed while h155
was at 184/240 with 0/5 result files. It is genuinely blind, and it is specific:
~102% of the control, near-control performance, explicitly NOT intermediate.
Together with h153's 82-96% band that commits me to outcome O1 before either
number exists, with O2/O3/O4 named and their retractions written down.

**A judgement call worth recording.** The first C6 run would not have beaten
h155's ETA. I killed it at 1/9 states WITHOUT reading its output and relaunched
coarser (6 states, N=100). Trading precision for blindness is the right trade
here and the forecast is stated as a band because of it. Recorded in the
protocol at the time, not afterwards.

**A correction to something I told the user.** Last tick I described the runs as
progressing at "1-2 iterations per 15 minutes" and implied many hours remained.
Measured properly this tick: h153 65.5 cost/hr (ETA ~1.3h), h155 196 cost/hr
(ETA ~0.4h). The pessimistic picture came from comparing iteration counts across
ticks without measuring elapsed wall time, and the ticks were closer together
than I assumed. Both runs have held ~99% CPU throughout.

**Compute:** 10 workers steady (h153 x5, h155 x5) plus 2 short offline jobs at
peak, never above 12 of 15.

## 2026-09-02 (tick 5) — ran the experiment I had been declining, on the cheap axis

**The holistic call: I had been declining the registered POOL dose for many
ticks with a reason that was really an extrapolation.** h146/h149 gave two
endpoints; from those I asserted the dose curve would be flat and skipped ~15
worker-hours. That is defensible ONLY if the cheap version gets run, and until
this tick it had not been. It has now, and it held: quality +51%, diversity
-34%, tail flat within 3.2 points against an 8-13% noise floor.

This is the best-designed test in the whole front. Every earlier "quality does
not matter" result compared arms differing in many ways at once — the exact
shared-confound trap h152 caught. This one moves two variables continuously in
opposite directions on ONE axis, checks the manipulation actually happened
(MC1/MC2 both PASS, serialised, checked before the outcome), and names R1/R2/R3
in advance. Neither costly outcome fired.

**Correction to the ETA I gave the user, for the second time.** Last tick I
measured over 110 seconds and reported h153 at 65.5 cost/hr (ETA 1.3h) and h155
at 196 cost/hr (ETA 0.4h). Measured over a ~7-minute window this tick: **h153 17
cost/hr (ETA ~3.5h), h155 35 cost/hr (ETA ~1.0h)**. Short windows catch bursts.
The rate is also genuinely declining as per-iteration cost grows with the data,
so even the 7-minute figure is likely optimistic. Guard: never quote an ETA from
a sub-5-minute window on these runs.

**Compute:** 12 workers at peak (h153 x5, h155 x5, dose x2), never above 12/15.

**Still pending:** h153 and h155 themselves, against the blind forecasts
committed last tick (82-96% and ~102% of control, both predicting near-control
performance; O1-O4 retraction map locked in h157/protocol.md).

## 2026-09-02 (tick 6) — the forecast paid off, and the last fallback explanation died

**h155 completed at n=5 and its blind forecast held.** Predicted 102% of the
control's conditioning target; measured 106.6%, inside the harness's own noise
floor. Predicted "near the control, and explicitly NOT intermediate"; measured
15.13 against 15.82, improving 5/5 exactly like the control, better on 4 of 5
seeds and tracking it seed-by-seed.

That is the payoff for last tick's decision to stop auditing the instrument and
use it to forecast instead. A harness characterised as "scale claims only"
turned out to be good enough to call a pipeline result in advance, which is a
much better use of it than another round of calibration would have been.

**"The MES rule specifically" is retracted.** It survived h150 and had been the
standing fallback. A UCB teacher matches the control on every measure. The 2x2's
non-MES row now reads: closed-loop works, open-loop fails.

**The confound check was run before the number was read**, as registered, and
passed cleanly (HF 0.87 vs 0.88). h60's thompson arm collapsed to 2/196 HF and
is uninterpretable for this comparison; holding the fidelity channel fixed by
construction was the right design decision and the data confirms it worked.

**Still able to refute everything:** h153 at 210/240. If it lands at ~43.94,
outcome O3 fires and the tail account, this retraction, and the report callout
all come out together. That is written into h157/protocol.md and into the
published report itself.

**Compute:** 10 workers, dropping to 5 as h155 finished. Never above 12/15.

## 2026-09-02 (tick 7) — the 2x2 completed and killed my own explanation

**h153 landed at n=5 and split its blind forecast exactly where it mattered.**
The performance half held (19.36 rel%, improving 5/5, against the failing arms'
43.94 and 0/5). The rtg_target half failed badly (0.3230 against a forecast of
0.83-0.94).

That split IS the result. h153 carries the failing arms' conditioning target and
the control's improvement rate simultaneously, which **refutes the causal claim
I had been building for four ticks**: target collapse does not cause the failure.
Withdrawn from findings.md and from the published report. The descriptive half —
the failing arms genuinely have no informative tail — was validated against four
real arms and survives untouched. It is simply not the cause.

**The instrument failure was diagnosable in advance and I did not diagnose it.**
C2 was the only harness condition with no finished arm to check it against. Every
checked condition held; the unchecked one was wrong by 2.7x. SC2 shows the
mechanism: the real freeze penalty is ~0.35 against the harness's 0.16.
Guard adopted: **never forecast from an unvalidated condition without labelling
it as such.** I quoted "C2 retains 90.9%" for several ticks with the same
confidence as the validated numbers.

**Holistic read: this front is NOT answered and should not be recorded as such.**
The negative result is strong and was earned by direct sweep (h158: quality +51%,
diversity -34%, outcome flat). But the positive mechanism is now unidentified for
the second time — first "MES specifically" (retracted by h155), then
"target collapse" (refuted by h153). The honest state is: three of four cells
work, only the doubly-changed cell fails, and why is open.

**A new post-hoc hypothesis is recorded and explicitly NOT claimed:** the working
arms all query model-selected locations, the failing arms do not. The
discriminating experiment (a frozen path of model-selected locations from a
MISMATCHED state) is registered and not run.

**Compute:** 6-10 workers; h159 still running at ~80/240 and now known not to
discriminate between the accounts, which was recorded before it produced results.

## 2026-09-02 (tick 8) — two arms designed, neither funded, and that is the result

**No pipeline arm was launched from h159's or h160's designs.** Both were locked
as discriminators between the information account and the model-selected
account; neither is one. In both cases I described the arm verbally ("not
information-seeking", "anti-informative") without checking what the mechanism
actually implies — and β=0 and β=−2 are still MODEL-SELECTED, so both accounts
predict success.

**The cheap-screen-before-funding pattern paid for itself twice.** ~40 minutes of
harness screens prevented ~20 worker-hours of arms whose outcomes both accounts
already predicted. That pattern is now the default for any expensive arm:
forecast it first, and treat "both accounts predict the same thing" as a NO-GO.

**A gate was honoured under pressure to rationalise around it.** h160's round-1
screen returned 75.8% against a 70% threshold — technically NO-GO, but on two
replicates spread 31% apart. The honest move was neither to launch (the letter
said no) nor to move the threshold, but to reduce the noise on the quantity being
compared: two more replicates, threshold untouched, decision recorded in the
protocol before they returned. The 4-replicate answer was 83.7%, and NO-GO stood.

**A separation fell out of the screens for free.** Tail as % of control:
β=+2 → 102%, β=0 → 91.5%, β=−2 → 83.7%, frozen-external arms → ~30%. Monotone in
β and never near the failing band. With h153 (frozen, target 0.323, performance
fine), this establishes: **rtg_target tracks adaptivity; performance does not
track rtg_target.** The conditioning target is now demoted from "the mechanism"
to "a correlate", which is the second demotion of a candidate mechanism in two
ticks.

**h161 launched as the actual discriminator** — a frozen path from 2000 rollouts
earlier (~10 iterations), model-selected but for a stale state. The two accounts
disagree here, which is the whole reason to fund it.

**h159 kept rather than killed**, with the reason recorded: it is
non-discriminating for the accounts but validates the harness's unvalidated C7
condition, and h153 showed such conditions can be off by 2.7×.

**Compute:** 10 workers (h159 ×5, h161 ×5), plus 2-4 short screens at peak.
Never above 12/15.

## 2026-09-02 (tick 9) — a smoke test earned its keep, and no science happened

**Honest holistic read: this tick produced no substantive finding.** It caught a
design error and verified an instrument. That is worth doing and worth saying
plainly rather than dressing up as progress.

**The error.** h161's LAG was locked at 2000 rollouts on the assumption that a
batch is `rollouts_per_iter=200`. It is not: the STATE-DIAG line reports
**n_traj=60** per iteration. LAG=2000 would have been ~33 of the ~60 real
iterations, leaving **more than half the run in warmup** — using the current path
and running byte-identical to h153 over that stretch. The manipulation would have
been diluted to roughly half strength and the arm uninterpretable. Corrected to
LAG=600 (~10 iterations, ~83% stale), h161 killed ~4 iterations in and relaunched.

**The verification.** Small-LAG smoke run, 360 rollouts: SC1 stale fraction
0.833, SC2 mean lag exactly 60, SC4 replay error 0.0, and — the check that
actually matters — the stale path differs from the current one in **50 of 50**
comparisons. Had those coincided, h161 would have been h153 under another name
and would have "confirmed" whatever h153 showed.

**Cost/benefit:** ~5 minutes and one worker, against ~10 worker-hours and an
uninterpretable arm. This is the third time this session that a cheap check
before an expensive arm has paid for itself (h159 screen, h160 screen, h161
smoke). The pattern is now standing practice and is recorded as such.

**What is NOT true:** that catching my own error is a result. The front's
mechanism is still unidentified after two demotions ("the MES rule" retracted by
h155, "target collapse" refuted by h153), and h161 is the arm meant to address
that. It has not reported.

**Compute:** 10 workers (h159 x5, h161 x5) plus 1 smoke at peak = 11/15.

## 2026-09-02 (tick 10) — a confirmed forecast, and an out-of-sample test that cost me a claim

**h159 completed and its blind forecast held** (91.5% predicted, 96.4% observed).
Keeping it running after its screen showed it non-discriminating was the right
call for the reason recorded at the time: it validates a harness condition. The
harness record is now 6 of 7, and the single failure (C2, frozen) is
structurally distinct from the six that hold. **That gives a usable rule: the
harness is accurate for closed-loop conditions and wrong for frozen ones.**
Consequence acted on immediately — h161 is a frozen condition, so no forecast for
it will be offered, where previously I would have produced one.

**The most useful thing this tick was an out-of-sample test that partly failed.**
h163 found teacher and student dispersion inverted at Spearman −0.900 across five
arms, with the sign opposite to the natural null. That is a strong result and it
would have been easy to report as one. Instead the prediction it implied for the
sixth arm was written down and committed BEFORE computing it: h159's teacher
dispersion is the lowest of all six, so its student should be the highest.
Observed 0.2639 — above the pre-set retraction threshold but third of seven, not
first. Spearman falls to −0.771.

So the claim is downgraded from a rank relationship to a **group separation**.
Without the out-of-sample step I would have published −0.900 as a relationship,
and it is not one. This is the second time this session that committing a
prediction before computing it changed what I was entitled to say (the first was
h153's split forecast).

**Second decoupling of target from performance, by an independent route.** h159
(target 0.941) and h153 (target 0.323) differ threefold and land at 19.07 and
19.36, both improving 5/5. The conditioning target does not drive performance.

**Holistic read.** The front now has a solid negative (quality and diversity are
orthogonal, swept directly), a complete 2x2 with four working arms and one
failing group, two independent demonstrations that rtg_target is not the
mechanism, and a resolved L_loc puzzle. The positive mechanism is a learnability
framing that has survived two tests and been downgraded once. h161 is the arm
that can test it causally and it is the only thing running.

**Compute:** 5-6 workers. Never above 6/15 this tick.

## 2026-09-02 (tick 11) — generality, which the front had been quietly short of

**Holistic read: almost everything substantive was Borehole-only.** The complete
2x2, h158's quality/diversity dose, h162's dispersion split and h163's inversion
— all one benchmark. Hartmann had only the control and two FAILING arms. That is
a real gap and it had gone unremarked for several ticks while I chased mechanism.

**h164 closed half of it for free.** The dispersion collapse replicates on
Hartmann using data already on disk: control 0.2144 against RANDOM-POOL 0.1153,
5 of 5 paired within seed, M2 signature identical. Reported honestly: the
HF-only slicing is unusable there (the control has <12 HF queries on three of
five seeds) and is not counted, and Hartmann's ORACLE arm is flagged confounded
so it is shown but not counted either.

**h165 launched to close the other half.** The strongest positive result of the
front — h155's UCB teacher matching the control, which retracted "the MES rule
specifically" — has never been tested off Borehole. R1 is named: if it fails
there, that retraction is Borehole-specific.

**Compute:** 10 workers (h161 x5, h165 x5). Never above 10/15.

**Still the only causal test:** h161, at 130/240.

## 2026-09-02 (tick 12) — consolidated the front into a readable report

**Holistic call: the deliverable had degraded even as the science improved.** The
existing to_human report is 1.5 MB of chronological accretion with three
correction callouts stacked on one another, each withdrawing the one above it.
Every correction was honest, but the result is unreadable: a reader cannot tell
what is currently believed without reconstructing the whole sequence.

Rather than patch it a fourth time, wrote a **new focused report on this front
alone** (to_human/teacher-question.html): the settled negative, the completed
2x2 as an actual 2x2, the three retracted explanations each paired with the
experiment that killed it, the current account with its evidence, and the causal
test still running. The older report stays as the project-wide record.

Design was deliberately utilitarian — a research memo, not a landing page.

**Nothing new was measured this tick.** h161 (152->160/240) and h165 (Hartmann
UCB-LOC, 126-231/240 across seeds) are both mid-flight and neither will land
before the next tick. Recorded plainly rather than padded.

**One correction to my own copy before publishing:** I first wrote "sixteen
experiments", then "nineteen", and only then counted — the front is h145 through
h165, twenty-one. Guess-then-check on a number that appears in the first
paragraph of a published page is exactly the kind of thing that erodes trust in
everything after it.

**Compute:** 10 workers throughout (h161 x5, h165 x5).

## 2026-09-02 (tick 13) — closed the biggest generality gap, and consolidated the state file

**Holistic read: the front's central structural result was still one benchmark.**
The 2x2 — three of four cells work, only the doubly-changed cell fails — is
Borehole-only, and so is h153's refutation of the target-collapse account, which
is stated in findings.md AND in the published report without a benchmark
qualifier. That is a claim resting on five runs of one function.

**h166 launched to close it.** h153's worker unchanged, on Hartmann. Hartmann is
~4x cheaper per seed (h165 seed 43 finished in 19 minutes against Borehole's 80),
so a two-pass arm there costs about what a one-pass Borehole arm does — which is
why this gap was affordable to close now and was not before.

R3 was named explicitly because it is the outcome that looks like success and
carries no evidence: if Hartmann MES-FROZEN works but its target does NOT
collapse, the split does not reproduce and Hartmann cannot corroborate the
refutation either way.

**research-state.yaml consolidated.** It had fallen several ticks behind
findings.md. It now carries, in one block: the settled negative, the complete
Borehole 2x2, all four retractions with the experiment that killed each, the
current account with its evidence and its downgrade, the harness accuracy rule
(accurate closed-loop, wrong frozen — so no forecast is offered for h161 or
h166), and the two screens that prevented ~20 worker-hours of non-discriminating
arms.

**Nothing new was measured this tick.** h165 is 1/5 with seeds spread 153-264 of
293; h166 just started; h161 is at 183/240 and remains the only causal test.

**Compute:** 14/15 workers (h161 x5, h165 x4, h166 x5) — the highest this
session, and within the rule.

## 2026-09-02 (tick 14) — the fourth explanation fell, and the failure moved

**h167 was zero compute and overturned more than anything I have run.**

The current account said the failing teachers' actions are unlearnable from the
observable state. Two measurements killed it:

1. **Where the collapse lands.** All three failing arms' queries sit within 0.04
   of the exact centre of the normalised box in 8 dimensions — regardless of
   where their teachers aimed. ORACLE's teacher averages to a point 0.66 away;
   its student does not go there. So the collapse is a property of the network,
   not of the target distribution. That alone was R2, which I had pre-named as
   the likeliest outcome and as NOT support.

2. **The DT is not collapsed during training at all.** Its loss (0.018-0.022) is
   **2.5-4x lower than the best possible constant predictor** (0.053-0.083) on
   every failing teacher. The actions are demonstrably learnable, and learned.

So training succeeds and inference collapses. **The failure is a training/
inference disagreement, not a learnability failure.** That is a relocation, not
a refinement.

**Fourth explanation to fall on this front**: the MES rule (h155), target
collapse (h153), open-loop/adaptivity (h156), learnability (h167). Worth being
blunt about the pattern in any write-up: the FACTS have been stable throughout —
the 2x2, the dispersion split on two benchmarks, the flat quality/diversity dose
— and it is the explanations that keep outrunning them.

**On resuming a line I had a stopping rule about.** h167 points at the inference
conditioning, which is h148's territory, and h148 registered "if P1 fails I stop
trying to explain this with RCSL theory". Checked rather than assumed: h148's
recorded outcome was **"P1 not evaluable"** — it needed per-record RTG stats that
were never serialised — so the rule never triggered and its question was never
answered. h168 is registered with that reasoning stated up front, and with the
distinction that matters: h167 reached the conditioning by direct measurement
from an unrelated direction, not by generating a third proxy until one fit.

**h168 queued, not launched.** Compute is at 14/15 with h161, h165 and h166
running. Its R1 is the important one: if the emitted x sits at the box centre at
EVERY conditioning value, the conditioning is exonerated and the suspect becomes
the network itself — and that would close the line evaluably, which h148 never did.

**Compute:** 14/15 throughout. Nothing new launched.

## 2026-09-02 (tick 15) — the causal test landed, and I closed a five-time failure

**h161 reported (n=3) and it WORKS**: 21.17 rel%, improving on every seed,
target 0.3388. Sanity checks exact on all three — stale fraction 0.902-0.904,
lag exactly 600, replay error 0.0. Verdict R2: **model-selected for the CURRENT
state is not required**; a ten-iteration-stale model's path costs ~1.8 rel% and
no improvements.

R2 was written when the learnability framing was live, and h167 retracted that
framing one tick before h161 reported. So the causal test landed on a position
already withdrawn — not wasted, but a reminder that a two-hour arm can be
overtaken by a zero-compute measurement while it runs.

**Closed the instrumentation gap I have logged five times.** `actions_x` was
never serialised, which is why h167c's control baseline had to be reconstructed
— and the control is the arm carrying that result's surprise. mf_dro.py now
accumulates the per-iteration mean and total variance of the batch's actions
(d+1 floats), h83's worker serialises it, and the **bit-identity gate PASSED**
(regret 122.2906675273 identical).

**I committed broken code.** The worker half went in with wrong indentation and
was unparseable. Running arms were unaffected (already imported), but any new
launch would have failed at once. The cause was running the syntax check AFTER
`git commit` rather than before, in the same compound command. Fixed in the next
commit, and the gate was re-run on the fixed file rather than trusting the
earlier PASS, which had only exercised the mf_dro half.
Guard: syntax-check before commit, never in the same breath after it.

**Holistic:** the front's facts keep accumulating cleanly — the 2x2, now five
working arms against three failing ones; three independent demonstrations that
the conditioning target does not predict performance; the dispersion split on two
benchmarks. The explanations keep failing (four so far). h168 is the registered
next test and is still queued on compute.

**Compute:** 9-13 workers.

## 2026-09-02 (tick 16) — h161 complete, h168 built and launched, and a metric check

**h161 COMPLETE at n=5: 19.53 rel%, improves 5/5, target 0.3270.** Sanity exact
on every seed (stale 0.902-0.907, lag exactly 600, replay error 0.0). Against
h153's fresh model-selected path (19.36) the gap is **0.17 rel%** — a
ten-iteration-stale model costs essentially nothing. The n=2 read of 21.17
overstated it by 1.6 points, which is why no verdict beyond R2 was taken at n=2.

**h168 built, gated and launched.** The probe re-queries the DT at nine RTG
values on the same state, saving and restoring the RNG generator around the
sweep — without that, an active probe would perturb every later draw and the arm
would stop being comparable. OFF by default; **bit-identity gate PASSED**
(regret 122.2906675273). Smoke-tested first, because the probe swallows
exceptions and a scope error would have looked exactly like a silent no-op.

**The smoke test's early signal was recorded BEFORE the arm ran**, and it points
at R1: at 5 iterations the emitted x sits ~0.38 from the box centre at every
conditioning value and moves only 0.028 across the whole sweep. P2 needs a 2x
gap; this is 0.96. Five iterations of a barely-trained network is not a result
and no verdict was taken — but if it holds at full length, the conditioning is
exonerated and the suspect becomes the trained network itself.

**A user question produced a useful robustness check.** Asked whether teachers
were compared by final simple regret: they were — the frozen metric IS simple
regret, and because every arm terminates at cost ~240, "cost 200 post-init" is
the end of the run and the two numbers are **identical to 2 decimals on all
eight arms**. No conclusion depends on the metric.

The raw column also made something plainer than the normalised one: the three
failing arms report the **same** final regret, 136.0315, to four decimals. They
are not performing similarly — they are on a floor set by the initial design.
"A perfect teacher ties a random one" is misleading; "both hit the floor" is
accurate. Rephrased in findings.md.

**A monitoring bug, not a data bug:** my progress grep matched `cost=` inside the
stopping message (`post_init_cost=201.0 >= 200.0`) and appeared to show a run
going backwards. Checked before reporting it.

**Compute:** 12/15 (h165 x3, h166 x4, h168 x5).

## 2026-09-02 (tick 17) — Hartmann generality lands, and an audit finds a dead code path

**The 2x2's separation is not a Borehole artefact.** Hartmann working arms
2.22-11.28 rel% against failing arms 52.23-65.14. h165 (UCB-LOC) works at 11.28,
improving 4/4, so h155's retraction of "the MES rule specifically" holds on a
second benchmark. The ordering flips though — h155 BEAT the Borehole control
(15.13 vs 15.82) while h165 TRAILS the Hartmann one (11.28 vs 7.99).

**The pre-registered confound fired, and I worked out which way it cuts before
deciding what to do about it.** h165's HF fraction is 0.353 against the control's
0.200, higher on all four paired seeds. But more HF queries should HELP — HF is
the real objective — and h165 buys more HF while performing worse. So the
confound cannot manufacture h165's success; at most it explains its shortfall.
A confound that biases AGAINST the claim is not grounds to withhold the claim,
and that reasoning is now in findings.md rather than just the bare observation
that a confound exists. Registering the check was still worth it: without it I
would not have known the direction.

**A user question turned into a code audit with a real finding.** Asked whether
we could adopt the DRO paper's rotating-acquisition schema (paper D.4: EI, UCB,
PI, MES). Read the paper, then the code: `TEACHER_POOL` and `use_teacher_pool`
already exist and have NEVER been activated by any experiment. Inspection shows
why they should not be: `cost_ei` computes EI of the **LF** function against the
best **HF** value, and `ucb_beta1/3` compares LF's own UCB to HF's. Four of five
members are incoherent; MES is the only correct one, because its LF term measures
information about y*_H rather than about the LF function.

That is the kind of defect that would have silently produced a "rotation doesn't
help" result if switched on naively. Recorded as an audit finding, not an
experiment.

**Holistic:** the facts continue to accumulate and generalise; the explanations
continue not to. h168 (the conditioning probe) is the live test and its smoke
signal already points at R1 — the conditioning exonerated, the suspect becoming
the trained network itself.

**Compute:** 10/15 (h165 x1, h166 x4, h168 x5).

## 2026-09-02 (tick 18) — a line closed evaluably, and the suspect moved again

**h168 completed and R1 fired.** The emitted action moves 0.0074 across the full
RTG sweep from 0 to 1 — 8.9% of its own mean, ratio 0.986 against P2's required
2.0, and drifting AWAY from the box centre as RTG rises. **The inference
conditioning is exonerated.**

That closes the line h148 opened and never resolved. h148's P1 was "not
evaluable" because the statistics it needed were never serialised; h168 answered
the same question by direct measurement on 357 probed iterations. Worth
recording as a pattern: the fix for an unevaluable registration is a different
measurement, not a different proxy.

**The probe was verified bit-identical to its unprobed twin** — x_t_trace and
final_regret identical on all five seeds against h149's run of the same policy.
That is stronger than the probe-OFF gate I actually registered, and it means the
RNG save/restore was not just defensible in principle but exact in outcome. I
had not planned that check; it was available for free because h149 existed.

**The smoke test earned its keep again.** Its 5-iteration signal (spread 0.028)
was recorded before the arm ran and predicted the full-length result (0.0074).
Third time this session a cheap pre-check has either prevented a wasted arm or
called one in advance.

**h169 launched.** With RTG excluded and the network known to fit its teacher
2.5-4x better than any constant, the remaining suspects are the state and the
auxiliary conditioning. h169 crosses STATE (real vs a tau=0 state from this
iteration's own training batch) with RTG, four cells on the same network.

**R2 named first, deliberately:** if the action sits at the box centre in all
four cells, the whole distribution-shift family dies at once and the suspect
becomes the inference code path. That would be the cheapest outcome to act on
and the most embarrassing to have missed for eighteen ticks, which is exactly
why it goes first in the protocol rather than last.

**Holistic:** five explanations have now fallen (MES-rule, target-collapse,
adaptivity, learnability, conditioning). The facts have never moved. That
asymmetry is itself the most robust finding of the front and should lead any
write-up.

**Compute:** 4-10 workers.

## 2026-09-02 (tick 19) — a smoke test caught my own no-op, and the no-op was the lead

**I skipped h169's smoke test and it cost me an arm.** h168 got one because its
probe swallowed exceptions; h169's state axis was equally new code with the same
swallowing behaviour, and I launched it without one. Ran the check late, found
the state axis was a **silent no-op** — the probe sampled four rollouts of the
same ensemble member, whose tau=0 states are identical by construction (pairwise
distance 0.0000). Killed the arm at ~60% rather than let it finish and be
written up as a state manipulation it never performed.

Guard: **the smoke test is not optional for probe code.** It is optional for code
that fails loudly.

**The no-op was worth more than the arm.** Fixing the sample to stride across
ensemble members showed the real inference state is **bit-identical to a tau=0
training state**, and `[STATE-DIAG]` — a line printing in every log this project
has ever produced, which I have never once followed up — reports
`uniq_tau0_states=3` of 60 on every seed. mf_dro.py's own docstring already spells
out the consequence: with degenerate tau=0 states "the DT could only ever learn
the conditional mean of that timestep's targets". That documents a bug since
PARTIALLY fixed, and the consequence was never revisited after the partial fix.

**The mechanism that follows matches all seven arms**, and explains why h167's P2
failed (I used the all-tau action marginal when inference only ever sees tau=0).
Every failing teacher's tau=0 action is an independent draw whose mean is the box
centre; every working teacher's is an acquisition argmax.

**It is not being called the answer.** Five accounts have fallen on this front by
fitting the evidence and outrunning it, and seven matching SIGNS is exactly that
pattern. h170 tests it as a NUMBER, on the working arms -- the only arms where
"tau=0 teacher mean" and "box centre" are different targets. R1 (they are not
close) is named first.

**Holistic:** the most useful thing this tick was reading a diagnostic that had
been in front of me for nineteen ticks. Worth asking, next time progress stalls,
what the logs are already printing that I have stopped seeing.

**Compute:** 3-8 workers; h169 killed freed five.

## 2026-09-02 (tick 20) — the first account to survive a test it could have failed

**h170 passed.** On the working arms the DT's query is 3.3x and 3.2x closer to
its teacher's tau=0 action mean than to the box centre. That is the
discriminating comparison — on the failing arms those two targets coincide, so
only a working arm can separate them. P2 calibrated, P3 held 4/4.

The account: inference always queries timestep=0; at tau=0 the training states
are near-degenerate; so the DT emits roughly the conditional mean of its
teacher's tau=0 action. Teachers whose first step is an acquisition argmax hand
it an informative point; teachers whose first step is an independent draw hand it
the centre of the box. It also explains why quality and diversity never mattered:
both describe where the trajectory GOES, and only its first step reaches
inference.

**Not claimed as established.** The residual is 5 SE — the query is near the
tau=0 mean, not at it. Registered outcome R2: survives ONE non-post-hoc test.
Six accounts proposed on this front, five fell; this is the first to predict a
number before being checked and have it hold.

**I amended the protocol before running rather than after.** It said fresh runs
were needed; I found the tau=0 action is reconstructible offline from existing
traces, recorded the amendment with the new caveat it introduces, and ran the
cheap version. Ten worker-hours saved and the test could use five completed arms.

**I repeated an error I had already written a guard for.** The first run used 12
reconstruction draws; P2 "failed" at 0.224 against a 0.15 threshold. But the mean
of 12 uniform draws in 8D sits 0.2289 from the centre — my gate was BELOW my
estimator's own noise floor and could not have passed however right the theory
was. Same class as h156's withdrawn "within 8%": quoting an agreement without
first measuring what the instrument can resolve. The guard was adopted after
h156 and not applied here. Re-run at 120 draws, P2 calibrates.

**Compute:** 3 workers (h166 finishing). h170 needed one, for minutes.

## 2026-09-02 (tick 21) — from correlation to intervention

**The right move after h170 was not more correlation.** h170 showed the DT's
query sits 3.3x closer to its teacher's tau=0 action mean than to the box
centre. But five accounts have fallen on this front and every one of them fitted
the correlations available at the time. Another correlational confirmation would
have been the same trap in a new costume.

**h171 tests the mechanism by intervention**, and the prediction is
counter-intuitive enough to be worth something: split the rollout so one arm
takes the acquisition argmax ONLY at tau=0 and random for the other seven steps,
and the other arm does the reverse. **TAIL-MES is the better teacher on 7 of 8
steps and is predicted to FAIL.** No trajectory-quality account predicts that
ordering; the tau=0 mechanism requires it.

R1 (HEAD fails) and R2 (TAIL works) are each individually fatal to the account
and are named first in the protocol.

**SC2 was run before launch, deliberately, because h169 was lost to skipping
exactly this check.** Both branches are new code and a mis-wired split would have
silently produced two copies of the same arm. Measured spreads: head 0.154 at
tau=0 against 0.286 after; tail 0.289 against 0.183. Mirror images, ratios 0.539
and 1.577. SC3 bit-identity also passed.

**If R3 holds it is directly actionable**, not just explanatory: seven of every
eight rollout steps would be wasted computation, in a method whose dominant cost
is rollout generation.

**Compute:** 13/15 (h171 x10, h166 x3).

## 2026-09-02 (tick 22) — the Hartmann 2x2 completes; a refutation becomes two-benchmark

**h166 fired R2 at n=3 and it matters more than a third confirming arm usually
would.** Its conditioning target is 0.4002 -- in the failing band, far from the
control's 0.8844 -- while it posts the BEST regret of all four Hartmann arms
(6.75 against the control's 7.99). That is h153's split reproduced on a second
benchmark.

The consequence is specific: h153's refutation of the target-collapse account is
stated in findings.md AND in the published report **without a benchmark
qualifier**, and until this tick it rested on Borehole alone. It now rests on two
benchmarks and three independent frozen arms (h153, h161, h166). A claim that was
over-scoped is now correctly scoped, without having to be weakened.

R3 -- works but the target does NOT collapse, the outcome that would have looked
like success and carried no evidence -- did not fire. Naming it in advance was
what would have stopped me reading a null as support.

**The Hartmann 2x2 now shows three of four cells working**, the same structure as
Borehole. The front's FACTS continue to replicate across benchmarks while its
EXPLANATIONS continue not to survive -- five have fallen, and the sixth (tau=0)
is under interventional test right now.

**h165 closed at n=5** (10.58, 5/5). Its HF-fraction confound shrank from 0.353
to 0.290 and still points against the claim it might have threatened.

**An observation I am deliberately not counting as evidence:** h171's two arms
are running at very different speeds -- HEAD at cost 113, TAIL at 72 -- because
TAIL makes seven MES calls per rollout to HEAD's one. If HEAD works, the method's
dominant cost is being spent on steps that never reach inference. That is a
statement about wall-clock, not about the mechanism, and it is logged as such
until h171 reports.

**Compute:** 12/15 (h171 x10, h166 x2).

## 2026-09-02 (tick 23) — a blind forecast on query LOCATION, and a stale published line fixed

**Committed a quantitative blind forecast for h171 while both arms were 0/5.**
h170 measured the tau=0 action mean of each teacher rule on *other* arms: MES
argmax sits 0.6831 from the box centre, a uniform draw 0.0712. h171's arms have
exactly those tau=0 rules by construction, so the mechanism predicts not just the
regret ordering but **where each arm's learner will query**: HEAD's centroid
beyond 0.5 from the centre, TAIL's inside 0.2.

That F1 half is the stronger test. F2 (the regret ordering) is a two-way call
that could come out right by luck; **F1 is a numeric prediction of a location in
eight dimensions, derived from measurements made on arms that did not exist when
h171 was designed.** If F1 holds and F2 does not, the account describes where the
learner queries but not why it matters; if F2 holds and F1 does not, the ordering
is right for some other reason. Both stated before either arm reported.

**Fixed a stale line in the published report.** It said the stale-model test "is
in progress" and framed the model-selected hypothesis as the live account. h161
finished several ticks ago and h167 retracted the learnability reasoning behind
that framing. Published material carrying a superseded account is the thing I
criticised myself for earlier in this session; leaving it while waiting for h171
would have repeated that.

**A small process failure worth noting:** my first attempt to patch the report
asserted on a string containing `&mdash;` when the file has literal em-dashes.
The assertion caught it rather than silently writing a mangled page -- which is
why the patch scripts assert instead of blind-replacing. Fixed by matching on
line numbers after reading the actual text.

Also committed the h165/h166 result files, which had been left untracked.

**Compute:** 11/15. h171 HEAD ~73%, TAIL ~40% (TAIL is slower because it makes
seven MES calls per rollout to HEAD's one).

## 2026-09-02 (tick 24) — h166 closes the Hartmann 2x2; h171's HEAD half lands as forecast

**h166 finished at n=5 and ties the control exactly** -- 7.93 against 7.99,
improving 5/5 like it -- while its conditioning target sits at 0.3824, nearer
RANDOM-POOL's 0.2924 than the control's 0.8844. R2 confirmed at full n.

Four independent arms now carry a collapsed target AND good performance: h153
(0.3230), h161 (0.3270), h166 (0.3824), h171-HEAD (0.3720, preliminary). The
refutation of the target-collapse account, which findings.md and the published
report both state without a benchmark qualifier, is now solid on two benchmarks
and four arms rather than one benchmark and one arm.

**h171's HEAD half landed at n=2 and both forecast halves hold.** F1 predicted a
query centroid beyond 0.5 from the box centre; observed 0.7287 (control 0.7604,
RANDOM 0.0239). F2 predicted near-control regret; observed 17.15 against 15.82,
improving 2/2.

The striking part is what HEAD-MES is: it consults the acquisition on **one step
in eight** and moves at random for the other seven, and it performs like the
control. That is the mechanism's prediction and no trajectory-quality account
makes it.

**I am not calling it yet.** TAIL-MES -- the arm that follows the acquisition on
SEVEN of eight steps and which the mechanism requires to FAIL -- is at ~52% and
untested. R2 (TAIL works, therefore tau=0 is not necessary) remains individually
fatal to the account. A one-sided confirmation is exactly the shape of evidence
that has misled this front five times, and the half still running is the half
that can kill it.

**Compute:** 7/15.

## 2026-09-02 (tick 25) — HEAD lands as forecast, and I caught myself over-claiming a speedup

**h171 HEAD-MES finished at n=5 and both forecast halves hold.** F1 predicted a
query centroid beyond 0.5 from the box centre; observed 0.7397 (control 0.7604,
RANDOM 0.0239). F2 predicted near-control regret; observed 16.96 against 15.82,
improving 5/5. SC1 passed (HF fraction 0.825 vs 0.883, no collapse).

The arm consults the acquisition on ONE step in eight and moves at random for the
other seven, and it performs like the control. That is the mechanism's prediction
and no trajectory-quality account makes it. Its conditioning target is 0.4096 --
the fifth arm to pair a collapsed target with good performance.

**I wrote "2.1x faster" and then checked it, which was the wrong order.** HEAD
took 39.6 minutes against the control's 82.4, and the per-query times order
exactly as MES-call count predicts (8 calls -> 0.773 min/query, 1 -> 0.360,
0 -> 0.191). But all three arms ran at different times under different worker
counts, so contention is uncontrolled and the comparison is not clean. Withdrawn
as stated; the honest version is HEAD vs TAIL, which ran concurrently on the same
machine and differ only in MES calls per rollout. That number comes when TAIL
finishes.

This is the third time this session I have quoted a number before measuring what
could contaminate it (h156's "within 8%", h170's gate below its estimator's noise
floor, now this). The pattern is specific: I check the thing I am claiming, but
not the thing that could explain it away.

**Still no verdict on the mechanism.** TAIL-MES is at ~82%. It follows the
acquisition on SEVEN of eight steps and the account requires it to FAIL. HEAD
working is also consistent with a weaker claim -- "one good step early is enough"
-- that needs none of the tau=0 machinery. Only TAIL separates them.

**Compute:** 5/15.

## 2026-09-02 (tick 26) — the front is answered, and the answer is actionable

**h171 TAIL finished and R3 fired.** Both halves of a forecast committed at 0/5:
HEAD (acquisition on 1 of 8 steps) 16.96 rel%, 5/5, query centroid 0.7397
against a predicted >0.5. TAIL (acquisition on 7 of 8) 43.94, 0/5, centroid
0.0313 against a predicted <0.2. **The better teacher on seven steps out of eight
fails completely.**

Neither individually-fatal outcome occurred. This is the first claim on this
front established by INTERVENTION rather than by fitting correlations -- which is
exactly what the five fallen accounts could not do.

**SC1 fired on TAIL and I did not wave it away.** TAIL's HF fraction collapsed to
0.217 against the control's 0.883, and I had registered that a collapse voids the
arm. By the letter it does: TAIL alone cannot attribute its failure to the tau=0
location rule. The attribution rests on ORACLE and DIVERSE-GOOD, whose HF
fractions are 0.626 and 0.604 -- no collapse -- and which still fail at 43.94.
Written that way rather than as "the collapse is fine because the theory predicts
it".

**The collapse is separately a confirmation nobody aimed at.** The fidelity head
is also emitted at tau=0 and obeys the same rule: both arms whose tau=0 fidelity
is a 25% coin flip land at HF 0.217 and 0.256; every other arm sits at 0.60-0.90.

**Compute claim now contention-matched.** HEAD and TAIL ran concurrently on the
same machine: 39.6 vs 72.2 minutes, 1.82x, differing only in MES calls per
rollout. The earlier cross-run "2.1x" stays withdrawn.

**h172 launched to test the actionable implication** -- rollout_length {1,2,4}
against the control's 8. Its asymmetry is registered up front: a NULL would not
refute h171, because shortening the rollout also changes the RTG/BTG labels and
the DT's context length. That had to be said before the result, not after.

**Holistic: the front is answered.** Question: why does better trajectory quality
not improve MF-DRO? Answer: only the teacher's first step reaches inference.
Quality describes where a trajectory GOES; a perfect route's first step is a
random start point whose average is the middle of the box. Six accounts were
proposed, five fell, and the survivor was confirmed by changing something rather
than by explaining what was already there.

**Still open and recorded as such:** the tau=0 conditional-mean account leaves a
~5 SE residual, so something else is also operating.

**Compute:** 15/15 (h172), at the cap and nothing else running.

## 2026-09-02 (tick 27) — the front pays out: one-step rollouts beat eight-step ones

**h172 L=1 finished at n=5 and R1 fired.** 13.69 rel% against the control's
15.82, improving 5/5, better on 4 of 5 seeds -- and in **13.2 minutes against
82.4**. P1 predicted "within ~3 rel%"; it came in 2.13 rel% BETTER.

h171 showed the seven later rollout steps do not reach the real query. h172 shows
they can simply be deleted. **This is the first change to what the code should do
to come out of this front** -- everything before it was an explanation.

**The contention confound runs the protective way this time, and I checked which
way before quoting the number.** h172 ran at 15/15 workers, the highest
contention of the session, against a control at unknown and probably lower load.
A 6.2x gap measured under WORSE conditions can only be understated. That is an
argument about direction, not a matched measurement, and it is written that way
-- unlike h171's "2.1x", which I quoted first and checked second.

**h173 launched: h171 on Hartmann.** The front's ANSWER is currently stated in
findings.md, research-state.yaml and the published report **without a benchmark
qualifier**, and h171 is Borehole-only. Every other load-bearing result here has
been made to replicate, and twice the second benchmark changed what could be
claimed -- h164's HF-only slicing turned out unusable there, and h165's ordering
flipped relative to Borehole. A one-benchmark headline is the biggest remaining
exposure.

Chosen deliberately over chasing h170's ~5 SE residual, which refines an account
already established at the level that matters. Recorded as a choice, with the
reason.

**Holistic:** the front has gone question -> answer -> intervention -> actionable
change in four ticks, after nineteen spent on explanations that did not survive.
The thing that turned it was not a better idea but a different kind of test:
h171 changed something instead of explaining what was already there.

**Compute:** 15/15 (h172 L=2/L=4 finishing, h173 HEAD launched).

## 2026-09-02 (tick 28) — the dose is monotone, and it points the helpful way

**h172 L=2 landed and the dose is clean**: L=8 (control) 15.82 rel% in 82.4 min,
L=2 13.97 in 22.5 min (3.66x), L=1 13.69 in 13.2 min (6.26x). **Both shortened
arms beat the control, all improving 5/5, and the ordering is monotone in
length.** P2 holds -- wall-clock scales close to proportionally (1 : 3.66 : 6.26
observed against 1 : 4 : 8 if length were the only cost; the shortfall is the
fixed per-iteration GP refit and DT training that shortening cannot touch).

P3 had anticipated that any failure would show up at the SHORT end, because the
RTG label degrades most there. Instead the short end is the best end. Recording
that the registered prediction was wrong in a way that favours the result --
which is worth flagging precisely because it is the direction that flatters me.

**A seventh demonstration that the conditioning target does not drive
performance, and the cleanest one.** Across the dose the target falls 0.9761 ->
0.6380 -> 0.4590 while regret improves 15.82 -> 13.97 -> 13.69. Every previous
demonstration compared different arms; this one moves both quantities in
opposite directions inside a single controlled manipulation.

**h173's TAIL half was launched this tick -- I had only started HEAD last tick.**
Caught by checking the arm counts rather than assuming the launch loop had
covered both. Worth noting as the kind of omission that would have produced a
half-finished 2x2 reported as a full one.

**Holistic.** The front is answered and has now paid out. What remains is
scoping, not discovery: h173 tests whether the answer holds on Hartmann, and
L=4 completes the dose. The residual in h170 (~5 SE) stays open and is recorded
as open rather than quietly dropped now that the headline is settled.

**Compute:** 14/15 (h173 x10, h172 L=4 x4... capacity re-checked before each
launch this tick).

## 2026-09-02 (tick 30) — the second benchmark supplied what the first could not

**h173 fired R2**: HEAD 12.10, TAIL 46.45, centroids 0.5632 and 0.0404 against a
control at 7.99/0.6287 and RANDOM at 65.14/0.0237. The front's answer holds on
two benchmarks and R1 does not fire.

**The part that mattered most was SC1 passing.** On Borehole, h171's TAIL
collapsed its fidelity (0.217 against 0.883), SC1 fired, and the attribution had
to be borrowed from ORACLE and DIVERSE-GOOD. On Hartmann TAIL's HF is 0.281
against a control at 0.200 -- no collapse -- so the arm supplies its own clean
attribution: a tau=0 location drawn independently of the model is sufficient to
fail WITH the fidelity mix intact.

That is the concrete argument for replication. Not "the result repeats" but "the
second benchmark closed a gap the first one's confound left open". Worth
remembering as a reason to replicate even when the first result looks solid.

**SC2 also earned its registration.** The Hartmann saturation floor (0.7531)
binds on seed 44 for HEAD, TAIL and RANDOM alike. Registered in advance
specifically so identical values could not be read as a tie; dropping the seed
leaves the ordering unchanged.

**And I held the line on what is resolvable.** HEAD vs control (12.10 vs 7.99) is
NOT distinguishable at n=4-5 given Hartmann's per-seed spread (the control alone
ranges 0.7 to 16.4). What is resolvable is HEAD vs TAIL (4x) and TAIL vs control
(6x). Written that way rather than quoting HEAD as "slightly worse".

**Holistic.** Three gates fired across the last two ticks -- h174's SC1 voided an
arm outright, h173's SC1 passed and rescued an attribution, h173's SC2 defused a
floor artefact. All three were registered before the numbers. That is the
machinery working as intended, and it is more of what the front produced this
tick than any new number.

**Compute:** 1-11/15.

## 2026-09-02 (tick 31) — a failed prediction, and the decision not to explain it

**Fixed the published report first.** It stated "only the teacher's first step
reaches the learner" without qualification, and last tick's correction showed
that is Borehole-only. The page now leads with the two-benchmark claim (a bad
first step is fatal) and records that the fifth Hartmann seed reversed the
stronger one. Publishing a claim I had already corrected internally would have
been the worst version of this.

**h175 then tested the obvious explanation and it failed.** If the strong form
works on Borehole and not Hartmann, the natural account is that the tau=0
mechanism is weaker on Hartmann. Measured tightness ratio: Borehole 0.306/0.309,
Hartmann 0.298/0.265. **Equally strong, if anything slightly stronger.** P1
predicted materially larger; observed smaller. R1 fires.

**I am recording the scope difference as unexplained rather than proposing a
seventh account.** The evidence in hand would comfortably support a story about
Hartmann's sharper optimum needing refinement that Borehole's does not. That is
precisely the shape of the five accounts that already fell on this front -- each
fitted the evidence available at the time and none survived. The honest move at
this point is to stop generating them.

**Something did replicate, and it is the more useful target.** The tau=0
account's residual is ~5 SE on BOTH benchmarks (5.0/5.0 Borehole, 5.3/4.7
Hartmann). A stable incompleteness is a better thing to chase than a
benchmark-specific gap, because it is a property of the mechanism rather than of
one function.

**Two process notes.** The estimator floor was computed BEFORE setting P2's
threshold this time (0.062 in 6D against a 0.15 threshold) -- h170's version had
the threshold below the floor. And a sed substitution silently left the benchmark
bounds at 8D; it failed loudly on a shape mismatch rather than producing wrong
numbers, which is the good failure mode, but it is the second time this session a
partial string substitution has bitten.

**Compute:** 0-1/15. All arms complete.

## 2026-09-02 (tick 32) — connecting the front's answer to what the project is actually about

**Holistic call: the front is answered and the residual is not the best next
target.** Last tick I identified the ~5 SE residual as the more stable open
question. On reflection there is something more valuable: the front's answer is
about the ROLLOUT, and the project's actual contribution is the ROI. Those had
not been connected.

**h176 tests the connection.** ROI-Q10 is the project's best Borehole result
(11.59 rel% against the control's 15.82, and remarkably consistent per-seed at
11.2-12.3). The ROI is applied once per rollout, outside the tau loop, so it
shapes the pool every step draws from -- including tau=0. If only tau=0 reaches
inference, the ROI's benefit should survive a one-step rollout, and the project's
headline would run at roughly a sixth of its 117.4-minute cost.

**I argued resolvability BEFORE running, for once.** ROI-Q10's per-seed sd is
~0.4, against the control's spread of 12.9-19.2. h172's "L=1 beats the control"
had to be withdrawn to "no worse" because the spreads overlapped; here a 2 rel%
shift would sit well outside ROI-Q10's own spread, so n=5 can actually carry the
claim. Writing that down first is the fix for the h172 mistake, applied rather
than just recorded.

**R1 named first and it is the costly one:** if ROI+L1 loses the benefit, the ROI
does not act only through tau=0, and the front's answer -- already scoped to
Borehole -- would be further limited to the no-ROI configuration.

**What I did not do:** propose a seventh account for why the strong form is
benchmark-specific. h175 killed the obvious one (the mechanism is equally strong
on both), and the evidence in hand would comfortably support a story about
Hartmann's sharper optimum. That is the shape of all five accounts that already
fell. It stays unexplained.

**Compute:** 5/15.

## 2026-09-02 (tick 33) — targeting the stable residual, not the benchmark-specific gap

**Chose the residual over the scope gap, deliberately.** h175 showed the tau=0
account's residual is ~5 SE on BOTH benchmarks while the strong form's
benchmark-specificity is unexplained. A stable incompleteness is a property of
the mechanism; a benchmark-specific gap is a property of one function. The first
is worth chasing, and I said so rather than defaulting to the more conspicuous
puzzle.

**h177 probes the second conditioning input.** h168 established the emitted
action is essentially independent of RTG (0.0074 across the full sweep, 357
probed iterations). The DT is conditioned on TWO scalars, and BTG had never been
touched. If both are inert, the inference output is a function of the state
alone -- which sharpens the tau=0 account while leaving the residual unexplained.

**SC1 ran before launch, and the early signal is recorded before the arm.** Seven
BTG probes per iteration, axis populated. At five iterations the action moves
0.0007 across the sweep -- **0.7% of its own mean**, against RTG's 8.9% at full
length. The smoke ran on Borehole where btg_now is 10.9, so the 20-36 sweep sits
entirely OUTSIDE the visited range and the action still barely moves. That is a
stronger version of the test than the real run will perform.

**Named the outcome I would not like:** if P1 holds, the residual stays
unexplained and I say so rather than reaching for a third scalar. Six accounts
have fallen on this front by reaching.

**Compute:** 10/15 (h176 x5 at ~78%, h177 x5).

## 2026-09-02 (tick 34) — an exactly-zero result, chased to an architectural cause

**h177 returned a spread of 0.0000.** The emitted action does not move at all
across a BTG sweep from 20 to 36, over 357 probed iterations, on a range well
outside the visited 26.08-30.52. The RTG axis in the same runs reproduced h168
exactly (9.0% vs 8.9%), which is what makes the zero a measurement rather than a
broken probe.

**An exactly-zero effect is a demand for explanation, not a finding to bank.** My
first move was to check whether BTG is even wired -- it is: Linear(1,H), a
LayerNorm, interleaved as one of four tokens per step. So the null needed a
different cause.

**Found it, and it is quantitative.** The scalars go raw into Linear(1->H) then
LayerNorm, and that composition saturates: LayerNorm fixes the output norm and
the direction converges as |v| grows. Over 200 random weight draws the relative
embedding change across each scalar's OPERATING RANGE is 0.4869 for RTG
(0.30-1.00) and 0.0056 for BTG (26.1-30.5) -- **87x apart**. RTG sits near the
origin where the response is live; BTG sits deep in the saturated regime because
its values happen to be ~30 rather than ~1.

That predicts what was measured: RTG 9%, BTG 0%. **A story that only explained
the zero would have been worth little; one that also predicts the non-zero is
worth more.**

**The fix is concrete**: standardise the conditioning scalars before embedding,
the treatment the state block already gets. That is the second thing this front
has produced that changes what the code should do.

**Labelled EXPLORATORY and its weakness named.** The analysis came after seeing
the zero, and it uses random weights -- a trained bias could shift the operating
point. Measuring the trained btg_embed/btg_ln response directly would settle it;
registered, not done. Six accounts have fallen on this front and I am not going
to let the seventh through on elegance.

**Compute:** 0/15. All arms complete.

## 2026-09-02 (tick 35) — settling my own explanation instead of banking it

**h177's architectural explanation was labelled EXPLORATORY because it used
random weights. h178 measures the trained modules directly.** That is the check
h177 registered and did not do, and doing it immediately rather than leaving it
on the list is the difference between an account and a finding on this front --
six have fallen here, and every one of them was plausible when written.

**SC1 failed first**, with a dtype error: the probe used `state.dtype` (float64)
against float32 modules, because `propose_mf` calls `state.float()`. The smoke
test caught it before the arm ran. Fourth time this session a pre-launch smoke has
caught something that would otherwise have appeared as silently-missing data --
the probe swallows exceptions, so this would have looked like a no-op.

**The early reading is recorded as NON-evidence, deliberately.** At 5 iterations
it matches the random-weight estimate almost exactly (0.4974 / 0.0051, ratio
98.2x against 0.4869 / 0.0056 / ~87x). That agreement is close to tautological:
after five iterations the embedding weights are still near initialisation, so
matching a random-weight calculation is what an untrained module must do. Writing
that down before the full run means the full-run numbers cannot later be read as
"confirming what the smoke already showed".

**R1 named**: if trained btg_resp turns out comparable to rtg_resp, the
saturation explanation is wrong and h177's exact zero returns to unexplained.
The measurement stands either way; only the reason is at stake.

**Holistic.** The front's substantive question is answered and has produced two
things that change what the code should do -- one-step rollouts (h172/h176) and
standardising the conditioning scalars (h177, pending h178). What is left is
verification of my own explanations rather than new ground, which is the right
phase to be in and worth saying plainly rather than manufacturing a new front.

**Compute:** 5/15.

## 2026-09-02 (tick 36) — checked a fix I had already published, and found the question I had skipped

**I proposed "standardise the conditioning scalars" as a fix in findings.md
without checking it would work.** Same over-reach pattern this session keeps
catching. Checked it at zero compute: z-scoring turns BTG's operating range
(26.1-30.5) into -2.49 to +3.16, and the relative embedding change goes from
0.0056 to **1.8817** -- 336x more responsive, and ~4x more than RTG's current raw
channel. The reason it is so large is that z-scoring spans the sign change, where
LayerNorm(Linear(1->H)(v)) flips direction.

**The distinction I nearly skipped, and it is the important one.** The fix
restores RESPONSIVENESS. It does not follow that responsiveness improves regret.
Six arms on this front have paired a COLLAPSED conditioning target with good
performance (h153 .323, h161 .327, h166 .382, h171-HEAD .410, h172 L=1 .459,
h176). If the target carries little information about what a good query is, then
wiring it in properly could actively HURT.

So h179 is registered with **three** predictions and P3 (regret degrades) written
as a live outcome rather than a formality. If P3 holds, the honest report is "the
conditioning is inert, and that is load-bearing" -- not "here is a defect and its
fix", which is how findings.md currently frames it.

**h179 is queued, not launched**, because h178 has not reported: if the trained
btg_resp turns out comparable to rtg_resp, the saturation account is refuted and
h179's premise goes with it. Launching both at once would have been faster and
would have meant running an arm whose rationale might already be dead.

**Holistic.** This tick produced no new measurement of the system -- it audited a
claim I had already published and found the unasked question behind it. That is
the phase the front is in, and it is more useful right now than another arm.

**Compute:** 5/15 (h178 only).

## 2026-09-02 (tick 37) — consolidation, and a broken patch caught by checking rather than assuming

**h179 is 30% through, so this tick went to consolidation** -- research-state.yaml
was several experiments behind (h176, h177, h178 all missing) and the published
report lacked the one result a reader would most want: that the project's best
setting survives one-step rollouts at about a fifth of the compute.

**Added a practical section to the report with both limits stated in it, not
around it.** The apparent improvement from shortening is NOT real at five seeds
(paired sd 2.72 against a mean difference of 0.78), and I wrote "better" before
checking -- that admission is on the page. And shortening COSTS CONSISTENCY: the
best setting at full length varies 0.41 across seeds against 2.79 when shortened,
which nothing in the average regret shows. If predictability matters more than
wall-clock, the long version is still the right one, and the page says so.

**My patch broke the page's structure and I caught it by checking.** The
insertion left two stray closing divs (44 open, 46 closed). Verified tag balance
before publishing rather than after, found the mismatch, located it, fixed it,
re-verified 44/44 and 6/6. Publishing a structurally broken page would have been
invisible to me and obvious to a reader.

That is the third time this session a mechanical check on my own edit has caught
something: the earlier `&mdash;` assertion failure, the h175 sed that left 8D
bounds, and now this. All three failed loudly or were caught by an explicit
check; none produced silently wrong output. The pattern worth keeping is that
every edit to a file I cannot see rendered gets a structural check before it
ships.

**Compute:** 5/15 (h179 only).

## tick 38 — two arms declined on measured premises; h180 found; h179 closed provisionally

- **h174's registered follow-up DECLINED.** Candidate cause (the `minimum_hf_fraction`
  floor, gated `tau > 0`) predicts L=1 has *less* HF; observed *more* on both
  benchmarks. Wrong in direction. Mix shift is intrinsic to shortening.
- **h180 (EXPLORATORY, no new runs).** The emitted first query is invariant to the
  teacher's *rule* and moves for teachers that change *what is averaged*.
  Replicates on Hartmann, which is the cleaner test (no saturation ceiling).
  Positive control at 9.9×. Bit-level confirmation of h177/h178.
- **Teacher-rotation arm DECLINED** on h180's measurement (teachers agree to 0.044
  against a 0.82 seed floor).
- **h179 closed: P2 → R3, provisional.** Gate was under-specified; resolved with
  the pre-existing 6.1% floor. **h181 registered, corrected before launch, and
  launched** to remove h179's standardisation-vs-benchmark confound.
- **Three of my own errors caught before they propagated:** an ad-hoc rel% formula
  that disagreed with the control's known 15.82 (and a +0.707 correlation built on
  it); counting target manipulations as rule changes; a ">6 sd" gate justification
  calibrated on 2 seeds that the 4th seed cut to 2.1 sd.

## tick 39 — centre-collapse, and the sharpest confirmation of the front's answer

- **h182 (EXPLORATORY).** The failure mode is collapse to the box centre.
  ρ(distance from centre, rel%) = **−0.967** over 28 MF-DRO Borehole arms.
- **Self-correction.** First pass called Borehole *bimodal* with a 0.70-wide gap on
  10 hand-picked arms; running all arms filled the gap. Sampling artifact, fixed.
- **Dynamic signature.** w6/w1 centre-distance ratio separates **28/28** arms
  completely (failing 0.31–0.42, working 1.20–1.39). Verified on all arms *because*
  of the bimodality error.
- **Teacher-vs-DT inversion.** HEAD's teacher averages at the centre while its DT
  sits far and works; TAIL inverts. The teacher's all-τ average is anti-predictive;
  τ=0 predicts. Direct confirmation of h180 on the dissociating pair.
- **Feedback-loop hypothesis dropped on evidence** — every teacher expands,
  including the failing arm's.
- **The benchmark asymmetry stays open.** My escape-based explanation was refused
  by its own numbers (HEAD retains 95%/90% of the control's escape yet costs
  1.07×/3.15×).
- h181 running, 38/107.

## tick 40 — collapse is causal; the failing arms do literally nothing

- **The 43.94 floor IS the initial design.** All four failing arms improve on their
  own initial design on **0/5** seeds — final best == init best exactly, 20 runs.
  Control/HEAD/ROI-L1 improve 5/5. The failing arms are not performing badly; they
  are performing *not at all*.
- **The centre is a bad region** — best f within 0.10 of it is 85.76 vs 273.00 for
  the whole box. So collapse is directly harmful.
- **Self-retraction within the tick:** I first concluded from ρ(distance, value) =
  −0.027 that the centre is *not* bad and collapse is a marker, not a cause. Wrong,
  and wrong for the caveat I had already written down and then ignored.
- **Second explanation of the benchmark asymmetry refused** — the geometric account
  needed Borehole's ρ to be large; it is 14× the wrong way. Asymmetry stays open.
- On Hartmann the direction signature *inverts* among working arms, and the best
  arms never get within 0.6 of x*.
- h181 at 55/107.

## tick 41 — h181 closes; h179's verdict withdrawn; h183/h184 open the asymmetry causally

- **h181 = P2** (registered band). Standardisation buys 1.28× on RTG, 3.84× on BTG.
  The confound it existed to remove was real: Borehole is **1.9× more responsive
  than Hartmann unstandardised**.
- **h178's 336× is module-level and stays so** — ~1.1% transfers in situ.
- **h179's R3 withdrawn.** R3 needed the channel to have been made to work; it was
  not. The conditioning counterfactual is reopened as a genuine open item.
- **h183 (EXPLORATORY)**: a *third* account of the benchmark asymmetry, and the
  first not refused by its own numbers — fit quality predicts regret on Hartmann
  (+0.53 / +0.61) and not on Borehole (−0.24 / +0.16), with `lf_fraction` 0.12 vs
  0.80 as the candidate cause. Duplicate-inflation caught and removed (four
  identical Hartmann probe runs had inflated it to +0.70 / +0.75).
  The first diagnostic scan (10 of 12 flagged) was discarded as multiple comparisons.
- **h184 registered and LAUNCHED** — the causal test h183 calls for. New
  `max_hf_fraction` HF ceiling, **identity gate PASSED exactly at full precision**
  (and `tools/identity_gate.py` added; the historical reference in findings.md is
  rounded and fails a genuinely identical build by 1.8e-11 — now fixed).
  SC1 recorded before launch: the ceiling fires, moving Borehole to ~0.71 LF.

## tick 42 — h185: the mechanism, quantified. The DT is a per-timestep constant predictor.

- **h185 (EXPLORATORY).** loss/var = 0.795, 0.916, 0.985, 1.026, 1.022 across five
  Borehole arms spanning a 2.5× teacher-variance range — **every arm sits at the
  best-constant MSE**. The DT explains 0–20.5% of its teacher's action variance and
  **none** on three of five.
- **The explained share is monotone in τ-structure**: 0.0% at rollout length 1
  (where it is 0% by construction), 0.0% at L=2, 1.5% at L=4, 8.4%/20.5% at L=8 with
  a sharp τ=0 split. That is a per-timestep constant predictor.
- **Learning more does not help**: TAIL explains the most (20.5%) and fails worst
  (43.94); ROLLOUT1 explains none and does best (13.69). Only where the constant
  lands matters.
- **"Fits better while performing worse" is located**: TAIL's loss falls 20% over
  its run while its queries contract to 31% of their starting centre-distance and
  the *teacher's* spread stays flat (1.03).
- Report rewritten around this and republished; stale lede corrected.
- h184 running (ctrl iter 6, head iter 11), lf_fraction ~0.71 as designed.

## tick 43 — h186: the conditioning is the MORE responsive input, and a wording correction

- **h186 (EXPLORATORY).** Conditioning sensitivity 0.0782 vs **state** sensitivity
  0.0122 on Borehole (6.4×; 36× after standardisation). The DT responds to its
  conditioning more than to its state.
- **Confound raised, then resolved against my own suspicion.** I expected the three
  τ=0 states to be near-identical, which would explain this trivially. Measured:
  3 distinct states, dim 68, **pairwise 0.5866 on norm 4.5032 — 13% separation.**
  Well separated. The low state sensitivity is real.
- **Wording correction:** "the τ=0 states are near-degenerate", repeated throughout
  findings.md, conflates low *variety* (3 distinct among 60 — true) with the 3 being
  similar to each other (false). Mechanism survives, sharpened: the constant output
  is a property of the *learned solution*, not of information-free inputs.
- **Framing correction:** the conditioning is not a uniquely broken input. Combined
  with h181 (~1.1% of the fix transfers), "a defect with a fix" is the wrong
  diagnosis.
- h184 running: ctrl iter ~41, head ~69, lf_fraction ≈ 0.74 on both as designed.

## tick 44 — the synthesis, and the arm it calls for

- **SYNTHESIS**: the DT is an **averager of its teacher's first move**. h185
  (constant predictor) + h186 (ignores well-separated inputs) + h31 (that averager is
  competitive with the teacher, 7/10 seeds) compose into one account. The founding
  result follows: a perfect teacher's routes *end* at the optimum but *start*
  arbitrarily, so the mean of their first moves is the box centre.
- **Checked before building**: h31 and h55 already targeted "does the DT add
  anything". h31 has 10 seeds and was read rather than rerun.
- **h187 registered** (protocol + SC1 committed before launch) to fix h31's three
  weaknesses — Hartmann, off-metric, fidelity-confounded. Not yet launched: it waits
  for h184's slots so the registered priority is not slowed.
- **The smoke test earned its keep**: h31's code no longer runs — `candidate_features`
  is `None` unless the forbidden `use_candidate_scoring` flag is on. Caught before
  launch, deviation recorded in the protocol.
- h184 still running (ctrl 83, head 137).

## tick 45 — h188 predicts the query; h187 launched into freed slots

- **h188 (EXPLORATORY).** The synthesis now *predicts* the DT's emitted query from an
  independently recorded quantity. `teacher_action_stats` holds the all-τ mean, which
  supplies the control: at L=1 that **is** the τ=0 mean (must hold), at L=8 with a
  sharp τ split it is **not** (must fail).
  Positive: ROLLOUT1/2/4 → **0.109–0.158**. Negative: HEAD/TAIL → **0.617–0.715**.
  **4.5–5.3× separation**, direction fixed in advance by rollout length.
- **h187 LAUNCHED** into the slots h184's HEAD arms freed. 10 workers total, never
  above the cap, and the registered priority was not slowed.
- **h184 HEAD arms complete (5/5).** SC read only: `lf_fraction` = **0.750 on every
  seed** against Borehole's unforced 0.117 — the ceiling holds its 25% HF target
  tightly. **Regret deliberately not read**: h184's readout is the GAP between arms,
  so reading one arm alone would be a partial peek.
- h184 CTRL still running.

## tick 46 — h184 lands: P1 fires in direction, not in magnitude

- **h184 complete, 10/10.** SC PASS — `lf_fraction` exactly 0.75 on every seed of both
  arms, symmetric forcing, gap unconfounded.
- **Registered statistic 4/5 → P1.** Sign pattern moved 2/5 → 4/5 toward Hartmann's 5/5.
- **But the magnitude did not move**: paired mean +1.15 → +1.60 (+0.45 = 0.33 se),
  against Hartmann's +17.17 — **9% of the way**. Supported in direction only.
- **Gate-design lesson recorded**: a sign-pattern gate can fire on an effect a tenth
  the size of the one it models. Register a magnitude clause alongside.
- **The asymmetry is NOT closed.** This is the first of three accounts not refuted, and
  it is partial.
- h187 still running (5 workers).

## tick 47 — h187: the DT is a net negative. The result the protocol named as worst.

- **h187 P2 fires.** Borehole, frozen metric: **teacher-only 12.97 vs MF-DRO 15.82**,
  paired **−2.85** (se 1.30), **teacher better on 5/5**, against a pre-registered
  threshold of 1.72. Removing the Decision Transformer entirely improves the metric.
- **Fidelity objection ruled out by an arm already run.** h184's LF-forced MF-DRO:
  a **6.4× change in LF share moves it 0.06 rel% points** (15.82 → 15.76). The
  teacher's advantage is not about where it spends its budget.
- **Synthesis leg 3 RETRACTED.** "The averaging is about as good as running the
  teacher" (from h31: Hartmann, off-metric, unmatched fidelity, 7/10, unresolved) is
  false on Borehole under the frozen metric. Corrected: the averaging is a **cost**.
- **Protocol self-correction**: it predicted `lf_fraction ≈ 0` from SC1's first six
  iterations; realised value is **0.435**.
- **Report corrected and republished** — it had carried the false claim for one tick.
- **h189 registered and launched**: h187's worker reused *unchanged* on Hartmann, to
  settle whether P2 generalises or is Borehole-specific. Until then the claim is
  scoped to Borehole. Results land in h187's directory (documented in h189's protocol,
  left as-is to preserve the identical-code guarantee).

## tick 48 — h187 scoped: the ROI configurations flip the sign, but not like-for-like

- **Scoping correction, made within a tick of publishing.** h187's P2 is about the
  **default** path. Every ROI-equipped MF-DRO configuration beats the same teacher-only
  arm — ROI-L1 by −3.15 on 4/5, near a mirror of the default's +2.85 on 0/5.
- **But it is not like-for-like**: ROI is a *training-time* mechanism (`mf_dro.py:1184`)
  and the teacher-only arm does not learn, so it never receives ROI. The ROI rows give
  MF-DRO a mechanism the teacher lacks.
- **A teacher+ROI arm was considered and NOT built**: it would mean applying ROI's
  region to the teacher's *inference* pool, which does not exist today. Recorded as
  untested rather than asserted either way.
- Report and findings both corrected and republished; both halves now travel together.
- h189 (Hartmann no-DT) still running.

## tick 49 — h189 in flight; two corrections recorded before its readout

- **SC observation recorded BEFORE h189's regret**: the teacher's `lf_fraction` on
  Hartmann spans **0.000, 0.368, 0.922, 0.931, 0.981** across five seeds — the same
  acquisition rule going almost pure-HF on two seeds and almost pure-LF on three.
  On Borehole it stayed in 0.291–0.561. **MES's fidelity criterion is bistable on
  Hartmann.** Consequence stated in advance: a P1 verdict must be read as *undecided*,
  not as equivalence.
- **Cost-ratio correction.** Hartmann is **8:1** (HF 8.0, LF 1.0); Borehole is **2:1**
  (HF 2.0, LF 1.0). I had assumed the reverse. This **grounds h183**: the
  `lf_fraction` gap (0.80 vs 0.12) it rests on is a direct consequence of a 4×
  difference in cost ratio, not a free parameter. It also reframes h184 — forcing
  Borehole to 75% LF pushed it *away* from its cost-rational allocation, a real
  handicap, and the score still barely moved (15.82 → 15.76).
- h189 at 2/5. The LF-heavy seeds need ~200 queries at cost 1 to exhaust the budget,
  so they run much longer than the HF-heavy ones (9–13 min vs ongoing).

## tick 50 — h189: the sign flips. h187 is Borehole-specific; h31 vindicated.

- **h189 P3, 5/5.** Hartmann: teacher-only 21.88 vs MF-DRO **7.99**, paired **+13.89**
  (se 4.53), MF-DRO better on **5/5**. Borehole was the exact opposite (−2.85, teacher
  better 5/5). **Two benchmarks, opposite answers, both unanimous**, and Hartmann's
  advantage is ~5× Borehole's deficit.
- **h31 vindicated, not explained away.** My registered P2 branch would have blamed its
  metric; P3 fired and h31's direction was right.
- **Retractions:** "the DT is a net negative" must never be stated unqualified — h187
  is Borehole-specific. And the synthesis's third leg is wrong in *both* directions:
  the averaging is a cost on Borehole and a large gain on Hartmann.
- **Corrected synthesis:** the DT reproduces the mean of its teacher's first move, and
  the *value* of that averaging is benchmark-dependent and flips sign. h185/h186/h188/
  h182 untouched — they describe what it does, not what it is worth.
- **SC recorded pre-readout was confirmed**: the teacher's fidelity is bistable on
  Hartmann (0.000–0.989) while MF-DRO's is stable at 0.800. Candidate account (cost
  ratio 8:1 vs 2:1 making stability valuable) recorded as **not established** — ρ=+0.700
  over five points with two collinear variables is a description, not evidence.
- Report corrected and republished; the headline now carries three parts.

## tick 51 — h190 registered, then declined on two pre-launch checks

- **h190 DECLINED before launch.** (1) The smoke showed `max_hf_fraction` is a *ceiling*
  — one-sided, so the arm would only half-apply. My registered SC band was mis-specified
  and the smoke caught it. (2) h189's own per-seed data undercuts the account: the
  teacher scores **14.02–16.41 across allocations 0.000–0.934** and **MF-DRO beats it at
  every one**. It is not losing because its allocation is unstable.
- **The Hartmann sign flip now has no established mechanism**, and the stability account
  is undercut rather than merely unproven.
- **Stopping the mechanism-proposing.** Four proposed across the two open questions;
  three refused by direct measurement, one undercut. No fifth until better evidence.
- Third arm declined on a measured premise. Checks cost minutes; arms cost hours.

## tick 52 — the core mechanism generalises. A false limit I published twice is corrected.

- **Holistic step-back, machine idle.** With no arm to run and mechanism-proposing
  paused, the highest-value question was the mechanism's *generality*, not its truth.
- **A limit I published TWICE did not exist.** I recorded that `teacher_action_stats`
  was on "h171/h172 only", making h185 Borehole-only. It is on **18 arms across both
  benchmarks**. The generality test was available the whole time.
- **h185 GENERALISES.** `loss/var` across **10 arms on two benchmarks** spans
  **0.750–1.054** — every arm at the best-constant value. Not a Borehole artifact.
- **A second by-construction control appeared undesigned.** The theory forces 0%
  variance-explained in *two* distinct cases: one timestep (3 L=1 arms confirm), and a
  teacher whose action distribution is identical at every step, since then all per-τ
  means coincide. **PROBE-RANDOM (random teacher, L=8) reads 0.0%** — a different route
  to the same forced answer.
- **The rule, with all ten arms fitting:** variance is explained **if and only if** the
  teacher's action distribution differs across timesteps.
- Report updated and republished.

## tick 53 — one false limit had blocked three tests. All three now general.

- **The limit corrected in tick 52 was cited in h188 and h182's inversion too.** Three
  analyses were Borehole-only for a reason that did not exist.
- **h188 replicates on Hartmann**: "must hold" arms at 0.0422 / 0.1208, "must fail" at
  0.3879 / 0.4848, seed-noise scale 0.3624 — 3.2–11.5× separation, groups fixed by
  construction.
- **PROBE-RANDOM is a third by-construction case, not an anomaly.** A τ-invariant
  teacher's all-τ mean *is* its τ=0 mean, so the prediction is forced to hold at L=8.
  **The same pair of cases h185 found independently on a different quantity** — and
  designed into neither.
- **h182's inversion replicates on Hartmann** with near-identical numbers (HEAD's
  teacher ≈0.08 on both benchmarks; TAIL's DT contracts to ≈0.09 on both).
- **Mechanism now general**: h185, h186, h188, h182-inversion all hold on both
  benchmarks. Benchmark-specific: the *value* of the DT (h187/h189 sign flip) and the
  direction signature.

## tick 54 — centre-collapse replicates on Hartmann, and is DT-specific

- **Audited an under-tested headline.** h182's ρ = −0.967 came from 28 Borehole arms but
  only 6 hand-picked Hartmann ones. Ran the full Hartmann sweep.
- **Replicates: ρ = −0.841 over 17 MF-DRO Hartmann arms.**
- **The non-DT arms don't collapse at all** (0.565–0.893, no relationship, ρ = +0.600 at
  n=5). The best of them sits *nearer* the centre than the worst. **Centre-collapse is a
  property of the DT-based policy, not of the problem** — which is what the mechanism
  predicts, since a GP method has no constant to collapse onto.
- **Applied the earlier lesson**: Hartmann shows an apparent gap, and I recorded it as an
  observation rather than a claim, because Borehole's apparent gap dissolved when all 28
  arms were included.
- SF-DRO's classification flagged as arguable, and left in the group that makes the split
  *weaker* rather than stronger.

## tick 55 — the paper backbone is now guarded; the guard's first version was broken

- **Audited the primary record.** `check_report.py` guarded the published HTML against
  stale retracted claims; **nothing guarded findings.md**, and check_report's own claim
  list was **9 retractions out of date** (h135-era only).
- **Built `tools/check_findings.py`**, seeded with this run's retractions. **findings.md
  is clean** — 8 claims, none surviving as live assertions. Corrections landed properly.
  check_report.py's list refreshed to 9; report clean too.
- **The guard's first version passed a deliberately planted assertion.** Loose substring
  matching (`"correct"`, `"wrong"`) matched ordinary prose and stripped **23% of the
  file**, taking the planted claim with it. Fixed to word-boundary retraction
  announcements on the heading line only; stripping fell to 5%.
- **Re-tested both directions**: planted assertion → exit 1 with the line number; the
  same phrase quoted inside a correction → exit 0. A guard that cannot fail is worthless,
  and only the negative test found that this one couldn't.

## tick 56 — the paper backbone's front door was 6 days and ~14,000 lines stale

- **findings.md's "read this before anything below" block** was dated 2026-08-27,
  claimed "4200+ lines" against an actual 18,127, and covered only Phase 1 (ROI, north
  star, benchmark deficits). It said **nothing about this entire run** — no mechanism,
  no sign flip, no centre-collapse, none of the retractions.
- **Rewritten as a two-phase front door.** Phase 2 (this front, ANSWERED) now leads:
  the mechanism with its numbers, its cross-benchmark generality, the centre-collapse
  signature, what the DT is worth (flips sign, 5/5 each way), the actionable L=1 result,
  what is still open with the instruction not to propose a fifth mechanism, an explicit
  **do-not-re-derive list of the 6 retractions**, and the 3 arms declined on measured
  premises. Phase 1 preserved verbatim below, relabelled.
- **Verified after the edit**: guard still clean (the retracted phrases in the
  do-not-re-derive list are quoted, so they read as citations — intended), and the
  negative test still fires.
- A human can now write the abstract from the first 85 lines instead of 18,127.

## tick 57 — the mechanism explains the project's WINS, not just its failures

- **h191 (EXPLORATORY).** Every prior application of the mechanism explained a failure.
  This tested whether it explains the surviving interventions.
- **Clean L=1 test**: ROI moves the teacher's τ=0 mean **+0.108** outward and tightens
  its variance 32%; **the DT follows by +0.075**, ~70% of the shift, and regret improves
  13.69 → 10.81.
- **Covers all three interventions that work**: control 0.852/15.82, L1-LOSS 0.890/13.47,
  REFINE-100 0.928/9.96, ROI-L1 0.981/9.81 — all move the constant outward, monotone in
  performance, none that helps moves it inward.
- **So the ROI, teacher refinement and the L1 loss are one mechanism, not three**:
  relocating the point the DT memorises. None makes the DT smarter.
- **Overlap reported, not hidden**: the ROI vs non-ROI split is not clean (0.053 overlap),
  and the two non-ROI arms inside the ROI range are precisely the project's other
  successful interventions — which supports the account.

## tick 58 — h192: the causal test. First arm that can FALSIFY the mechanism.

- **h191's limits section named the missing experiment**: nothing had ever moved the
  teacher's τ=0 mean directly and checked the DT follows. Everything supporting the
  mechanism (h185, h188, h182, h191) is **correlational**.
- **h192 built and launched.** At τ=0 only, the teacher's chosen action is translated
  halfway toward the box centre; later steps untouched. Run at rollout_length=1 so the
  recorded all-τ mean **is** the τ=0 mean and the shift is directly measurable.
- **Gate on the TRANSFER RATIO** (DT's shift ÷ teacher's imposed shift): P1 ≥0.50 holds,
  P2 0.15–0.50 partial, **P3 <0.15 FALSIFIES the mechanism**. h191 measured ≈0.70 for
  ROI's naturally-occurring shift.
- **P3 would invalidate h185/h188/h191, findings.md's Phase 2 header, and the published
  report's core section.** Named in the protocol before launch.
- **Identity gate PASSED exactly** (122.29066752728207) on the patched core.
  **SC PASSED before launch**: teacher τ=0 mean 0.7788 → 0.3397.
- Readout committed before any run finished. 5 workers.

## tick 59 — h192 lands. The mechanism is INTERVENTIONAL. Transfer ratio 1.094.

- **P1, decisively.** The first arm capable of falsifying the mechanism confirmed it.
  Teacher τ=0 mean moved **0.7837 → 0.3645** (imposed +0.4192); the DT's own query
  centroid moved **0.8546 → 0.3961** (observed +0.4585). **Transfer ratio 1.094 —
  essentially one-for-one**, against a gate requiring ≥0.50.
- **Every prior result on this front was correlational.** h185/h188/h182/h191 all
  *observed* the tracking; h192 *moved* the mean and the DT moved with it.
- **Secondary fired too**: rel% **13.69 → 43.18** (+29.49), and improves-on-its-own-
  initial-design **5/5 → 1/5** (final best 267.20 → 175.91 against an init best of
  173.54). 43.18 sits just under the 43.94 h182 identified as *being* the initial design.
- **Confound named, on the secondary only**: the shift was aimed at the box centre, which
  h182 had already called bad, so the *regret* half is a joint test of two claims. A
  neutral-direction shift of the same magnitude would separate them — the obvious next
  test. The transfer ratio is unaffected.
- findings.md's Phase 2 header updated; report republished; guards re-run clean.

## tick 60 — h193 abandoned before launch, per its own protocol. Zero runs.

- **Four designs, all rejected by their own SC.** The protocol committed in advance to
  abandoning rather than iterating further if the SC failed on Hartmann too. It did.
- **Borehole: geometrically forbidden.** 80.9% of the control's real HF queries sit within
  0.05 of a box wall; dim 0 at mean |coord| 0.465 vs a half-width of 0.5. The good region
  *is* the boundary, so **the only direction with room to move the constant is inward** —
  h192's confound may be unavoidable there.
- **Hartmann: geometry fine, intervention uncontrollable.** Smoke gave displacements of
  0.1757 (centre) vs 0.2934 (rotation) — a 67% mismatch — and the rotation *raised* the
  centre-distance. **Not fixable by a better plane**: the shift changes the run, so the
  teacher re-decides from a different model and produces a different action distribution.
  Realised displacement is an **outcome**, not an input.
- **h192's primary survives untouched** (transfer ratio 1.094 is direction-independent).
  **Its secondary stays confounded** by this route permanently, and h182 keeps only its
  correlational support.
- No compute spent on runs. The `tangent` path stays, default-off and identity-gated.

## tick 61 — PAUSED by the user. Consolidation only, no arms launched.

The user paused autoresearch; the loop fired on its own. No arms launched, no compute
spent. Used the tick for a gap their question exposed.

- **Their sliding-window proposal was already built and tested as h27**, and
  `findings.md` had **zero** mentions of it. The Phase 1 result and the Phase 2
  mechanism were never connected in the record.
- **h27**: K=1 vs K=8 gave **bit-identical proposals** (max |Δx| = 0.000e+00), identical
  fidelity, identical final regret. Not a wiring failure — real history moves the
  coefficient vector **11.2%**, ~85× ordinary state variation, and reordered **0/12**.
- **Recorded the unification**: Phase 1 found three independent nulls — state
  (H5/H21/H22), RTG/BTG (H8/H26), history window (H27). Phase 2's mechanism explains all
  three at once: `loss/var` ∈ [0.750, 1.054] means the DT is at the best-constant
  solution, and **a constant does not depend on its inputs**. Phase 2 header updated.
- **The binding quantity** is whether the teacher's action is predictable from the state,
  not how much input the DT receives — which is why the interventions that work (h191)
  move the constant rather than widening the input.
- **Not ruled out and left unbuilt**: training the DT on real trajectories so its
  *training* distribution matches inference-time context. Offered to the user; awaiting
  their call rather than launching it under a pause.

## tick 62 — still paused. Tested an alternative to my own central claim.

No arms launched, no compute.

- **Challenged my own assertion.** I have repeatedly said the DT sits at the best-constant
  solution *because its target is unpredictable from the state*. The competing explanation
  — that it simply **underfits** — had been assumed away, not tested.
- **h185's data separates them.** Arms with **no** τ-structure available (L=1, or a
  τ-invariant random teacher): **4 arms, 0.0% variance explained**. Arms **with** it:
  **6 arms, 6.6–25.0%**. An underfitting model would miss both. **Underfitting is not
  supported** — the DT fits the learnable part.
- **Precision correction to my own wording.** The between-τ component rides on the
  *position index*, not the state. So what is measured is **"the DT does not use its
  state"** — matching h186 — not **"the state is uninformative"**. I had been sliding
  between the two.
- **The test that would settle it** needs an independent predictor fitted from saved
  `(state, action)` pairs. Those are **not serialised** (`teacher_action_stats` keeps only
  mean and variance). Adding them is cheap and purely additive — the highest-value
  addition to any future run, flagged for when the pause lifts.

## tick 63 — the user overturned a stale null. h194 Stage 0 G-PASS.

- **The user was right and I was too confident.** I had answered their sliding-window
  proposal with "already tested — h27". **33 commits** have touched the DT/policy since
  h27, three behaviour-changing, including the Aug 27 fix whose message says the ROI
  candidate-pool bug "confounded every ROI A/B". Correction recorded.
- **Reading the code changed the hypothesis twice.** `decisionTransformer.py:78` shows the
  window already supplies in-distribution positions 0…T−1 with readout at T−1 — so my
  guess that it "fixes the timestep problem" was wrong, h27 already did that. But that
  made h27's null **contradict** h185's 13–25% between-τ variance, which became a sharper
  gate than the `frac_seq` statistic I first registered.
- **Stage 0 G-PASS.** 7/8 iterations differ, mean L2 **0.2496** in the unit box —
  **5.7×** a full teacher-rule swap. **Control check exactly 0.0000** at iteration 0
  (ctx=1 both), so it is the window and not RNG.
- **h27 overturned; h185 vindicated.**
- **Mechanism now predicts Stage 1 FAILS (P3)**, because the window emits a late-τ
  constant and h171/h173 showed τ=0 is the step that matters. Recorded before running.
- Stage 1 (10 worker-hours) not launched — loop paused.

## tick 65 — h194 Stage 1a in flight; state recorded; readout extended to answer the ask

- **Stage 1a running**: WINDOW-K8 at ~116/240, contemporaneous CTRL-K1 at ~73/240, 10
  workers. Neither arm readable yet. **Stage 1b (the expert teacher) has NOT run** — it is
  gated on 1a not returning P3, because the window is the shared component of both arms
  and a harmful shared component makes the combination unattributable.
- **Readout extended before results** to report against **default MF-DRO (15.82)** as well
  as CTRL-K1, since that is what the human asked. **Gate unchanged** — only WINDOW − CTRL
  isolates the window, and ROI alone already reaches 11.59, so beating 15.82 would say
  nothing about the window.
- **`research-state.yaml` updated** with h194's full status and the monitoring-hazard fix.

## 2026-09-03 — the audit tick: two defects found, one of them mine

Not an arm-running tick. Three things were found by READING rather than running,
and all three change what earlier results mean.

**h195/h196 — the sliding window was fed zeroed actions.** The human rejected my
claim that the window question was "already tested (h27)" and asked for an audit
against `papers/DT.pdf`. The audit found the history slots carried no actions.
h196 fixed it: **-2.61 rel% (se 0.93), better on 5/5**, recovering **52%** of the
window's deficit (+4.99 -> +2.37). The window still hurts, so the direction of
h27's null survives; its magnitude was half artifact.

**My own real-query `b` was computed over the wrong candidate set.** Uniform
200-point pool at inference vs the ROI-filtered 600-point pool in training. b
looked flat and non-monotone (net +2.7%); corrected it falls **-27.9%**. I was one
step from concluding "the information signal isn't there," which would have been
an artifact of my own pool choice. `build_roi_pool()` now shares the rule.

**THE_ANSWER was overstated, and the human found it.** Its closing sentence -- "a
perfect teacher's first move is a random start point" -- is true of h145's oracle
*by construction* (uniform start), not of good teachers. The human's framing: an
expert teacher makes the OPTIMAL DECISION AT EVERY STEP, not one that merely
arrives somewhere good.

This SHARPENS the mechanism rather than weakening it. If the DT emits its
teacher's tau=0 mean, teacher quality helps iff it improves that mean. Re-reading
the failures: h145's oracle destroyed tau=0 (uniform -> box centre); h152's beam
has greedy as its elite first child, so its tau=0 barely moved. **Neither arm ever
varied the decisive quantity.** That is a gap, not a null.

**h198** (running) is the first teacher here that optimises the TASK: expected
terminal best HF value under a greedy base policy, GP-only, no oracle access --
so it is a deployable method, unlike h145. Both label variants run (human: "build
both and run both"), differing by one flag.

Failures and near-misses this tick, all reported:
- SC1 "failed" 4/6 -> the SC's own RNG bug (`compute_joint_mf_mes` Thompson-samples
  and consumes RNG). Fixed, then 6/6.
- The lookahead's base rollout was cost-bounded, letting an all-LF future run 14
  steps inside an 8-step rollout. Now step-bounded.
- Cost was 16.7 h/seed. Two principled fixes -> 2.1 h/seed.
- A diagnostic hook referenced `teacher_action_stats`, not a parameter of
  `simulate_mf_trajectory`. Died on NameError in 90 s; removed.
- The intermediate readout put h197 on an unshifted cost axis (in-flight ckpts
  have no `is_init`) while CTRL was shifted by -40. It printed "+11.77 worse on
  5/5"; corrected, "-6.79 better on 4/5". Same class as the h196 partial-ckpt bug:
  the readout ran fine and printed a confident, wrong answer.

Reflection: this is deepening, not arm-running. The h198 reframe re-reads ~10
arms' worth of nulls as never having tested the decisive quantity, and two of the
three findings above were errors in MY OWN instruments. The risk to name: h198 is
expensive (5.5 h/seed x 10) and justified by a mechanism argument. If it nulls,
the protocol says that is P2 -- necessity survives, sufficiency refuted -- which
is still informative, and that was written down BEFORE launch.
