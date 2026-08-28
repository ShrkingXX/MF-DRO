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
