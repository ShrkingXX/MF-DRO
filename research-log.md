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
