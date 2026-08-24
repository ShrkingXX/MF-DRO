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
