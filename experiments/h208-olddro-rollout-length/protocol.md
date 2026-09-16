# h208 -- does rollout length matter in the ORIGINAL SF-DRO (papers/Old_dro.py)?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY (h172 transplanted to
the original single-fidelity implementation and its original setting).

## Question

h171/h172 showed on MF-DRO that only the teacher's FIRST rollout step reaches
the real query, so rollouts can be cut to length 1 at no cost in regret. Is that
a property of MF-DRO's later engineering, or was it already true of the ORIGINAL
DRO design in `papers/Old_dro.py` (the file behind the DRO paper's experiments)?

The original inference stage (`_propose_next_candidate`, Old_dro.py:1023-1044)
feeds the DT a length-1 sequence: the real state, a zero dummy action,
`target_rtg = 1.0` and `timestep = 0`. Training (`_train_decision_transformer`)
supervises every position 0..L-1 of each rollout. Positions >= 1 therefore reach
the real query ONLY through shared weights -- they are never read at inference.
If that is what "the original does not use rollout data beyond step 1" means, a
rollout-length dose must be flat in regret.

## What is run -- and what is deliberately NOT changed

`papers/Old_dro.py` is imported UNMODIFIED (the file's own header forbids edits;
the worker subclasses/wraps it). It runs against the CURRENT `src/` -- in
particular the current `src/model/decisionTransformer.py` (causal mask, RTG-first
ordering, state-token readout from commit 7bcc3b8) rather than the bidirectional
action-token-readout DT the paper ran. That is what executing Old_dro.py today
does; it is stated here so the result is not misread as a paper reproduction.
The original DT's action-token readout under bidirectional attention saw its own
target action during training, so on that model a flat dose could be flat for
the wrong reason (nothing learned at all). The current readout cannot copy its
target, so a flat dose here is informative.

Original setting, from `config/method/dro.yaml` + `config/test_function/Ackley.yaml`
+ the paper's Sec 5.1: Ackley 10D on [-32.768, 32.768]^10, input shift 10
(optimum at x = 10*1, f* = 0 after negation), observation noise sd 0.01,
5 Sobol initial points, 500 real iterations (505 evaluations), maximise.
DRO config verbatim: 5-GP RBF ensemble (lengthscales 0.1..10), rotate_acq
teacher over {ei, ucb, pi, mes}, UCB>=max(LCB) ROI with kappa 6, 10 rollouts per
iteration, DT 128x4x4 with dropout 0.1, lr 1e-4, 100 epochs per iteration,
max_seq_length 20. `max_rollout_length` is the config default 4.

## The dose, and why early-stop is switched off for it

Smoke run at the config default (`early_stop: true`, threshold 1e-4,
`max_rollout_length = 4`): realised lengths over 30 rollouts were
{1: 6, 2: 15, 3: 6, 4: 3}, mean 2.2. The Bayesian early stop truncates a rollout
the first time a simulated step fails to improve, so under the original flag the
nominal dose is NOT realised -- L=4 and L=8 would differ by a handful of steps.
h172's SC ("the parameter reaches the rollout, every trajectory at exactly T")
would fail. So the dose arms run with `early_stop: false`; smoke at L=8 gave
30/30 trajectories of length exactly 8.

Arms, all `papers/Old_dro.py`, Ackley 10D, seeds 42-51 (10, matching the paper's
10 trials):

| arm | max_rollout_length | early_stop | DT trained on | rtg[0] spans |
|---|---|---|---|---|
| ESON-L4 | 4 | true (original flag) | all realised positions | realised steps |
| L1 | 1 | false | position 0 | 1 step |
| L2 | 2 | false | positions 0-1 | 2 steps |
| L4 | 4 | false | positions 0-3 | 4 steps |
| L8 | 8 | false | positions 0-7 | 8 steps |
| L8-TRUNC | 8 | false | position 0 ONLY | 8 steps |

L8-TRUNC is the arm that separates the two things a shorter rollout changes at
once (h172 flagged this and could not separate them): the later-step TRAINING
DATA and the HORIZON of the position-0 RTG label. It simulates the same 8-step
rollouts as L8 and keeps rtg[0] = sum of all 8 rewards, but truncates every
trajectory to its first (state, action) before training. L8 vs L8-TRUNC isolates
positions 1-7 as training data; L8-TRUNC vs L1 isolates the label horizon.
Implemented by wrapping `_train_decision_transformer` in a subclass; Old_dro.py
itself is untouched.

ESON-L4 is the literal original configuration and anchors the dose to it.

Within a seed, all six arms share the Sobol initial design (seed set in
BaseBayesianOptimizer.__init__ before sampling), so seed-paired differences are
paired on the initial data; RNG streams diverge from the first rollout on.

## Metric

Final simple regret at 505 evaluations: f* - f_true(x+), where x+ is the
argmax of OBSERVED y and f_true is the noiseless Ackley (so the 0.01 noise cannot
flatter or penalise a run), f* = 0. Also recorded: observed-y regret, the full
per-iteration curve of both (DIAGNOSTIC only, per the standing instruction --
intermediate regret is never quoted as quality), realised rollout lengths, DT
final training loss, rtg[0] label statistics, wall-clock.
Not the MF frozen metric (sr_curve/grid): this is the single-fidelity,
iteration-budgeted original setting; there is no cost axis.

## Predictions, registered

P1 (the claim under test): the dose is FLAT. Paired over seeds,
   |mean(L8 - L1)| < 2 SE_paired and |mean(L8 - L8TRUNC)| < 2 SE_paired.
P2 wall-clock rises with L in the ES-OFF arms, roughly proportionally minus the
   fixed per-iteration cost (GP fit, DT epochs); L8-TRUNC costs the same as L8
   (it simulates the same rollouts).
P3 ESON-L4 lands inside the spread of the ES-OFF arms.

## Readings, registered

NULL (supports "the original does not use rollout data beyond step 1"):
   P1 holds. Later-step training data does not move the final regret.
DATA-MATTERS (refutes it): L8 better than L1 by > 2 SE_paired AND L8 better than
   L8-TRUNC by > 2 SE_paired. Positions 1-7 as training data improve the query.
LABEL-ONLY: L8 differs from L1 by > 2 SE_paired but L8 ~ L8-TRUNC. Then it is
   the RTG label's horizon, not the later-step data, that matters -- still
   consistent with "the data beyond step 1 is unused", and h172's own asymmetry
   resolved in the direction it anticipated.
Any other pattern (e.g. L8-TRUNC differs from BOTH) is reported as is.

n = 10 seeds. Effect sizes (paired d) are reported alongside the SE test; no
p-values are claimed at this n. Per-seed values are reported for every arm.

## Sanity checks before the numbers are read

SC1 every run has exactly 505 evaluations. BaseBayesianOptimizer.run_optimization
    catches per-iteration exceptions and SKIPS the iteration, so a run can end
    short without any error in the result. Any run with != 505 is excluded and
    reported.
SC2 realised rollout length: ES-OFF arms must be exactly L on every trajectory
    (min = max = L); ESON-L4 is reported (expected mean ~2).
SC3 L8-TRUNC's DT batches are length 1 (verified by the wrapper's own assertion),
    and its rtg[0] distribution matches L8's (same simulator, same horizon).
SC4 identical initial 5 (x, y) across the six arms of a seed.

## Compute

60 runs; measured 1.0-4.4 s/iteration -> 10-40 min each, ~25 core-hours.
Launched longest-first (L8, L8-TRUNC, L4, ESON-L4, L2, L1) through a 13-wide
queue so that with h207's one running process the machine stays at <= 14/15.
