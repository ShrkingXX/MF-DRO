# h208 — NULL (registered P1 holds): rollout length does not move the original SF-DRO's final regret

CONFIRMATORY. 60/60 runs, 0 failures, 10 seeds, all four sanity checks pass.
`papers/Old_dro.py` unmodified, Ackley 10D, 505 evaluations, original config.

## The registered contrasts

Final simple regret = noiseless Ackley at the observed incumbent after 505
evaluations (f* = 0). Paired over seeds 42–51 (shared Sobol init within a seed).

| contrast | mean | se_paired | d | first arm better | inside 2 se |
|---|---|---|---|---|---|
| **L8 − L1** | **−0.007** | 0.274 | −0.01 | 5/10 | **yes** |
| **L8 − L8-TRUNC** | **+0.202** | 0.278 | +0.23 | 3/10 | **yes** |
| L8-TRUNC − L1 | −0.209 | 0.275 | −0.24 | 6/10 | yes |
| L4 − L1 | +0.278 | 0.197 | +0.45 | 3/10 | yes |
| L2 − L1 | +0.642 | 0.216 | +0.94 | 3/10 | no |
| ESON-L4 − L4 | −1.052 | 0.188 | −1.77 | 9/10 | no |
| ESON-L4 − L1 | −0.774 | 0.256 | −0.96 | 8/10 | no |

**Verdict by the registered rule: NULL.** Both decisive contrasts are inside
2 se. Eight-step rollouts end at the same regret as one-step rollouts
(−0.007, better on exactly half the seeds — as flat as a dose can be), and
training the DT on positions 1–7 of those eight-step rollouts does nothing that
training on position 0 alone does not (L8 ≈ L8-TRUNC, and if anything TRUNC is
ahead). The label-horizon confound h172 could not resolve is resolved here too:
L8-TRUNC ≈ L1, so the horizon of `rtg[0]` is also inert.

| arm | n | mean | sd | median | wall | per-seed 42..51 |
|---|---|---|---|---|---|---|
| ESON-L4 | 10 | **2.316** | 0.485 | 2.317 | 36 min | 2.69 2.43 2.68 2.20 1.94 1.60 2.93 1.67 2.90 2.12 |
| L1 | 10 | 3.090 | 0.491 | 3.035 | 35 min | 2.81 2.32 3.61 2.91 3.09 3.04 2.58 3.63 3.03 3.89 |
| L2 | 10 | 3.732 | 0.653 | 3.649 | 51 min | 2.69 3.29 4.81 4.05 3.63 3.02 4.54 4.06 3.56 3.67 |
| L4 | 10 | 3.367 | 0.508 | 3.481 | 82 min | 3.40 2.24 3.76 3.56 3.91 3.26 3.56 2.92 3.89 3.17 |
| L8 | 10 | 3.083 | 0.782 | 2.961 | 136 min | 4.02 2.82 4.45 2.08 2.96 2.96 2.96 3.75 2.77 2.05 |
| L8-TRUNC | 10 | 2.881 | 0.715 | 2.822 | 120 min | 3.71 3.75 2.89 2.58 2.49 1.61 2.76 3.72 2.17 3.15 |

**P2 holds.** Wall-clock 35 → 51 → 82 → 136 min for L = 1 → 2 → 4 → 8: a 3.9×
cost for zero regret. L8-TRUNC is 12% cheaper than L8 (its DT trains on
3-token instead of 24-token sequences) with the same rollouts.

**P3 fails, and it is the interesting part of the run** (below).

## Why this is stronger than "flat dose": the original's rollouts were length 1 anyway

The literal original configuration (`ESON-L4`: `max_rollout_length 4`,
`early_stop true`, threshold 1e-4, improvement reward) realised a mean rollout
length of **1.03** over the run. By 100-iteration block: 1.135, 1.012, 1.004,
1.001, 1.001. The first iteration at which all ten rollouts of an iteration
stopped at step 1 was iteration 13 (seed 42) and 26 (seed 43); from there on
97.5% of all rollouts were one step long.

The mechanism is in `Old_dro.py:836-841`: the Bayesian early stop breaks the
rollout the first time a simulated step fails to improve the simulated
incumbent by 1e-4. Once the GP ensemble is conditioned on a few dozen points, a
posterior sample at the acquisition's argmax almost never beats the incumbent,
so almost every rollout is a single non-improving step — and a non-improving
step is a zero reward, so **`rtg[0] = 0` for 97.5% of all training
trajectories** (the length-1 fraction and the rtg0-zero fraction are the same
event and match to three decimals: 0.975 / 0.975).

So the original DT was trained, in practice, on ten triples per iteration of
(the same real state, RTG = 0, the teacher's action), and then queried once
with `target_rtg = 1.0` at that state — a value it had essentially never seen.
The MSE minimiser for a constant input is the mean of the targets: the query
is the mean of the ten teacher actions, shifted by whatever the untrained
RTG-embedding direction does at 1.0. This is the same "DT emits its teacher's
action mean at the read position" mechanism found on MF-DRO (THE_ANSWER,
h171/h173), present in the original design in its most extreme form.

Even with early-stop forced OFF, the label stays degenerate: at L=8,
`rtg[0]` is exactly zero on 87% of trajectories over the run and on 90–97% of
them after iteration 100.

**So the claim "the original does not use training rollout data beyond step 1"
holds three ways in the original setting:**

1. by construction — `_propose_next_candidate` reads only position 0 at
   timestep 0 (Old_dro.py:1028-1037), so positions ≥ 1 reach the query only
   through shared weights;
2. in practice — with the original early-stop, positions ≥ 1 almost never
   existed after the first ~20 iterations (97.5% of rollouts were one step);
3. by intervention — when positions 1–7 are forced to exist (ES-OFF, L=8) they
   change nothing: L8 ≈ L1 (−0.007) and L8 ≈ L8-TRUNC (+0.20).

## Things that were NOT predicted, reported as they are

**The early-stopped original config is the best arm.** ESON-L4 beats its
ES-OFF twin L4 by 1.05 (se 0.19, 9/10, d = −1.77) and beats L1 by 0.77
(se 0.26, 8/10). This is unregistered and its mechanism is not identified by
this experiment. What can be said: ESON-L4 and L1 are indistinguishable in
realised length (1.03 vs 1.00), rtg0 statistics (zero fraction 0.975 vs 0.965,
mean 0.075 vs 0.075) and improvement count (24.7 vs 24.7). They differ only
in the ~11% of rollouts in the first 100 iterations that ran 2–4 steps, and in
the RNG stream from the first rollout on. The diagnostic curves have ESON-L4
*behind* L1 at evaluations 30–105 (11.6 vs 9.5; 5.9 vs 5.0), level at 305 (3.23
vs 3.30) and ahead only in the last 200 evaluations. A late-run divergence
between two arms whose training data is mechanically identical by then is
consistent with a chance excursion at n=10 across seven contrasts, and equally
consistent with the early longer rollouts steering the search somewhere that
pays off late. Not separable here; would need a matched-RNG or larger-n
follow-up if it matters.

**The dose is not monotone in the middle.** L2 is worse than L1 by 0.64
(se 0.22, 3/10) and worse than L4 and L8 too. The endpoints coincide exactly.
With the two registered contrasts both null and seven contrasts read, one
middle excursion at ~3 se is reported, not interpreted.

## Scope and caveats

- Runs on the **current** `src/model/decisionTransformer.py` (causal mask,
  state-token readout), as importing `papers/Old_dro.py` does today. The
  paper's bidirectional action-token-readout DT could see its target action
  during training, so on that model this dose would be uninformative — flat
  because nothing is learned. On the current readout the DT cannot copy its
  target, so the null means what it says. Stated in the protocol before launch.
- Ackley 10D only, n=10, no p-values. Effect sizes for the two registered
  contrasts are d = −0.01 and +0.23.
- Simple regret at a fixed 505-evaluation budget, not the MF frozen metric;
  this is the single-fidelity, iteration-budgeted original setting.

## Files

`code/worker.py` (subclass + config only; Old_dro.py untouched),
`code/readout.py` (applies the registered rule), `results/*.json` (60 runs,
full traces and per-iteration rollout-length / rtg0 diagnostics),
`results/summary.csv`, `logs/`.
