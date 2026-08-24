# When does return-conditioned supervised learning work for offline RL?

**Brandfonbrener, Bietti, Buckman, Laroche, Bruna — NeurIPS 2022**
arXiv:2206.01079 · https://arxiv.org/abs/2206.01079

## Why this is the frame for our entire negative result

MF-DRO *is* RCSL: a Decision Transformer trained on trajectories, conditioned at
inference on a return target. This paper gives the **necessary conditions** for
RCSL to return the optimal policy — and our setting violates them by
construction.

## The conditions (Theorem 2, Corollary 1)

**Theorem 2 (Reduction of RCSL to SL)** requires:

1. **Bounded occupancy mismatch**: `P_{π_f^RCSL}(s) / P_β(s) ≤ C_f` for all `s`
2. **Return coverage**: `P_β(g = f(s)|s) ≥ α_f` for all `s`

giving `J(π_f^RCSL) − J(π̃_f) ≤ (C_f/α_f) H² √(2L(π̃))`.

**Corollary 1** adds **near-determinism** (dynamics ε-close to deterministic):

    J(π*) − J(π_f^RCSL) ≤ ε (1/α_f + 3) H²

**Corollary 2**: exact optimality needs `α_f > 0`, **`ε = 0`** (fully
deterministic), *and* `f(s₁) = V*(s₁)` — i.e. the conditioning function must
already equal the optimal value function.

## The result that explains seven of our failed experiments

Figure 1c and the surrounding text (p. 4) construct an MDP where

> "there exist cases where the bias of RCSL in stochastic environments can remain
> **regardless of the conditioning function**."

They compute `π_f^RCSL(a₁|s₁) = 1/2` and `J(π*) − J(π_f^RCSL) = 1/2 − ε`, then
state plainly that **"merely changing the conditioning function is not enough to
overcome the bias of the RCSL method in stochastic environments."**

### Mapping to our experiments

Every intervention we ran on the conditioning side is an attempt to change the
conditioning function. All of them failed, and the theory says they had to:

| ours | what was changed | outcome |
|---|---|---|
| H4 | AdaLN-Zero conditioning (DDT mechanism) | REFUTED |
| H5 | deny the score head its GP features | REFUTED |
| H8 | sweep RTG within its realised band | 0/12 argmax moves |
| H9 | `alpha_rtg` (the floor) | VOID |
| H10 | un-normalised RTG | VOID |
| H11 | real history + DT-style RTG decrement | A/B null, C VOID |
| H12–H16 | the reward/conditioning quantity itself | better signal, decision unmoved so far |

## Why our setting violates the assumptions maximally

1. **Near-determinism (`ε ≈ 0`) — violated by construction.** The rollout
   transition is `y_τ = current_ko.sample_fantasy(x_τ, ·)`, a **draw from the GP
   posterior** (`mf_dro.py:1264`). It is Gaussian, not ε-close to deterministic.
   The stochasticity *is* the method: fantasy sampling is how MF-DRO simulates.
2. **Return coverage `α_f` — measured, and it is tiny.** Under
   `rollout_reward="improvement"`, **63.0%** of trajectories had `rtg[0] = 0`
   while the conditioning target `f(s)` sat in [0.57, 1.0]. So
   `P_β(g = f(s)|s)` — the chance the behaviour policy actually achieves the
   conditioned return — is small, and the bound scales as `C_f/α_f`. Our
   "starved reward" finding **is** low return coverage, in the theory's own
   terms.
3. **`f(s₁) = V*(s₁)` — we do not have it.** Our target is
   `max(batch_max, α·running_max)`, a heuristic whose band is provably capped at
   `1/α` (our own algebra). Corollary 2 needs the optimal value function.

## Second relevant result: exponential sample complexity

Figure 2 / Corollary 4: even in a **deterministic** MDP, RCSL can require
`~10^{H/2}` samples, because it uses *trajectory-level* information instead of
dynamic programming over transitions. Their framing:

> "Fundamentally, the problem here is that RCSL uses trajectory-level
> information instead of performing dynamic programming on individual
> transitions."

Relevant to us: we train on 200 trajectories of length 8 per iteration. That is
trajectory-level data in exactly the regime the paper warns about.

## Also: they show RCSL cannot stitch

Appendix B gives theoretical and empirical evidence that RCSL **cannot perform
trajectory stitching**, even with infinite data. Stitching is precisely what a
BO policy needs to do — combine good sub-decisions from different rollouts.

## The constructive direction, if we want one

**Dichotomy of Control** (Yang, Schuurmans, Abbeel, Nachum, arXiv:2210.13435)
targets exactly this failure: separate what the policy *can* control from what it
cannot. In our setting the fantasy draw `y_τ ~ N(μ,σ²)` is **not controllable by
the policy**, yet the return conditions on it — the textbook DoC failure mode.
See [[dichotomy-of-control]] (to be written).

## How to use this in the paper

This converts a messy sequence of negative results into a single
theory-predicted claim:

> MF-DRO applies return-conditioned supervised learning to GP-fantasy rollouts.
> The fantasy transition is maximally stochastic, so the near-determinism
> condition of Brandfonbrener et al. (2022) fails by construction, and their
> Figure 1c shows **no** conditioning function can repair the resulting bias.
> Our seven independent conditioning-side interventions, each pre-registered and
> each null, are the empirical confirmation.

Related: [[rtg-attention-underallocation]] (RADT/DDT) — which we refuted as *the*
mechanism; this paper explains why those architectural fixes could not have
worked here regardless of implementation quality.
