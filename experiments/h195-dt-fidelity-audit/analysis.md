# h195 — audit: sliding-window inference vs the Decision Transformer paper

**EXPLORATORY, no runs.** Human-requested code review. Compares our sliding-window
inference against Algorithm 1 of `papers/DT.pdf` (Chen et al., *Decision Transformer*).

## The paper's evaluation loop (Algorithm 1, p.5), verbatim

```
R, s, a, t, done = [target_return], [env.reset()], [], [1], False
while not done:
    action = DecisionTransformer(R, s, a, t)[-1]
    new_s, r, done, _ = env.step(action)
    R = R + [R[-1] - r]                      # decrement returns-to-go with reward
    s, a, t = s + [new_s], a + [action], t + [len(R)]
    R, s, a, t = R[-K:], ...                 # only keep context length of K
```

Four properties matter: **`a` accumulates the actions actually executed**; **R
decrements by the achieved reward r**; **t is the true episode step**; context is the
last K.

## What our implementation does

**Tracking (`mf_dro.py:3404`).** Each real query appends
`{'state', 'rtg', 'btg'}` to `_real_hist` — recorded *after* proposing, so a step never
sees itself. **The action taken is not stored.**

**Feeding (`decisionTransformer.py:668`).** With `hist`, the window is assembled as
states/rtg/btg for the K−1 preceding queries plus the current one, then:

```python
ax = torch.zeros(1, T, self.action_dim, dtype=state.dtype)   # ALL T action slots
ae = torch.zeros(1, T, dtype=torch.long)
ts = torch.arange(T, dtype=torch.long).unsqueeze(0)
```

Sequence per step is 4 tokens `[rtg, btg, state, action]`; causal mask applied; readout
is the **last state token**.

## FOUR INCONSISTENCIES

### 1. Past actions are ZEROED — the one that matters for the window

The paper feeds `a`, the actions actually executed. We feed **zeros in every action
slot, including the historical ones**. Training does the opposite — `forward_mf`'s own
docstring says *"the sequence still needs a **real filled action token**, same as always"*,
and in candidate-scoring mode it even reconstructs the action from
`candidates[b,t,chosen_idx[b,t]]` precisely so the slot is real.

With the 4-token layout and a causal mask, the state token at step t sits at position
4t+2 and **does attend** to every earlier action token at 4t'+3, t'<t. So the DT was
trained with real past actions in exactly the slots inference fills with zeros.

**Consequence:** the window hands the model step-tuples it never saw in training —
"state s followed by action 0". The extra context is not merely uninformative, it is
**off-distribution**. This is a strong candidate explanation for why added history has not
helped, and it is **fixable**: the actions are known (`x_t`, `ell_t` from prior
iterations); they are simply not recorded in `_real_hist`.

*(The **current** step's zero placeholder is correct and matches the paper — `a` is short
by one at prediction time by construction.)*

### 2. Returns-to-go are not decremented

Paper: `R = R + [R[-1] - r]`. Ours (`mf_dro.py:3251`): `rtg_tgt =
self._last_rtg_target`, a training-batch target recomputed per iteration
(`max(batch_max, 0.5·running_max)`). The window therefore carries a sequence of
**targets**, not a returns-to-go trajectory that responds to what happened.

### 3. No reward is computed for real queries at all — the root cause of #2

The reward label `rtg[τ] = log(b_τ) − log(b_T)` exists **only inside
`simulate_mf_trajectory`**; `b_τ` is a Gumbel-fitted posterior-max scale computed from
the *fantasy* rollout. **No `b_τ`, and no MES information gain, is ever computed for a
real query.** So there is no achieved reward to decrement by — #2 is not an oversight in
the window code, it is a consequence of the real trajectory having no reward labels.

Answering the question directly: **the information gain of real queries is never
calculated.** Only simulated rollouts carry it.

### 4. Timestep semantics differ

Paper: `t = t + [len(R)]`, the true episode step, with a learned per-timestep embedding.
Ours (`decisionTransformer.py:670`): `ts = arange(T)` — positions **within the
window**, re-based every iteration, so the same real query occupies a different position
depending on when it is read. This is deliberate and documented (real iteration counts are
far outside the trained range 0..L−1), but it is not the paper's semantics.

## What is faithful, and should not be "fixed"

- **Causal mask at inference** — matches GPT masking; a previous bug made inference
  bidirectional while training was causal, and it was fixed.
- **Readout at the state token** — matches *"the prediction head corresponding to the input
  token $s_t$ is trained to predict $a_t$"*.
- **Position embedding repeated across the token group** (`repeat_interleave`) — matches
  *"an embedding for each timestep is learned and added to each token … one timestep
  corresponds to three tokens"* (four here, since we carry both rtg and btg).
- **Context truncation** to the last K−1 plus current — matches `R[-K:]`.

## Ranking, and what to do

**#1 is the actionable defect** and the only one that plausibly changes the window's
outcome: it is a genuine train/inference mismatch, it lands precisely on the tokens the
window adds, and the missing data is already available. #3 is the deepest but is a design
property of the method (real trajectories are unlabelled), not a bug. #2 follows from #3.
#4 is a deliberate, documented trade-off.

**This does not invalidate h194 Stage 1a**, which is a like-for-like K=1 vs K=8 comparison
under the current implementation. It does mean a null there would be **a result about this
implementation of the window, not about windows in general** — and that caveat must travel
with the number.
