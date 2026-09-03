# h196 — feed the REAL past actions in the window, as the DT paper does

**CONFIRMATORY.** Registered before the code is written and before any result exists.
Follows directly from the h195 audit.

## The defect this fixes

h195 (human-requested audit against `papers/DT.pdf` Algorithm 1) found that our
sliding-window inference **zeroes every action token, including the historical ones**:

```python
ax = torch.zeros(1, T, self.action_dim, ...)   # decisionTransformer.py:668
ae = torch.zeros(1, T, dtype=torch.long)
```

while the paper accumulates the actions actually executed (`a + [action]`) and **training
feeds real ones** — `forward_mf`'s docstring: *"the sequence still needs a real filled
action token, same as always"*.

With the 4-token layout `[rtg, btg, state, action]` and a causal mask, the state token at
step t (position 4t+2) **does attend** to earlier action tokens (4t'+3). So the window
hands the model step-tuples it never saw in training — *state s followed by action 0*.
The added context is **off-distribution**, not merely uninformative.

## The fix, and why it is exact

- `_real_hist` (`mf_dro.py:3404`) gains `ax` and `ae`.
- **`x_t` at that line is still NORMALISED [0,1]^d** — the rescale to the benchmark domain
  happens later at `mf_dro.py:3472`. Training stores
  `actions_x = (x_tau - bounds[0]) / (bounds[1] - bounds[0])`, i.e. **the same space**. So
  the recorded action needs no conversion, which is the whole reason this is safe.
- **Fidelity is recorded after its overrides.** `ell_t` at line 3404 is the DT's proposal;
  the HF floor and HF ceiling can still change it before the query executes. The paper
  records the action *executed*, so `_real_hist[-1]['ae']` is updated at the point the
  fidelity is final. `x_t` is never modified by those overrides, so it is correct at 3404.
- `propose_mf` uses the recorded actions for the K−1 historical slots and keeps a **zero
  placeholder for the current step only** — which matches the paper, whose `a` is short by
  one at prediction time.

## Gate

Borehole, seeds 42–46, ROI-Q10, `inference_context_k=8`. Compared against **h194's
WINDOW-K8** (same config, zeroed actions) and h194's **CTRL-K1**.

Threshold: the pre-existing 10.9% worst-case floor on ROI-Q10's 11.59 = **1.26 rel% pts**.

- **P1 — the fix matters**: |h196 − h194 WINDOW| > 1.26. Feeding real actions changes the
  window's outcome, so h194's window result was about a **broken** window.
- **P2 — the fix is immaterial**: |h196 − h194 WINDOW| ≤ 1.26. The zeroed actions were not
  what limited the window; h194's conclusion stands as stated.

Secondary, reported alongside: **h196 − CTRL-K1**, i.e. does a *correctly fed* window beat
no window. That is the question h194 was meant to answer and could not.

## What this could RETRACT

- **P1 fires → h194's Stage 1a conclusion is scoped to a defective implementation** and
  must be restated. Any claim of the form "the window does not help" would need
  "as implemented with zeroed past actions" attached, which materially weakens it.
- If h196 beats CTRL-K1 beyond the threshold, that is **P1 for h194's own gate** and forces
  the exception into "input-side fixes cannot help" that h194's protocol already names —
  it would break the unification of three Phase-1 nulls.
- **P2 leaves the mechanism untouched** and makes h194's null the stronger reading.

## Prerequisite and compute

`tools/identity_gate.py` must PASS exactly — the change is additive and the modified
`propose_mf` branch only executes when `inference_context_k > 1`, so the default path
must be untouched.

5 workers. **Launched only after h194 Stage 1a's two arms finish**, to avoid adding
contention to a comparison already in flight.
