# h197 — the human's full specification, and the audit that found what was missing

**CONFIRMATORY.** Registered before the code is written and before any result exists.

## The requested configuration, audited item by item

| # | spec | current implementation | status |
|---|---|---|---|
| 1 | default MES acquisition + fidelity selection as teacher | `rollout_policy='mes'` (default); h83 does not override | **MATCHES** |
| 2 | MES reward (RTG) labelling | h83 sets `rollout_reward="mes_entropy"` → the `b_τ` path, `rtg[τ]=log(b_τ)−log(b_T)` | **MATCHES** |
| 3 | training on GP rollouts, labelled states + RTG, **L1 loss** | `loc_loss` defaults to `'mse'`; h83 never sets it | **DOES NOT MATCH** |
| 4 | inference on the real sequence; <8 iters use all, ≥8 use the 7 most recent | `inference_context_k=K` gives K−1 history + current; K=8 ⇒ exactly 7 most recent | **MATCHES** (when K=8 is set) |
| 5 | real points labelled with real states **and information gain computed the same way as rollouts** | **NOT IMPLEMENTED.** `_real_hist` stores the RTG *target*; no `b` is ever computed for a real query | **MISSING** |
| 6 | timestep = relative position within the window | `ts = torch.arange(T)` | **MATCHES** |
| 7 | dynamic RTG prompt at inference | `rtg_tgt = self._last_rtg_target`, recomputed each iteration | **MATCHES** |

**Two gaps: (3) the loss is MSE, and (5) real queries carry no computed information gain.**

**The currently-running h196 does NOT implement this spec.** It fixes only h195's defect #1
(feeding real past *actions*); its loss is MSE and its historical RTGs are still targets,
not computed information gains. h196 remains a valid single-variable test of that defect
and is allowed to finish; h197 is the spec-compliant arm.

## What (5) requires, and why it is buildable

The rollout computes `b_τ` via a closure `_rollout_gumbel_b(ko)` = Thompson-sample `y*_H`
from the HF proxy over the candidate pool, then Gumbel-fit the scale. Every ingredient is
module-level and reusable outside the rollout: `_build_hf_proxy_model`,
`thompson_sample_y_star`, `fit_gumbel_to_samples`.

So for each real query we compute `b_real` from the **real** GP at that iteration, store
it, and label the window's historical entries with

```
rtg[τ] = log(b_τ) − log(b_last)
```

exactly the rollout's formula, with the window's final step playing the role of `T`.
**The current step keeps the dynamic target** (item 7), so history carries what actually
happened and the current position carries what is being asked for.

Cost: one extra Thompson+Gumbel fit per real iteration, against 8×60 of them per iteration
inside the rollouts — negligible.

## The arm

Borehole, seeds 42–46, ROI-Q10, `inference_context_k=8`, **`loc_loss='l1'`**, real-action
feeding (h196), **plus computed real-query information gain**.

Controls, all already run: **CTRL-K1 = 11.59** (K=1, MSE), **h194 WINDOW-K8 = 16.58**
(K=8, zeroed actions, MSE), and h196 (K=8, real actions, MSE) when it lands.

## Gate

Threshold: the pre-existing 10.9% worst-case floor on ROI-Q10's 11.59 = **1.26 rel% pts**.

- **P1 — the spec-compliant window HELPS**: h197 − CTRL-K1 < **−1.26**
- **P2 — no effect**: |h197 − CTRL-K1| ≤ 1.26
- **P3 — still hurts**: h197 − CTRL-K1 > **+1.26**

Reported alongside (not gates): h197 − h196 isolates the RTG-labelling + L1 change from the
action-feeding change, and h197 − h194 WINDOW gives the total effect of fixing everything.

## What this could RETRACT

- **P1 fires → "input-side fixes cannot help" acquires a real exception.** That claim
  currently unifies three Phase-1 nulls (state, conditioning, history) and appears in
  findings.md's Phase 2 header. It would have to be rewritten, and h194's P3 would be
  demoted to an artifact of a defective implementation.
- **P3 fires → the window is harmful even when fed exactly as specified**, which is the
  strongest possible form of h194's result and closes the question.
- **P2** leaves the mechanism intact and says the implementation defects were not what
  limited the window.

## SC before any result is read

1. Real-query `b` is actually computed and varies (a constant `b` ⇒ all historical RTGs
   identical ⇒ silently equivalent to the old behaviour).
2. Historical RTGs in the window differ from the current dynamic target.
3. `loc_loss` is genuinely `'l1'` in the built config.

**A silent no-op on any of these reads as P2**, which is why they are checked first — the
same failure that h196's SC caught, where `_hist` was rebuilt and dropped the actions.

## Prerequisite

`tools/identity_gate.py` must PASS exactly; the real-`b` computation is additive and gated
on `inference_context_k > 1`, so the default path must be untouched.
