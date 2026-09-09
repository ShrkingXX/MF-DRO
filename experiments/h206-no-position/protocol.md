# h206 — Does the positional embedding carry anything at all?

Registered BEFORE any h206 code is written or run. CONFIRMATORY.

## Where this comes from

The user's original question 1 listed three ways to handle the DT timestep: hardcode it,
substitute it, or **delete it entirely**. h205 ran the first two and never ran the third.

h205's outcome makes the deletion the sharpest remaining test. Of three arms, the only one
that moved was the one that changed the *label* (B, absolute, 10.68); the arm that added
real context depth (A, prefix, 13.12) was worse than using no history at all. So the
positional channel is where the action is — but h205 cannot say **why** B helped. Two
incompatible readings survive it:

- **(i) B ADDED a useful signal.** Absolute episode time is decision-relevant (how much
  budget is left), and `arange` failed to supply it.
- **(ii) B REMOVED a harmful one.** `arange` welded the index to rollout phase — index 7
  meant "7 steps into a fantasy, RTG≈0" in training and "the real current step, RTG large"
  at inference. B's gain is the deletion of that false signal, and the absolute time it
  substituted contributes nothing.

Deleting the embedding separates them. Under (i) NOPOS should fall back toward the
`arange` arms; under (ii) NOPOS should match B.

## Why (ii) is a live possibility, not a strawman

Three independent reasons, all already established here:

1. **Budget progress is already in the state.** Slot `5M+1` is `step_norm = n_real_iter /
   T_real` (mf_dro.py:3615). h197 had to overwrite the window's historical step_norm with
   the current value because the stale values were OOD. Episode time is therefore encoded
   TWICE, and one copy has already needed patching.
2. **DT's decorrelation mechanism is absent here.** DT samples length-K windows at random
   offsets inside full real episodes, so absolute timestep and window slot are independent
   by construction, which is what lets `embed_t(t)` mean "episode time". Our training
   sequences are forward GP fantasies always anchored at "now", so rollout phase and window
   slot are perfectly correlated with no offset to break them. We never had the property
   that makes DT's positional embedding meaningful.
3. **`nn.TransformerEncoder` here has no positional encoding of its own** (ctor at
   decisionTransformer.py:41). `position_embedding` is the ONLY explicit order signal, so
   the ablation is clean: removing it leaves the causal mask and the four modality-specific
   embedding layers, both of which still distinguish tokens.

## Arms — 2 new x 5 seeds, Borehole_8D, seeds 42-46

All arms K=8, ROI-Q10, `max_seq_length=256`, matching h205's workers exactly.

| arm | positional embedding | note |
|---|---|---|
| **N  NOPOS-K8** | none (pos_emb not added) | the deletion |
| **P  ARANGE-K8** | `arange(T)` (both h205 flags off) | matched control, CURRENT code |

**Why P is not redundant with h196 (13.96).** h196's config is otherwise identical, but it
ran on older core code — h198/h201/h203 have touched the model and policy since, and h203's
`_readout` refactor and `window_fidelity_single_token` flag both sit in the K=8 path. The
identity gate protects `use_roi=False` only; it says nothing about the window path. Quoting
13.96 as today's arange-K8 number would be an unverified assumption, and the whole
comparison rests on it.

In hand, not re-run: h205B absolute-K8 **10.68**, h205C 12.27, h205A 13.12,
h194 CTRL-K1 **11.59** (no window). Saturation floor 43.94.

## Stage 0 — sanity checks, run and reported BEFORE Stage 1

A silent no-op reads as "no effect", so the ablation must be proven real first.

- **SC1** — with the flag ON, `position_embedding.weight.grad` is None or exactly zero
  after a backward pass. The table receives no gradient; the ablation is not cosmetic.
- **SC2** — identity gate: `use_roi=False` still reproduces **122.29066752728207** with the
  flag OFF, bit-identical.
- **SC3** — context still flows. Two windows sharing an identical current state but
  differing history must produce different queries under the flag. This guards the failure
  mode where deleting position accidentally makes the model context-blind as well as
  position-blind, which would make N a disguised K=1 arm rather than a positional ablation.
- **SC4** — arm P reproduces h205's CTRL path: with both flags off and K=8 the training
  sequence is length 8 and touches exactly 8 embedding rows (h205 Stage 0 measured this).

## Predictions, committed now

Band is ±1.26, the same MDE h205 registered.

- **P1 — N matches B** (|N − 10.68| ≤ 1.26). Reading (ii): the positional embedding carries
  nothing, and B's gain was the removal of a misleading signal.
- **P2 — N falls back toward the arange arms** (N − B > 1.26). Reading (i): absolute
  episode time is genuinely informative and `arange` simply supplied it wrongly.
- **P3 — N beats B** (N − B < −1.26). The positional embedding is net harmful under *any*
  labelling, and the right fix is deletion rather than relabelling.

**My honest lean, stated so it can be wrong: P1.** Reasons 1–3 above. I record explicitly
that my last ordering prediction (h205's A ≥ C > B) was not merely wrong but exactly
inverted, so this lean deserves less weight than the argument behind it.

## What each outcome RETRACTS

- **P1** retracts the framing in my h205 readout that "absolute labelling is the fix."
  The mechanism would be subtractive, not additive, and the correct minimal change would be
  to delete the embedding, not to relabel it — simpler, and it removes the `max_seq_length`
  ceiling and the overflow guard entirely.
- **P2** retracts reason 1 above — it would show `step_norm` in the state does NOT already
  carry budget progress in a usable form, which bears directly on the state-encoding work.
- **P3** retracts h205's own conclusion that B is the best available configuration.
- **P on its own is a falsifier for h205.** If ARANGE-K8 on current code lands near B
  (|P − 10.68| ≤ 1.26) instead of near h196's 13.96, then h205's arms were not separated by
  labelling at all, and h205's entire result — including the retraction it already fired —
  must be re-opened. This is checked FIRST and reported whatever it says.

## Compute

10 workers x 1 thread = 10 <= 15. Verify with `tools/count_workers.sh` before launching.
Borehole ~2h/seed, so one batch, ~2h wall.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost_curve 200, metric IMPORTED from h83's
`sr_curve`/`grid`, never re-derived. Read `results/*.json` finals only, never `results/ckpt/`.
Quality compared ONLY at the endpoint. No p-values at n=5. Every run reported, including
failures and gate misses.
