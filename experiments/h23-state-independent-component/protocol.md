# H23 — verify the explanation the paper now asserts

## Why

H22 concluded, and `paper/main.tex` now states, that

> "the ranking is dominated by a state-independent component of `w` whose margin
> over the runner-up exceeds anything an 11.5° rotation can overturn."

That was **inferred** from a rotation that failed to change an argmax, not
measured. Three inferences in this project have already failed re-measurement
(H5's swap, H18's diversity claim, H13's decomposition). This experiment tests
the assertion directly, before it ships in a paper.

## Decomposition

Over the 10 distinct τ=0 states, write

    w(s) = w̄ + δ(s),      w̄ = mean_s w(s)

`w̄` is the state-independent component; `δ(s)` carries everything the state
contributes.

## Locked predictions

1. **PRIMARY**: scoring with `w̄` **alone** reproduces the full model's argmax on
   ≥ 11 of 12 pools. If `w̄` alone gives the same decision, the state-dependent
   part is decision-irrelevant and the paper's sentence is verified.
2. **MARGIN**: the median ratio
   `(score gap between top-1 and top-2 under w̄) / (range of ⟨δ(s), cf_k⟩ over k)`
   is ≥ 5. This is the quantitative form of "the margin exceeds what δ can
   overturn". A ratio near 1 would mean the invariance is a coincidence of these
   pools rather than a structural fact.
3. **NULL**: if `w̄` alone does **not** reproduce the argmax, the paper's stated
   explanation is wrong and must be removed, even though the underlying 0/12
   observations stand.

## Secondary, structural observation to confirm

`bias_head(h)` adds a per-candidate **constant** to every score, so it cannot
change an argmax under any circumstances. If confirmed, one of the two
state-dependent quantities the architecture exposes is decision-irrelevant *by
construction*, which is worth stating in the paper.

Single process, 1 thread. `PROTOCOL.md` untouched, no regret claim.
