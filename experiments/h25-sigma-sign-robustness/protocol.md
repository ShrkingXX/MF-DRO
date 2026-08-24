# H25 — is the negative sigma_H weight real, or one seed's accident?

## Why this must run before the paper ships

H24 measured the learned rule's coefficients on **a single trained model**
(seed 44) and found `w_sigma_H = -0.5487`. That observation is now a headline
claim in `paper/main.tex`:

> "trained on rollouts from an information-seeking MF-MES teacher, it converges
> to a fixed rule that inverts the sign of its teacher's defining term."

A sign estimated from $n{=}1$ cannot carry that sentence. Every strong claim in
this project that rested on a single unverified observation has had to be
retracted or re-measured.

## Design

Train an independent model on each of the ten frozen-protocol seeds (42--51),
identical configuration otherwise, and record the mean scoring coefficients
$\bar{w}$ for each. One variable: the seed.

Report per-seed $\bar{w}_{\sigma_H}$, $\bar{w}_{\mu_H}$, $\bar{w}_{\mu_L}$ and
$\bar{w}_{\sigma_L}$.

## Locked predictions

1. **PRIMARY**: $\bar{w}_{\sigma_H} < 0$ on **at least 8 of 10** seeds. This is
   the robustness bar for keeping the uncertainty-aversion claim.
2. **SECONDARY**: $\bar{w}_{\mu_H} > 0$ on at least 8 of 10, confirming the
   exploitative component is likewise stable.
3. **NULL**: if the sign of $\bar{w}_{\sigma_H}$ is split roughly evenly, the
   H24 result is a single-seed artefact, the uncertainty-aversion sentence must
   be **removed from the paper**, and H24 is downgraded to an observation about
   one model.

## What a pass does and does not license

A pass licenses "the learned rule is consistently uncertainty-averse on this
benchmark". It does **not** license any claim about other benchmarks, and we
will not make one. Ten seeds of one objective is the scope.

## Compute

10 jobs, `num_workers=10 x threads_per_worker=1` (within the 15 limit).
`PROTOCOL.md` untouched; no regret claim.
