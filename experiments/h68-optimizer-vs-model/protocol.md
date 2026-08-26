# H68 — Is Borehole an OPTIMIZER failure or a MODEL failure?

**CONFIRMATORY.** Predictions locked below before any h68 number exists.
Offline diagnostic on traces already on disk. No new BO runs, no h57 contact.

## Why

h60 narrowed Borehole's unresolved mechanism to two remaining candidates:
**teacher optimisation quality** and **the surrogate class**. Everything else was
excluded by measurement — LF quality (corr 1.000), local refinement (MF-DRO is
the only contractor and still loses), boundary aversion (its queries are the
closest to x*), fidelity allocation, rho, and stall length.

Those two candidates are separable without running anything new, and the
distinction is the whole design question:

- **OPTIMIZER failure** — MF-DRO's acquisition *would* rank a better point highly,
  but its inner search never proposes one. Fix = search harder. h61/h64's pool
  widening is exactly this fix, and on Borehole it bought 1.44x acquisition value
  and moved regret 23.7% -> 19.5%.
- **MODEL failure** — MF-DRO's acquisition ranks the better point *low*. Then no
  amount of searching helps, and the surrogate or the acquisition is wrong.

MI-Greedy supplies the "better point": from the identical 10-point HF init it
reaches 264.41 within 20 HF queries where MF-DRO needs more than 109.

## Method

For each Borehole seed (44/46/48) and each checkpoint k in {10, 20, 40} HF
queries:

1. Refit MF-DRO's own KO surrogate on **MF-DRO's own data** through its first k
   HF queries (plus interleaved LF), from the h57 trace.
2. Score, under that surrogate's MES acquisition:
   - `x_dro`  — the point MF-DRO actually queried next,
   - `x_mig`  — the point MI-Greedy actually queried at its k-th HF query,
   - a 600-point Sobol reference pool, giving the acquisition distribution.
3. Report each as a **percentile of the reference pool**.

Evaluating MI-Greedy's point under MF-DRO's surrogate is well-posed even though
the two ran on different data: the question is "given what MF-DRO knew, would its
own acquisition have wanted this point?"

## Locked predictions

**PRIMARY (the discriminator).** `x_mig` scores **above the 50th percentile** of
MF-DRO's own acquisition pool. That is the OPTIMIZER-failure signature: the
acquisition likes the point and the inner search simply never finds it.

**If `x_mig` scores BELOW the 50th percentile**, it is a MODEL failure — the
acquisition actively disprefers the point that wins, and searching harder cannot
help. This would *contradict* h61's 1.44x Borehole result being a route to a fix,
and would make the surrogate/acquisition the only remaining candidate.

**SECONDARY.** `x_mig` outranks `x_dro` in at least 5 of the 9 (seed, k) cells.

**NULL.** Both sit in the same percentile band (within 10 points) — the
acquisition does not distinguish them at all, and neither candidate is
implicated. This would mean the acquisition is uninformative on Borehole, which
is itself a model failure of a weaker kind.

## What this cannot settle

n = 3 seeds, 3 checkpoints. It classifies the failure; it does not fix it. A
percentile is not a regret. And MI-Greedy's point being well-ranked would not
prove that MF-DRO could have found it — only that its acquisition would have
rewarded it.
