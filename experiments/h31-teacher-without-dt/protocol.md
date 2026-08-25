# H31 — does the transformer add anything over its own teacher?

## The question the paper needs

Everything so far characterises what MF-DRO's policy *is*: a fixed linear rule
that reproduces its teacher's score function (H23, H30). The obvious test has
never been run: **how does the teacher perform on its own, with no Decision
Transformer at all, under the frozen evaluation?**

H24 gives a reason to expect a gap --- the student's argmax agrees with the
MF-MES teacher's on only **8/12** pools, so the distillation is lossy.

Three outcomes, all informative:

- **Teacher better**: the transformer is a **net negative**. The entire DT
  apparatus costs performance relative to simply running the acquisition. That
  is the sharpest possible form of this paper's result.
- **Teacher equal**: the transformer is redundant --- consistent with the
  fixed-rule finding, and it means the method's complexity buys nothing.
- **Teacher worse**: the transformer genuinely contributes, which would
  contradict the fixed-rule account and require the paper to be rewritten.

## Design (one change: delete the DT)

Identical to H17's arm in every respect --- same seeds 42--51, `cost_budget=200`
post-init, `initial_hf=36`, `initial_lf=60`, same KO ensemble, same uniform
200-candidate pool per iteration, same cost accounting and regret computation ---
except that the query is `compute_joint_mf_mes`'s argmax over that pool instead
of the DT's. No rollouts, no DT training, no RTG/BTG.

Implemented by replacing the proposal step only, so the surrounding machinery
(initial design, budget bookkeeping, regret curve) is literally the same code.

## Locked predictions

1. **PRIMARY**: direct MF-MES mean final simple regret is **lower than or equal
   to** MF-DRO/joint-MES's $0.4007 \pm 0.0475$, paired across the ten seeds.
2. **SECONDARY**: report it against the frozen success bar ($0.3825$) as well ---
   if the *teacher alone* clears a bar the full method fails, that is worth
   stating plainly.
3. **NULL / CONTRADICTION**: if direct MF-MES is clearly worse, the fixed-rule
   account is incomplete: the DT must be contributing something the linear
   analysis missed, and Sections on the mechanism need revisiting.

## Scope

`PROTOCOL.md` untouched --- this adds a method arm, it does not change the
evaluation. Baselines reused from h1 as before.

10 jobs, `num_workers=10 x threads_per_worker=1`. Expected far faster than
MF-DRO's 98 min since there is no rollout generation or DT training; **ETA
15--40 min**, deliberately wide given two prior misses on this class.
