# h188 — the synthesis PREDICTS the DT's emitted query, and its own negative control fires

**EXPLORATORY.** No new runs. Re-analysis of `teacher_action_stats` against the DT's
realised queries. Borehole, seeds 42–46.

## The prediction and why it carries its own control

The synthesis says the DT emits **its teacher's τ=0 action mean**.
`teacher_action_stats` records the **all-τ** mean, which gives a built-in test:

- **Rollout length 1** — all-τ mean **is** the τ=0 mean. The prediction *must* hold.
- **Rollout length 8 with a sharp τ=0 vs τ>0 split** (HEAD, TAIL) — all-τ mean is
  **not** the τ=0 mean. The prediction *must fail*.

A positive and a negative control from one statistic, with the direction fixed in
advance by the arms' construction.

## Result

Distance between the teacher's action mean and the DT's own query centroid (both
late in the run, unit box):

| arm | rollout length | \|teacher mean − DT centroid\| | as a fraction of seed noise (0.82) |
|---|---|---|---|
| ROLLOUT2 | 2 | **0.1087** | 0.13× |
| ROLLOUT1 | 1 | **0.1353** | 0.16× |
| ROLLOUT4 | 4 | **0.1578** | 0.19× |
| TAIL-MES | 8 | **0.6170** | 0.75× |
| HEAD-MES | 8 | **0.7148** | 0.87× |

**Where the recorded statistic is the τ=0 mean, it predicts the DT's queries to
within 13–19% of the natural seed-to-seed variation. Where it is not, the prediction
fails by 75–87%.** A **4.5–5.3× separation** between positive and negative controls,
in the direction the synthesis requires and could not have chosen after the fact —
the split is fixed by each arm's rollout length.

## What this adds

h185 established the DT is a per-timestep constant. h186 established it ignores its
inputs. This shows **which** constant: the teacher's τ=0 action mean, recovered
quantitatively rather than inferred from performance orderings. It is the first
result on this front that *predicts* the DT's output from an independently recorded
quantity, rather than explaining it afterwards.

## Limits

- **Five arms, Borehole only.** `teacher_action_stats` was recorded on h171 and h172
  and nowhere else, so this cannot be extended.
- The teacher mean is the **final** iteration's; the query centroid is the **last 20**
  queries. Both are "late in the run" but not exactly time-matched.
- Only the L=1 case is an exact identity between all-τ and τ=0 means; L=2 and L=4 are
  approximations, and their agreement (0.109, 0.158) is correspondingly weaker
  evidence than ROLLOUT1's.
- Correlational. Nothing intervenes on the teacher's mean to move the DT's output.

## What could RETRACT it

- An arm where the two means coincide by construction and the prediction still fails,
  or an L=8 split arm where it unexpectedly holds. Both would break the control
  structure that makes this more than a curve fit.
- Recording a genuine **per-τ** breakdown of teacher actions would replace this whole
  inference with a direct measurement. It is a cheap addition (d floats per τ) and is
  the obvious thing to add to any future run.
