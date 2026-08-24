# H14 — joint MES is already implemented, and it dominates on signal health

**Status: EXPLORATORY.** The gate was run as a spec check on an existing mode,
not as a locked confirmatory test. Labelled as such deliberately.

## Spec result (the important part)

| mode | what `rtg` actually is |
|---|---|
| `improvement`, `kg_incumbent` | `flip(cumsum(flip(r)))` — a genuine **per-step forward cumsum** |
| `mes_entropy` | `log(b_tau) − log(b_T)`, `b_T` Thompson-drawn from the **fully-conditioned** model at the END of the rollout (`mf_dro.py:1040-1044`) |

For a Gumbel(mu, b), `H = ln b + gamma + 1`, so

    log(b_0) − log(b_T)  ==  H[y*|D_0] − H[y*|D_T]  ==  I(y*; y_1..y_T)

`mes_entropy` is therefore **already the joint, set-level information gain of the
final dataset about y\***, not a per-step cumulative sum. The per-step-cumsum
critique lands squarely on `improvement`/`kg_incumbent` — the modes H12/H13 were
building — and not on `mes_entropy`.

## Measured signal health (200-trajectory batch, identical seed)

| arm | dead `rtg[0]==0` | LF nonzero | mean / CV | negative |
|---|---|---|---|---|
| `improvement` (current default) | 63.0% | 0.0% | 0.1022 / 1.964 | 0.0% |
| A `kg` topk=1 clipped | 4.5% | 27.2% | 0.1396 / 1.374 | 0.0% |
| B `kg` topk=1 signed | 0.0% | 100.0% | −0.0204 / **−12.934** | 67.0% |
| C `kg` topk=5 signed | 0.0% | 100.0% | 0.0676 / 3.677 | 38.5% |
| **D `mes_entropy` (JOINT)** | **0.0%** | **100.0%** | **0.2903 / 0.661** | **5.5%** |

D dominates every KG variant: no dead signal, full LF credit, the **largest mean
and by far the tightest CV (0.661 vs 3.677)**, and almost no negative mass. The
whole H12/H13 line was reinventing — worse — a mode that already existed.

Minor caveat: D shows 677 LF steps vs 683 for the others. `_rollout_gumbel_b`
consumes RNG, so the fidelity draws diverge slightly. Irrelevant to these
fractions, but it means D is not bit-comparable step-for-step.

## The tension that must now be resolved

`mes_entropy` was **abandoned** because it scored worse on the teacher-quality
gate: Spearman **+0.129, 5/10 groups negative, p ≈ 0.32**, versus `improvement`
at **+0.191, z = 2.63, p = 0.0085**.

That gate is now suspect. If "teacher quality" was operationalised as a
regret-like quantity, then `improvement` — which *is* a regret-like quantity —
would correlate with it better by construction, and the gate would be measuring
*"does the reward resemble regret"* rather than *"is the reward a good
conditioning signal"*. On the signal-health axis measured here, the ranking is
reversed and not close.

Resolving this needs the gate's own definition re-read and, if it is
regret-anchored, a fair re-measurement. That is the next locked experiment, not
a conclusion to draw here.

## Correction to my own framing

I proposed `kg_incumbent` as "the evaluation-aligned choice" and treated joint
MES as the second arm. The spec shows the joint quantity was already available
and better-behaved. My H12/H13 work stands only as ablation machinery.
