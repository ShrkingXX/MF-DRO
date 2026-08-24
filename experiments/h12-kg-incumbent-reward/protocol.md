# H12 — the LF reward is the starvation mechanism

## Why this hypothesis exists

Under `rollout_reward="improvement"` an LF step earns **exactly 0.0**
(`mf_dro.py:1290+`). Measured on a real 200-trajectory batch:

- 63.0% of trajectories have `rtg[0] == 0` — dead conditioning signal
- only 23.7% of steps carry any nonzero reward
- `Spearman(n_HF, rtg[0]) = +0.355` (p<1e-6)

Three separate experiments (H9 floor, H10 normalisation, H11 arm C DT-style
decrement) each failed their own manipulation check, and all three failed for
this one reason: **a signal that is identically zero cannot be made to vary by
changing how it is scaled.** H12 attacks the reward definition instead.

## The intervention (one variable: the reward)

`rollout_reward="kg_incumbent"`:

    V_tau = max over a FIXED reference set of mu_H^(tau)
    r_tau = max(0, V_tau - V_{tau-1})      for BOTH fidelities

Reference set: Sobol prefix `y_star_pool[:64]`, fixed for the whole run.

Rationale: `V` is the model's belief about the best attainable HF value — the
same quantity the frozen evaluation's simple regret is defined on. An LF
observation moves `mu_H` only *through rho* in the KO model, so LF's credit is
automatically discounted by how much LF actually informs HF. The certainty
discount is derived from the fitted model, not hand-tuned.

## GATE — run FIRST, standalone, before any comparison

Given three consecutive voids, the manipulation is verified before spending
anything on an outcome comparison. Both must hold on a 200-trajectory batch:

- **G1**: fraction of trajectories with `rtg[0] == 0` drops from 63.0% to **< 20%**
- **G2**: fraction of **LF steps** earning nonzero reward is **> 50%**

If either fails, H12 stops here and reports the gate failure. No comparison is
run and no claim about regret is made.

## Locked predictions (only evaluated if the gate passes)

1. **PRIMARY**: teacher-quality Spearman under `kg_incumbent` is positive with
   p < 0.05 — i.e. the new reward ranks rollouts at least as usefully as
   `improvement` did (+0.191, z=2.63, p=0.0085). A *lower* correlation
   than `improvement` refutes the reward change even if the gate passed.
2. **SECONDARY**: with a non-dead RTG, an in-band RTG sweep moves the argmax on
   **> 30%** of candidate pools (vs 0/12 under `improvement`). This is the
   inert-vs-starved question, asked for the first time on a signal that is
   actually alive.
3. **NULL**: if 2 fails with the gate passed and the reward alive, then RTG
   insensitivity is a property of the **network**, not of the signal — which
   finally resolves the confound rather than voiding on it.

## Out of scope

No regret comparison in this experiment. `PROTOCOL.md` untouched; no run here
contributes to the frozen success test. Cost normalisation (`r/c_ell`) is
deliberately NOT combined with this change — BTG already carries budget, and one
variable at a time.
