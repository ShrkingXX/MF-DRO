# h145 — Does an ORACLE-quality rollout teacher lift MF-DRO? (performance ceiling)

STATUS: LOCKED before any code change or run.
TYPE: CONFIRMATORY, and it measures a CEILING rather than proposing a method.

## The question

Every ROI variant shapes the DT's training distribution by filtering the rollout
teacher's candidates. **h145 asks what the best possible teacher is worth.** If
trajectories that walk straight to the true optimum do not lift final regret, then
no amount of teacher-shaping can, and the ROI line is capped by something else
entirely. If they do, the gap between oracle and MES teachers is the headroom any
ROI strategy is competing for.

This is a diagnostic with oracle access. **It is not a method** — x* is not
available at run time in any real setting.

## Design: the ONLY difference is which x is queried

A prior implementation exists (`_synthetic_expert_worker.py`) and **its own NOTE
disqualifies it for this question**: it labels RTG as
`(y_final - y_tau)/(y_final - y_0)`, and records that this is *"a DIFFERENT
quantity than the real pipeline's MES-entropy RTG (log(b_tau/b_T))"*. Under that
version, teacher quality and reward definition change together and neither can be
attributed. The user's instruction — compute state, RTG and BTG exactly as normal
MF-DRO — rules it out, correctly.

**So the expert supplies the x-sequence and nothing else.** Everything downstream
runs through the identical real code path: `_extract_mf_state`, `make_fantasy_ko`
conditioning per step, `_rollout_gumbel_b` per step,
`rtg[tau] = log(b_tau) - log(b_T)`, BTG as the same backward cumsum of the same
per-step costs.

Implemented as an optional `forced_x` argument to `simulate_mf_trajectory`,
default `None`. When `None` the function is bit-for-bit what it is today.

**Expert path**, per trajectory, `T = rollout_length`:
`x_start ~ Uniform(domain)`, `x_tau = x_start + (x* - x_start) * tau/(T-1)`,
arriving exactly at x* at `tau = T-1`.

    Hartmann_6D  x* = [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]   f = 3.3224
    Borehole_8D  x* = [0.15, 100.0, 95090.98, 1110.0, 116.0, 700.0, 1120.0, 12045.0]
                                                                          f = 309.5751

**Fidelity is NOT forced.** The prior version forced HF at every step, which
changes the fidelity policy as well as the location policy. Here `ell_tau` is
chosen by the same cost-normalised MES criterion the real path uses, evaluated at
the forced x. So the location policy is the only thing replaced.

## Comparability with existing MF-DRO statistics

Same initial design (`n_hf`/`n_lf` per h83), same **cost budget 200 post-init**,
seeds **42-46**, read with h83's frozen `sr_curve` + `grid` at cost 200, rel% of
|optimum|. The prior implementation used `cost_budget=1e9`, `bo_iterations=100`
and a different initial design — **none of its numbers are comparable to anything
in the current record**, which is a second reason to rewrite rather than re-run it.

## Sanity checks — ALL must pass before any result is read

**SC1 endpoint.** For every trajectory, `x[T-1] == x*` to machine precision.
**SC2 interpolation.** Points are collinear and evenly spaced from `x_start` to x*.
**SC3 objective.** `y[T-1]` equals the benchmark's known optimal value to within
its own recorded solver gap (Hartmann 2.2e-06, Borehole 4.5e-04).
**SC4 default-path identity (THE GATE).** With `forced_x=None`, a full run is
bit-identical to a stored trace — identical `fid`, `x`, `y` at every query. This
is what licenses "the only difference is trajectory quality".
**SC5 state identity.** `states[0]` of an expert trajectory is bit-identical to
the state the real pipeline produces at that iteration. The SF port documents this
exact trap: seeding from an isolated `[x_start]` history instead of the real
accumulated data created a severe train/inference mismatch.
**SC6 RTG provenance.** Assert the `mes_entropy` branch produced `rtg`
(`log(b_tau) - log(b_T)`), not any y-improvement formula. Fails loudly otherwise.
**SC7 BTG.** Equals the backward cumsum of the same per-step costs; `btg[-1]`
equals the final step's own cost.
**SC8 manipulation observed (G3).** Mean normalised distance from teacher actions
to x* is **much smaller** for expert than for real MES rollouts, measured on the
same run. Confirms the intervention did what it claims rather than reading it back
from config — the standing G3 requirement.

## Predictions (locked)

**P1.** Oracle-teacher final regret is **better** than the matched MF-DRO control
(h83, same seeds), effect >= 1.0 on at least one of the two benchmarks.
FALSIFIED if effect < 1.0 on both.

**P2 (no direction registered).** The size of any gain, per benchmark, reported
whatever it shows. I will not predict magnitude: the DT may be capacity-limited
rather than teacher-limited, and I have no measurement either way.

## What this could RETRACT

**"The ROI is the lever that shapes the training distribution" as a route to
improving MF-DRO.** If a perfect teacher does not help, then teacher quality is
not the binding constraint, and every ROI variant — including the one that works
on Borehole — is exploiting something other than what the framing claims. That
would reframe the whole line, and it is the outcome I should watch hardest,
because P1 failing is the result that costs the most.
