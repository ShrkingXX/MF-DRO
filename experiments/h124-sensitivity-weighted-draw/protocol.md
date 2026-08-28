# H124 — the lever the diagnosis prescribes, which nobody has pulled

LOCKED BEFORE ANY RUN. Arm checked for prior existence first: no
sensitivity-weighted candidate draw exists anywhere in the tree.
**NOT LAUNCHED** — the shared `src/` is quiet-locked while a peer session has 10
runs live. Patch held outside the tree, as h102's was.

## Why this, and why it is the obvious remaining experiment

The founding diagnosis says MF-DRO's proposals are ~3x more dispersed than the
baseline's. Measured properly — weighted by how much each input moves the output
— that is **right, and closer to 4x, on both benchmarks where it resolves.**

**And no intervention in this project reduces it.** The ROI moves weighted
dispersion by −0.0013 (ratio 0.33); the L1 loss by +0.0045, the wrong sign. The
two things that work act through the fidelity mix and through something still
unnamed. So the diagnosis's actual prescription — *concentrate the search where
it matters* — is **untried, not tried-and-failed.**

This experiment tries it.

## The mechanism, and why it is in scope

The teacher's candidates come from `_draw_raw()` (mf_dro.py:1189):

    bounds[0] + (bounds[1]-bounds[0]) * torch.rand(roi_raw_pool, d)

**Uniform over the box, every dimension treated alike.** The ROI then *filters*
that draw — and a filter cannot create probability mass the proposal distribution
never had, which is why tightening the ROI has never concentrated anything where
it matters.

**The intervention changes the draw, not the filter.** Per-dimension spread is
scaled by the GP's own fitted **ARD lengthscales** (`RBFKernel(ard_num_dims=d)`,
`gp_ard=True`): short lengthscale ⇒ that dimension matters ⇒ draw tightly around
the incumbent there; long lengthscale ⇒ it does not ⇒ stay broad.

**This is in scope, and the distinction from the standing constraint is precise:**

  - It is **not** `use_candidate_scoring` — no acquisition ranks candidates and
    no argmax selects one. The DT still emits the query.
  - It is **not** imported baseline machinery — no L-BFGS, no Sobol pool, no
    acquisition optimiser. It uses the method's own GP.
  - It **is** the lever the task names: *"the ROI is the lever that shapes the
    training distribution."* This shapes that distribution directly instead of
    filtering it, which is what the diagnosis asks for.
  - Sensitivity comes from **fitted lengthscales, not oracle Sobol indices.**
    Using true sensitivities would be cheating; the model must earn them.

## Design

| | |
|---|---|
| benchmark | Borehole_8D, then Hartmann_6D if it separates |
| seeds | 42-46 **and** 47-51 — n=10 from the start |
| arm | **SENS-DRAW**: `use_roi=False`, lengthscale-scaled candidate draw |
| control | h83/h90 no-ROI at matched seeds (verified same arm, bit-identical) |
| runs | 10 |

Borehole first because it is the only benchmark with a real deficit and the only
one where any intervention has separated.

## Gate

The manipulation must be **observed**: log the realised per-dimension candidate
spread and require it to be **at least 2x tighter in the top-2 sensitivity
dimensions** than an unweighted draw, while the low-sensitivity dimensions stay
within 20% of uniform. A run failing that is void regardless of regret —
the arm must actually concentrate where it claims to.

## Predictions

**P1 (mechanism).** Weighted dispersion falls, ≥8/10. Registered **POSITIVE**:
unlike every previous mechanism prediction here, this one is close to definitional
— the intervention manipulates the draw the dispersion is measured on. If it
fails, the implementation is wrong, not the hypothesis.

**P2 (regret).** Registered **GENUINELY UNCERTAIN, no direction.** Ten mechanism
predictions have been refuted in this investigation, and the entire point of the
experiment is that the link from dispersion to regret is untested. Predicting it
would presuppose exactly what is being measured.

**P3.** Does not make MF-DRO competitive with MF-MES on Borehole. Registered
**POSITIVE** — true of every intervention tried.

## What each outcome means

  - **P1 and P2 both positive:** the diagnosis's prescription works, and the
    original lever was right all along — it had simply never been implemented,
    because filtering a uniform draw cannot concentrate it.
  - **P1 positive, P2 null:** the strongest possible version of "dispersion is
    not the mechanism". The quantity was moved, deliberately and by construction,
    and regret did not follow. That would close the founding diagnosis's own
    hypothesis by direct intervention rather than by correlation.
  - **P1 fails:** an implementation bug. Fix and re-run; no claim either way.
