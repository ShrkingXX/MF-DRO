# H36 — why does LF's MES collapse while sigma_L stays flat?

## The one open link

H35 measured the chain but left one step unexplained: across a rollout,
`LF/c_L` falls **-37%** while `sigma_L` is flat (it slightly *rises*). Something
other than uncertainty resolution is draining the LF branch's value.

Two candidates can be eliminated by inspection:

- **rho**: `make_fantasy_ko` conditions **without refitting hyperparameters**, so
  `rho` is frozen within a rollout. It cannot be the cause.
- **sigma_L**: measured flat in H35.

That leaves the **`y*` distribution**. Takeno's LF branch computes information
about `y*_H` through a quadrature over sampled `y*`. If fantasy observations push
the `y*` samples away from where LF observations are informative --- higher, or
tighter --- the LF branch loses value without any change in `sigma_L`.

## Measurement

Across rollout steps `tau`, on the same candidate pool, record:

1. the mean and spread of the Thompson-sampled `y*_H` draws
2. the gap `mean(y*) - max(mu_H)` over candidates --- how far the sampled optimum
   sits above the model's best prediction
3. `mean(mu_H)` and `mean(mu_L)`, to see whether the posterior means drift
4. the same at a real-inference state for comparison

## Locked predictions

1. **PRIMARY**: `mean(y*)` drifts **upward** with `tau`, and/or the gap
   `mean(y*) - max(mu_H)` **grows**, while `sigma_L` stays flat. That would make
   `y*` drift the mechanism draining the LF branch.
2. **NULL**: if the `y*` distribution is stable across `tau`, none of the three
   candidates explains the collapse, and the cause lies inside the LF quadrature
   itself --- which we would then state as an open question rather than guess at
   a fourth time.

## Discipline note

My mechanism guess has now been wrong twice in a row on this sub-question (H34's
floor hypothesis, H35's `sigma_L` hypothesis). If this prediction also fails, the
honest move is to stop proposing mechanisms and report the collapse as measured
but unexplained.

Single process, 1 thread. `PROTOCOL.md` untouched.
