# H33 — is the fidelity choice where student and teacher diverge?

## Why

H32 showed the student's *location* choice is a faithful approximation of the
teacher's (median teacher-rank 2 of 200). In recording the revised expectation
for H31 I named the one remaining candidate for a gap:

> "most plausibly the fidelity choice, which the student makes by a Bernoulli
> draw from a near-constant probability while the teacher chooses `ell`
> deterministically as part of its argmax."

This tests that hypothesis directly rather than waiting to infer it from H31.

## The asymmetry under test

- **Teacher**: chooses `ell` as part of `argmax_{x,ell} MES(x,ell)/c(ell)` --- a
  state-dependent, deterministic decision.
- **Student**: draws `ell ~ Bernoulli(p)` where H21 measured `p` spanning only
  **0.1248--0.1286** across an entire run with weights fixed --- i.e. very nearly
  a constant-rate coin flip.

If `p` carries no state information while the teacher's `ell` does, the student's
fidelity policy is **noise relative to the teacher's**, regardless of how well it
matches on location.

## Measurement

Over the same candidate pools and states, record:

1. the teacher's `ell` choice and its variability across pools/states
2. the student's `p` and its variability
3. `corr(p, teacher's ell)` --- does the student's probability track the
   teacher's decision at all?
4. the realised HF rate of each

## Locked predictions

1. **PRIMARY**: `corr(p, teacher ell)` is **below 0.2 in absolute value**, i.e.
   the student's fidelity probability is uninformative about what the teacher
   would do.
2. **SECONDARY**: the teacher's `ell` varies across decisions (it must, or the
   comparison is vacuous and this experiment is void).
3. **NULL**: if `p` tracks the teacher's `ell`, the fidelity channel is working
   and the H32 hypothesis is wrong --- H31's outcome would then need a different
   explanation.

Single process, 1 thread (11 with H31's 10). `PROTOCOL.md` untouched.
