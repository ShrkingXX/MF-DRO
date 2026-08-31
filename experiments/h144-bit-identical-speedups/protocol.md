# h144 — Bit-identical speed-ups, gated

STATUS: LOCKED before any code change.
TYPE: EFFICIENCY. **Gated on bit-identity, the h136 standard.**

## What h141 actually showed (and what my P1 got wrong)

The rollout is 96.2% of the run; DT training is 4.7%. Within it, self-time:

    torch.matmul                 253.0s      267,120 calls
    exp_ (RBF kernel)            243.0s      417,388
    exact_predictive_covar       126.8s      362,164
    linalg_cholesky_ex            65.2s      181,944
    _compute_mes_lf_vectorized    50.9s       32,508   (316.9s cumulative)
    scipy _distn cdf              31.0s    2,327,292   (110.1s cumulative)

Cumulative: `compute_joint_mf_mes` 555s, `_rollout_gumbel_b` 318s,
`_compute_mes_lf_vectorized` 317s, `thompson_sample_y_star` 602s (shared).

**GP posterior linear algebra is ~47% of the run**, driven by 362,164 posterior
calls. There is no single 10x win here; the wins are incremental.

## An assumption of mine that was WRONG, corrected before building on it

In h141's write-up I proposed "batch posteriors across trajectories — the GP does
not change during a rollout batch." **That is false.** `simulate_mf_trajectory`
conditions the model at every rollout step via `make_fantasy_ko` (its own
docstring: "conditioned model, reassigned every step below"). Every one of the
2,160 trajectory-steps has its own model, so there is nothing to share across
trajectories. I verified this by reading the code rather than proceeding.

## The two changes, and why each is safe

**C1 — `scipy_norm.cdf` -> `scipy.special.ndtr`.** `norm.cdf` delegates to `ndtr`
through `rv_continuous`'s generic machinery. **Verified bitwise identical on
200,000 random inputs: max |difference| = 0.0, `np.array_equal` True**, and
1.6x faster on large arrays (more on small ones, where the wrapper dominates).
Targets 110.1s cumulative of which only 31.0s is real work.

**C2 — share the `hf_proxy` posterior inside `compute_joint_mf_mes`.** It is
computed twice per call on the *identical* model and *identical* candidates: once
in `thompson_sample_y_star`, once in `_compute_mes_hf_vectorized`. Constructing a
posterior consumes no RNG (deterministic linear algebra); only `rsample` does. So
computing it once and reusing it leaves the RNG stream untouched.

**EXPLICITLY NOT DONE — `norm.pdf` -> explicit formula.** Verified **NOT** bitwise
identical (max rel err 4.3e-16, one ULP), despite being 2.6x faster. It would
break the bit-identity gate, so it is out. Recording the measurement so nobody
re-derives it and assumes it is safe.

## Gate (blocking)

Borehole ROI-Q10 seed42 on the changed tree vs h84's stored trace: **83+ queries,
identical `fid`, `x`, `y` at every one.** Any difference and the change is
reverted. This is the h136 standard and it is the whole reason these two changes
were chosen over faster but numerically-different ones.

## Prediction (locked)

**P1.** Combined speed-up is **>= 5% and < 25%** of wall-clock. FALSIFIED outside
that band.

Grounds: C1 targets ~110s of 1469s with maybe 70% of it recoverable overhead; C2
eliminates 17,280 of 362,164 posterior calls (4.8%). **This does not close the
16-29x gap to MF-MES and I am not claiming it will.** The honest framing is that
the gap is structural — 60 simulated trajectories per BO iteration, each running a
full MF-MES acquisition — and no bit-identical change reaches it.
