# h141 — WHERE does MF-DRO's wall-clock actually go?

STATUS: LOCKED before profiling.
TYPE: CONFIRMATORY (a locked prediction about the profile), then EXPLORATORY
      optimisation guided by whatever it shows.

## The direction this opens

MF-MES runs a Borehole seed in **5.0 minutes**; MF-DRO takes **82-143**, a
16-29x gap, and it is the same baseline that beats MF-DRO on regret (8/10) and on
the founding diagnosis's own metric (9/10). The ROI adds a further 1.2-1.7x on
top. **No profile of MF-DRO exists anywhere in this project** — every runtime
claim to date has been an end-to-end `_wall_s` number with no breakdown.

**Optimising before measuring is the error this project has made in other forms
all day** (a bar calibrated on the wrong quantity, a baseline lifted from an arm
that does not pay the cost). So: profile first, and commit the prediction so the
profile can contradict it.

## Prediction (locked, before running cProfile)

Cumulative time in a truncated Borehole ROI-Q10 run, by subsystem:

**P1.** **Decision Transformer training dominates** — more cumulative time than
any other single subsystem, and >= 40% of total. Grounds: it is a transformer
trained for `num_epochs=10` on **every** BO iteration, ~115 iterations per run,
while the GP has at most ~130 observations, where an O(n^3) Cholesky is
negligible.

**P2.** **ROI rejection sampling is second**, >= 10% of total, and its cost is
concentrated in `hf_posterior` calls on the 600-point candidate pool rather than
in the sampling itself. Grounds: `n_draws` ~3.5 at q=0.10 means the pool is built
~3.5 times per construction, each requiring a full posterior evaluation.

**FALSIFIED** if DT training is not the largest single consumer. Partitioned:
either it is the largest or it is not.

## Why this is worth a locked prediction rather than "just profile it"

If P1 holds, the speed direction is **DT training cost** — fewer epochs, early
stopping, incremental training, cached rollouts — and the ROI's 1.2-1.7x overhead
is a side issue.
If P1 fails and the GP or the ROI dominates, the speed direction is **the
candidate pool** — and the maths tricks available there (incremental Cholesky,
rank-one posterior updates, reusing draws across iterations, quasi-random pools)
are entirely different work.

**The prediction decides which body of optimisation is worth starting**, so
getting it wrong is informative rather than embarrassing.

## Constraint on anything that follows

**Any optimisation must be gated on bit-identity**, the same standard as h136:
identical `fid`, `x`, `y` at every query against a stored trace. A speed-up that
changes the trajectory is not a speed-up, it is a different method. Optimisations
that cannot meet that bar (e.g. changing the number of epochs) must be measured on
regret as well, at n>=5, and are a *method* change, not an efficiency change.

## Method

`cProfile` on a truncated run (reduced cost budget) of Borehole ROI-Q10 seed42.
Truncation changes absolute times but not the relative shares, which is what P1
and P2 are about. The truncated budget is recorded with the result.
