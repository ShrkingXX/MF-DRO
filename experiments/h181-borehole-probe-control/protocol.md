# h181 — unstandardised Borehole probe control

**CONFIRMATORY.** Committed before the arm is launched and before any result
exists.

## Why this exists — a gap in h179's design

h179 switched on h178's embedding-response probe so that a null regret result
would be attributable. But **every previously-run probe is on Hartmann**
(h168 PROBE-RANDOM, h177 BTG-RANDOM, h178 EMB-RANDOM); h179 is the first and
only probe on Borehole. So h179 varies **two** things at once relative to any
available comparison — standardisation *and* benchmark — and its measured
responsiveness cannot be attributed to standardisation.

Measured on h179 so far (2 of 5 seeds, real inference state, |x(rtg=0) − x(rtg=1)|):

| seed | mean | median | max | fidelity flips |
|---|---|---|---|---|
| 42 | 0.0961 | 0.0910 | 0.2092 | 20% of iters |
| 43 | 0.1070 | 0.0909 | 0.2830 | 10% of iters |
| **pooled** | **0.1015** | | | |

(real-state and train0-state responsiveness are identical to 4 d.p. on both
seeds — the inference state is bit-identical to a training state, as established.)

## The arm

h179's worker with **exactly one line removed**: `cfg.standardize_conditioning
= True`. Same benchmark (Borehole), same seeds (42–46), same probe values, same
everything else. A one-line diff is the whole point — anything more reintroduces
the confound this arm exists to remove.

## Predictions

- **P1 — standardisation worked in situ.** Control responsiveness falls **below
  0.051** (half of h179's 0.1015). Given the h179 seed spread (0.0961 vs 0.1070,
  sd ≈ 0.008), that is a >6 sd separation and unambiguous.
- **P2 — standardisation did nothing at inference.** Control lands within
  0.051–0.152 (i.e. indistinguishable from h179 at this spread). The 336×
  module-level restoration then does **not** transfer to the running model.
- **P3 — standardisation REDUCED responsiveness.** Control exceeds 0.152.

Gate computed only once **both** arms have 5 seeds. No p-values at n=5; the
threshold is a pre-registered ratio, not a test.

## What this could RETRACT

- **P2 or P3 retracts h179's premise.** findings.md currently reports h178's
  z-scoring result as restoring BTG responsiveness 336×, with the scope
  "module-level upper bound". If the control shows the running model is equally
  responsive **without** standardisation, then h179's probe measured Borehole,
  not standardisation, and the in-situ half of that claim is unsupported — the
  336× stays a module-level statement and nothing more.
- It would also mean h179's regret result (whatever it is) is **not**
  attributable to a responsiveness change, which is precisely what the probe was
  added to guarantee.

## Compute

5 workers × 1 thread. Launched only after h179's last worker exits, so the two
arms do not contend — h179's own numbers are already recorded per-seed, and
contention would not change regret, but matched conditions cost nothing here.

## The exact statistic the gate is registered against

Both arms are measured by `code/responsiveness.py`, so the calibrating number and
the tested number are the same quantity by construction:

```bash
python experiments/h181-borehole-probe-control/code/responsiveness.py
```

statistic = mean over probed iterations of |x(rtg=0) − x(rtg=1)| in the unit box,
at the **real** inference state, pooled over seeds. The 0.1015 that calibrates the
threshold is this statistic on h179's first two seeds; the control is this
statistic on h181's five. Gate verified to partition:

```
tools/check_gate.py --stat "control RTG responsiveness (Borehole, real state)" \
  --pass "<0.051" --falsify ">=0.051"      -> OK, partitions
```

Three-way bands are contiguous and exhaustive: P1 `< 0.051`, P2 `[0.051, 0.152]`,
P3 `> 0.152`.

## CORRECTION to this protocol, made before the control was launched

The ">6 sd" justification above was calibrated on h179's **first two** seeds
(0.0961, 0.1070, sd ≈ 0.008). h179's fourth seed came in at **0.0640**, which
widens the spread substantially:

| n | per-seed | mean | sd |
|---|---|---|---|
| 2 | 0.0961, 0.1070 | 0.1015 | 0.0077 |
| 4 | 0.0961, 0.1070, 0.0640, 0.1104 | 0.0944 | 0.0211 |

**The registered threshold stays at 0.051** — it is a pre-registered ratio (half
of the 2-seed mean) and moving it after seeing more data is exactly what
pre-registration exists to prevent. What changes is the honest description of its
strength: 0.051 sits **≈2.1 sd** below the n=4 mean, not >6 sd. P1 therefore
requires the control to come in ≈46% below h179 — a large effect, but the P1/P2
boundary is materially less crisp than first written.

This correction is recorded rather than silently applied, and is made with the
control at 0/5 results.
