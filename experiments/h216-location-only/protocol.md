# h216 — Let the random half teach LOCATION only, not fidelity

Registered BEFORE any h216 run. CONFIRMATORY. One new config knob
(`fid_loss_policies`, default None = bit-identical; identity gate PASS at
122.29066752728207).

## The decision this is meant to settle

MIXR helps Borehole (−4.42 vs CTRL-K1, 5/5) and hurts Hartmann (+5.26, 4/5), so the
registered "≤ CTRL-K1 on **both** benchmarks" rule is not met and there is no single
configuration to keep. This arm is the one intervention the evidence actually points at.

## The observation

The same intervention shifts real-query fidelity toward HF on **both** benchmarks:

| bench | HF:LF cost | CTRL-K1 LF | MIXR-K1 LF | shift | regret change |
|---|---|---|---|---|---|
| Borehole | **2:1** | 0.261 | 0.125 | **−0.136** | **−4.42 (helps)** |
| Hartmann | **8:1** | 0.744 | 0.649 | **−0.095** | **+5.26 (hurts)** |

Arithmetic on the budget: on Hartmann, moving 9.5 of every 100 queries from LF (cost 1) to
HF (cost 8) spends ~66 extra cost units out of a 200 budget — **about a third of the
budget**. On Borehole the same-sized shift at 2:1 costs ~7%. One mechanism, opposite
economics.

## Why the fidelity shift is *unearned*

The random teacher picks its fidelity from `Bernoulli(random_p_hf)` — it carries **no
information about which fidelity is right**. Training the fidelity head on it is noise
injection. Its **locations**, by contrast, do carry the ROI's spread, which is the part
that plausibly helps (MIXR-K1 raises query dispersion 0.237 → 0.276 on Borehole, +16%,
vs +1% on Hartmann).

So: keep the random half in the **location** loss, drop it from the **fidelity** loss.
`fid_mask` zeroes the per-token fidelity BCE on rollouts whose policy is not in
`fid_loss_policies`; the location loss is untouched.

## Arms — 10 runs, CTRL-K1 base (K=1, `mes_entropy`, ROI-Q10)

| arm | bench | seeds | config |
|---|---|---|---|
| **LOC-B** | Borehole | 42–46 | MIXR-K1 + `fid_loss_policies=('mes',)` |
| **LOC-H** | Hartmann | 42–46 | same |

References, same seeds: CTRL-K1 **11.59** / **5.93**; MIXR-K1 **7.17** / **11.19**.

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-H (the point of the arm):** LOC-H within the band of CTRL-K1-H (5.93), i.e. the
  Hartmann harm was the unearned fidelity shift and removing it removes the harm.
- **P-B:** LOC-B still beats CTRL-K1-B by > 1.26 on ≥ 4/5. **This is the real test of the
  mechanism.** If the Borehole gain survives with the fidelity head trained on MES only,
  the random half's contribution is its **locations** — the first positive identification
  after volume, boundary geometry, fidelity composition, RTG conditioning and the window
  were each ruled out. If the gain vanishes, the Borehole gain *was* the fidelity shift,
  and MIXR reduces to "query HF more often when HF is cheap" — a much smaller claim.
- **Both hold ⇒ the decision rule is met** and LOC is the first single configuration that
  is at least as good as CTRL-K1 on both benchmarks. That, and only that, reopens the
  four-benchmark rerun with Currin and Ackley as genuine held-out tests.
- **Ordering, so it can be wrong:** LOC-B < CTRL-K1-B and LOC-H ≈ CTRL-K1-H.

## What each outcome RETRACTS

- **P-B fails (gain vanishes)** → retracts the "location/coverage" reading of MIXR that
  `why-full-random-failed.md` argues for; the effect is fidelity economics, and it is
  Borehole-specific by construction.
- **P-H fails (harm persists)** → the Hartmann harm is not the fidelity shift, and the
  cost-ratio account above is wrong despite its arithmetic fitting.
- **Both fail** → the random half is doing something not captured by the
  location/fidelity split at all.

## Stage 0 (smoke)

(a) both arms build and run at BUDGET=30; (b) `fid_mask` exists, is False on exactly the
random rows and True on the MES rows, and `valid_mask` is unchanged; (c) with
`fid_loss_policies=None` the default path is bit-identical (identity gate, already PASS).
A silent no-op is the failure mode.

## Compute

10 runs, cap 15, queued behind h214. Hartmann first. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
