# Protocol — H9: is RTG inert, or merely starved?

**Locked before running.** Results commit must be separate.

## The confound this resolves

Every RTG result so far is ambiguous between two very different claims:

1. **the network ignores RTG** (an architecture/training failure), and
2. **RTG carries too little variation to be worth using** (a schema failure).

They are confounded because `rtg_target` is **structurally clamped to
[0.500, 1.000]** — CV 0.232 — by `max(batch_max, alpha_rtg * running_max)` acting
on a batch-max-normalised rtg with `alpha_rtg = 0.5`. H8 showed the policy does
not respond *within* that band (argmax moved 0/12, corr 0.99993), but a policy
cannot be blamed for ignoring a signal that barely moves.

This is the last question standing between the current results and a defensible
write-up: a paper claiming "the DT ignores its return conditioning" would be
overstating what the evidence supports.

## Hypothesis

**H9**: RTG-insensitivity is a property of the network, not of the band width.
Widening the realised band will NOT make RTG move the decision.

## Design — one variable

`alpha_rtg` only. Everything else is the current default.

- **NARROW** (control, current default): `alpha_rtg = 0.5` -> target in [0.5, 1.0], a 2x band
- **WIDE**: `alpha_rtg = 0.1` -> target in [0.1, 1.0], a **10x band**

For each arm: train normally, record the **realised** `rtg_target` support, then
sweep the target across *that arm's own realised band* (6 points, 12 resampled
candidate pools) and measure argmax movement — the H8 measurement, applied to
each arm's actual distribution rather than a fixed set of multipliers.

Measuring each arm on its *own* support is the point: it keeps the comparison
in-distribution for both, which is exactly the error H8 caught me making.

## Locked predictions

1. **Primary**: WIDE argmax movement stays **< 20%** of sweeps (i.e. widening the
   band does not rescue RTG -> the network, not the schema, is responsible).
2. WIDE's realised band is materially wider than NARROW's (a manipulation check;
   if it is not, `alpha_rtg` is not the lever I think it is and the experiment
   is void).

Prediction 1 again bets on my own prior conclusion holding.

## What each outcome means

- WIDE movement < 20% -> **H9 supported**: the insensitivity is the network's.
  "MF-DRO re-fits rather than conditions" becomes defensible as written, and the
  confound is closed.
- WIDE movement high -> **H9 refuted, and this is the more interesting result**:
  RTG was *starved, not ignored*. The fix is then a one-line schema change rather
  than an architecture change, and every conditioning-side refutation (H4, H5)
  needs re-reading as "tested under a starved signal".
- Manipulation check fails -> experiment void; report and do not interpret
  prediction 1.

## Compute

2 arms x 1 training run + probes, single process each.
Launched alongside: 4 LIVE-ext workers + 5 H7 workers + 2 = 11 <= 15.
