# H104 — does the pre-registered success test survive the seed count it registered?

LOCKED BEFORE ANY RUN. ID claimed via `tools/claim_id.sh`.

## Why

A peer session established, and I verified against the file, that PROTOCOL.md
registers **two** baselines (MF-MI-Greedy, MF-GP-UCB) on **one** benchmark
(Hartmann 6D) at **10 seeds**, with `Amendments: None.` Against that, MF-DRO's
registered success test passes — `mean+SE 10.60 < best-baseline mean-SE 35.55`.

But h83 ran **5** seeds, not 10. This project has repeatedly shown n=5 cannot
characterise a paired difference here: a paired sd of 0.45 on one seed set became
7.45 on another. **A pass at half the registered sample is not the registered
result.** This experiment supplies the other half.

## What it costs, and why that makes it obligatory rather than optional

Almost nothing. The two registered baselines are the cheapest methods in the
comparison — h83 wall-clock: MF-GP-UCB **0.0 min**, MF-MI-Greedy **0.1-0.2 min**.
Everything else needed already exists:

  - MF-DRO at seeds 52-56 — h89's `CONTROL` arm, spec-matched to h83's MF-DRO
    (n_hf=6, n_lf=45, budget=200, no ROI; verified, and `roi_summary` absent).
  - MF-MES at seeds 52-56 — h91.

**10 runs, ~2 minutes of compute**, to settle whether the project's one passing
pre-registered claim holds at the sample it registered. Leaving that unmeasured
while it is this cheap would be indefensible.

## Design

| | |
|---|---|
| benchmark | Hartmann_6D (the protocol's own, singular) |
| new runs | MF-MI-Greedy and MF-GP-UCB at seeds 52, 53, 54, 55, 56 |
| reused | MF-DRO (h89 CONTROL) and MF-MES (h91) at the same seeds; h83 for 42-46 |
| metric | the protocol's: final simple regret at matched cost, mean +/- SE |
| test | the protocol's: `MF-DRO mean+SE < best-baseline mean-SE`, pooled n=10 |

## Gate, before any verdict is read

The reused arms come from other experiments. Per this project's own code-drift
audit, a paired comparison is valid only if both arms ran on commits with **no
behavioural diff** in `src/`, `dro_runner.py`, `benchmarks.py`. The analysis
checks that across every contributing commit and **refuses to print the test**
if any pair differs. If it fails, the arms must be re-run rather than reused.

## Predictions

**P1.** The registered success test PASSES at n=10. Registered **POSITIVE and
near-certain**, stated as such rather than dressed up: MF-DRO scores 7.99 at
42-46 and 2.66 at 52-56, so its pooled mean *improves* to roughly 5.3, while the
registered baselines are the two weakest methods in the comparison. A prediction
this safe is worth registering only because the alternative — quietly not
checking — is how the half-sample pass got reported in the first place.

**P2.** MF-DRO still does not beat MF-MES at n=10. Registered **POSITIVE**.

**P3.** The registered baselines remain far behind at the new seeds, improving on
their own initial design in single digits out of ~20-25 HF queries. Registered
**POSITIVE**, and it is the qualification that matters: h103 established this is
a faithful port of the reference's deliberate UCB prior, not our defect, so a
pass against them is real and says little about competitiveness.

## What each outcome means

  - **P1 passes:** the one pre-registered claim this project has now holds at its
    registered sample, and the honest headline is the three-part one — passes the
    registered test, at the registered seed count, against faithfully-weak
    baselines, while not being the best method once MF-MES is included.
  - **P1 fails:** the pass was an artefact of the half sample, and the correction
    a peer session made to the headline needs correcting again in the other
    direction. That would be the more important result.
