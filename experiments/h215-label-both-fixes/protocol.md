# h215 — The label with BOTH deployment defects removed

Registered BEFORE any h215 run. CONFIRMATORY. No core change: both flags
(`rtg_target_schema='percentile'`, `rtg_rng_parity=True`) already exist and are gated;
identity gate PASS at 122.29066752728207.

## The question, and why it is the last one on this thread

`terminal_improvement` looked harmful on Hartmann. Two *deployment* defects were found
behind that, each measured separately as a one-factor test:

| configuration | Hartmann | isolates |
|---|---|---|
| CTRL-K1-H — `mes_entropy`, floored | **5.93** | the reference |
| K1-TI-H — label, floored, RNG unmatched | 11.73 | h211's original arm |
| TI-PAR-H — label, floored, **RNG matched** (h213) | 8.99 | −2.74 from stream matching |
| TI-PCT-H — label, **percentile**, RNG unmatched (h212) | 6.46 | −5.27 from un-pinning |

Neither fix alone reaches the control, and the two have never been applied together. Until
they are, we cannot say whether the reward *definition* costs anything at all, or whether
the whole apparent penalty was how it was deployed.

## Arm — 10 runs

| arm | bench | seeds | config |
|---|---|---|---|
| **TI-BOTH-H** | Hartmann | 42–46 | CTRL-K1 + `terminal_improvement` + `percentile` + `rtg_rng_parity` |
| **TI-BOTH-B** | Borehole | 42–46 | same on Borehole |

Borehole is included because the label was neutral there under both prior treatments
(h207 +1.23; h213 −1.06). If TI-BOTH-B *moves* on Borehole, the percentile schema is doing
something benchmark-specific and the Hartmann reading needs qualifying — the same caution
that P-NEUTRAL's failure in h212 already flagged (q=90 asks for less, and on the unsigned
label that cost +3.52).

## Prediction, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-BOTH-H (registered in h212's analysis, before this arm existed):**
  |TI-BOTH-H − CTRL-K1-H (5.93)| ≤ 1.26. The label's Hartmann penalty was **entirely its
  deployment** — the pinned target and the unmatched stream — not the reward definition.
- **P-BOTH-B:** |TI-BOTH-B − CTRL-K1-B (11.59)| ≤ 1.26, i.e. still neutral on Borehole.
- **Ordering, so it can be wrong:** TI-BOTH-H < TI-PCT-H (6.46) < TI-PAR-H (8.99) <
  K1-TI-H (11.73), i.e. the two fixes compose.

**Honest note on my own record here.** My last registered lean on this thread — that the
label would be inert once RNG was matched (h213) — was **wrong**; the label cost +3.06. I
am nonetheless registering P-BOTH-H, because h212 showed the schema alone recovers 5.27 and
lands within 0.53 of the control, which leaves little room for a residual label cost. If
TI-BOTH-H comes in above CTRL-K1-H by more than the band, the reward definition itself is
harmful on Hartmann and that is a finding about the reward, not its plumbing.

## What each outcome RETRACTS

- **P-BOTH-H supported** → retracts "the `terminal_improvement` label is harmful on
  Hartmann" in all its forms (h211 P-LABEL, h213 P-LABEL-H). The label is fine; its
  deployment was not. Every h207–h211 Hartmann arm using it was measuring the plumbing.
- **P-BOTH-H failed** → the two fixes do **not** compose, and there is a residual label
  cost. Then h213's +3.06 is the live estimate and the reward definition is implicated.
- **P-BOTH-B failed** → the percentile schema is benchmark-specific; the Hartmann result
  above would need re-reading against a schema that does not move Borehole.

## Compute

10 runs, cap 15. Queued behind h214's 15 by the launcher's slot guard; Hartmann first.
1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.

## STOOD DOWN before launch (2026-09-28) — the label it repairs is being abandoned

Nothing ran. On review prompted by the user: **`terminal_improvement` has never produced a
benefit on any benchmark**, and the case for repairing it is weak next to the case for
dropping it.

| test of the label, in isolation | result |
|---|---|
| h207 P-NIR — Borehole, label alone (RNG unmatched) | +1.23, inside band — neutral |
| h213 P-LABEL-B — Borehole, **RNG matched** | −1.06, inside band — neutral |
| h213 P-LABEL-H — Hartmann, **RNG matched** | **+3.06, 0/5 — harmful** |

Neutral at best, harmful at worst, and it cost two repair experiments (h212, h213) plus
this one. Meanwhile `mes_entropy`:

- is what the **best configuration on both benchmarks** uses (CTRL-K1: 11.59 / 5.93);
- is what the **best deployable result** uses (MIXR-K1 Borehole 7.17, h211);
- has an RTG that is already a **proper future-only return-to-go** — `log b_τ − log b_T`
  telescopes to the remaining information gain and decreases over τ, matching DT's
  `R_t = Σ_{t'≥t} r_{t'}` in shape as well as in name;
- needs **no** schema fix — its target already varies (82–110 distinct values), and h212's
  P-NEUTRAL showed the percentile schema *hurts* it by +3.52.

So the pinned-target defect, the RNG-parity defect and the flat-RTG credit problem are all
consequences of a label we adopted for a principled reason (match the scored metric) that
never paid off. h215 would have told us whether that label's Hartmann penalty was
deployment or definition — a question about a label we are not going to use. Confirmatory
of a retraction, not decision-relevant.

**Kept on the record, not deleted**: the flags it would have exercised
(`rtg_target_schema`, `rtg_rng_parity`) are committed, gated, identity-gate-clean, and
would be needed by any future signed-reward label.
