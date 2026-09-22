# h210 — The missing Hartmann control, and the principled version of MIXR

Registered BEFORE any h210 run. CONFIRMATORY. Core code unchanged since h207 (49fbcd1);
every arm is a config of existing, gated flags (`loc_loss` is H102's, in the code since
then). Stage 0 is a smoke, not a gate re-run.

## Where this comes from

h209 established that MIXR's Borehole gain is real (10/10 seeds) and does not transfer to
Hartmann, and the traces say why: the random half **shifts the MSE regression mean** — the
location head toward box faces, the fidelity head toward HF. Borehole rewards both (corner
optimum, 2:1), Hartmann punishes both (interior, 8:1). That is MF-DRO's known boundary
aversion corrected by accident. H102's `loc_loss='l1'` exists for exactly this: the median
sits *at* a bound where the mean is pulled inward — and it carries no fidelity cost.

h209 also surfaced an unregistered warning: NIR-H (K=8, no positional embedding,
`terminal_improvement`) = **9.75** on Hartmann vs the last full run's ROI-Q10 K=1 = **5.93**
on the same seeds. The K=8 line was validated on Borehole only. That comparison conflates
K, label and code drift since h84 — it needs its own control before anything is tuned.

## Arms — 20 runs, 15-wide, longest-first, remainder queued

| arm | bench | seeds | config | purpose |
|---|---|---|---|---|
| **CTRL-K1-H** | Hartmann | 42–46 | h84's ROI-Q10 exactly: K=1, `mes_entropy`, current code | the missing control |
| **L1-NIR-B** | Borehole | 42–46 | NIR + `loc_loss='l1'` (20 MES, K=8, nopos, `terminal_improvement`) | principled boundary fix |
| **MIXR-P25-B** | Borehole | 42–46 | MIXR with `random_p_hf=0.25` (was 0.5) | separates location-shift from fidelity-shift |
| **L1-NIR-H** | Hartmann | 42–46 | NIR-H + `loc_loss='l1'` | does the median head cost anything where there is no boundary to gain? |

Controls in hand: NIR-B 12.62, MIXR-B 6.22, NIR-H 9.75, MIXR-H 12.08 (all same seeds).

## Predictions, committed now (band ±1.26; endpoint only; no p-values at n=5)

- **P-CTRL:** CTRL-K1-H within the band of 5.93 (|Δ| ≤ 1.26). Then the ~4-point Hartmann
  gap is the K=8/label configuration, not code drift, and **the window line must be
  re-examined on Hartmann before it is called "parity"**. If CTRL-K1-H instead lands near
  9.75, something in the core since h84 hurts Hartmann at K=1 and a bisection is next.
  Lean: 5.93 — every core change since h84 is behind a flag defaulting off. This is the
  arm whose failure would matter most.
- **P-L1-B:** L1-NIR-B − NIR-B < −1.26 on ≥ 4/5, with dims-on-face up. The median head
  recovers a substantial part of MIXR's Borehole gain with MES data alone. Lean: ~8.
- **P-P25:** MIXR-P25-B's real LF fraction within 0.05 of NIR-B's (0.44) **and** regret
  < NIR-B − 1.26. Then the location shift is a real mechanism independent of fidelity.
  If regret returns to NIR's, the Borehole gain was the fidelity head after all. Lean:
  partial (~9), LF ~0.38.
- **P-L1-H:** |L1-NIR-H − NIR-H| ≤ 1.26. No boundary to gain, no fidelity to lose; L1
  should be neutral on an interior optimum. If L1-H is worse by > 1.26 the median head
  has its own cost and is not a free replacement.
- **The claim "L1 is the principled MIXR" is made only if P-L1-B holds AND P-L1-H holds.**

## What each outcome RETRACTS

- P-CTRL fail (near 9.75) → the code has drifted on Hartmann since the last full run;
  every K=8 result on Borehole stands but nothing can be said about generality until a
  bisection finds the change.
- P-L1-B fail → the mean-vs-median account of MIXR is wrong; the random half does
  something L1 cannot reproduce, and the boundary-coverage diagnostic was correlation.
- P-P25 fail (regret returns to NIR) → the Borehole gain was the fidelity head; the
  boundary story is retracted.
- P-L1-H fail → L1 is a Borehole-specific fix too.

## Stage 0 (smoke)

(a) each arm builds and runs for BUDGET=30 on its benchmark; (b) `mf.dt.loc_loss == 'l1'`
on the L1 arms and `'mse'` on the others (a silent default is the failure mode);
(c) CTRL-K1-H has `inference_context_k == 1`, `rollout_reward == 'mes_entropy'`,
`disable_position_embedding == False`; (d) MIXR-P25's random rollouts have HF fraction
≈ 0.25 in the batch. ~10 min.

## Compute

20 runs, cap 15. Order: Hartmann arms first (longer), then Borehole. 1 thread/worker.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.

## AMENDMENT 1 (before launch) — smoke v1 FAIL corrects the fidelity mechanism; arm re-specified

**Smoke v1: FAIL** on the MIXR-P25 check (log kept as `logs/smoke_v1_FAIL.log`):
`random HF frac 0.37 (want ~0.25), MES 0.72`. The 0.37 is `minimum_hf_fraction=0.25`
flooring the running fraction inside each rollout. The 0.72 is the finding: **MES's own
training rollouts are 72% HF**, so the random half at p_HF=0.5 (≈54% HF) was *diluting*
HF in the training mix — while MIXR's real queries went *more* HF (LF 0.44 → 0.27 on
Borehole, 0.56 → 0.44 on Hartmann). The real-query fidelity moved opposite to the
training-mix shift.

**Correction, recorded here and in findings.md:** the sentence "the random half's 50% HF
moves the fidelity head toward HF" is WRONG in its causal form. The LF-drop observable on
both benchmarks stands; its cause is not the random half's fidelity composition. Live
candidates: the fidelity head's state-dependence (MIXR's boundary-shifted queries produce
posteriors where the head prefers HF), or the `minimum_hf_fraction` / `p_pred` interaction
at inference. Neither is tested here.

**Arm re-specified: MIXR-P72-B** — `random_p_hf = 0.72`, matching the MES rollouts'
measured HF fraction, so the training mix's fidelity composition is UNCHANGED by the
random half and any real-query LF change must come from location/state effects. Smoke
check becomes |HF_random − HF_MES| < 0.08 on an actual batch (mix-matched), not a fixed
target. **P-P72** replaces P-P25 with the same logic: if regret stays < NIR − 1.26 with
the mix matched, the location shift is a real mechanism independent of the fidelity
composition; if regret returns to NIR's, the gain rode on the random half's fidelity
composition after all (which, given 0.72 vs 0.54, would mean *diluting* HF helped — a
different story from the one retracted above).
