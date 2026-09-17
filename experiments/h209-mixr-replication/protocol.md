# h209 — Does MIXR replicate, and is it just more data?

Registered BEFORE any h209 run. CONFIRMATORY. Core code unchanged from h207 (49fbcd1);
only workers and the analysis are new, so Stage 0 is a smoke check, not a gate re-run.

## What h207 left open

MIXR (20 MES + 20 random rollouts per member, `terminal_improvement` label, K=8, no
positional embedding, ROI-Q10) reached **6.22** on Borehole 42–46 vs NIR **12.62** — −6.40
(se 2.33, 4/5), matching MF-MES (6.40). First deployable arm to do so. But: one benchmark,
one seed set, mechanism unknown, and one **unregistered confound** — MIXR trains on 120
trajectories per iteration, NIR on 60.

## Arms — 25 runs, launched 15-wide longest-first, remainder queued

| arm | benchmark | seeds | per member | purpose |
|---|---|---|---|---|
| **NIR120-B** | Borehole | 42–46 | 40 MES | data-quantity control (120/iter, MES only) |
| **MIXR-H** | Hartmann_6D | 42–46 | 20 MES + 20 random | second benchmark |
| **NIR-H** | Hartmann_6D | 42–46 | 20 MES | its matched control |
| **MIXR-B47** | Borehole | 47–51 | 20 MES + 20 random | fresh seeds |
| **NIR-B47** | Borehole | 47–51 | 20 MES | its matched control |

Hartmann: d=6, cost ratio 8:1 (vs Borehole's 2:1), init 6 HF + 45 LF, real horizon ~159
queries. Reference (h83/h84, seeds 42–46): ROI-Q10 K=1 5.93 (sd 3.26), MF-MES 6.62,
MF-DRO no-ROI 7.99. Note ROI-Q10 already beats MF-MES on Hartmann — less headroom.

## Predictions, committed now (band ±1.26 rel%; endpoint only; no p-values at n=5)

- **P-Q (quantity):** |NIR120-B − NIR-B(12.62)| ≤ 1.26 **and** MIXR-B(6.22) − NIR120-B
  < −1.26. Doubling MES rollouts from the same posterior adds little; the random half is
  doing something MES data cannot. **If instead NIR120-B lands within the band of MIXR,
  the h207 gain is RETRACTED as a data-volume effect** and "diversity helps" is withdrawn.
- **P-H (Hartmann):** MIXR-H − NIR-H < −1.26 on ≥ 4/5. Lean: replicates, smaller margin.
  **Discriminating outcome, registered:** if MIXR-H is *worse* than NIR-H and its LF
  fraction drops as it did on Borehole (0.44 → 0.27), the mechanism is the random half
  retraining the fidelity head toward HF — helpful at 2:1, costly at 8:1 — and the gain
  is Borehole-specific. That would be the h202 confound in full.
- **P-B47 (fresh seeds):** MIXR-B47 − NIR-B47 < −1.26 on ≥ 4/5.
- **The gain is called REAL only if all three hold.** Any one failing is reported as the
  failure it is; two of three is "partial, unexplained".

## What each outcome RETRACTS

- P-Q fail → h207's headline. The right control was never run; MIXR = NIR with more data.
- P-H fail with the LF signature → the gain is a fidelity-mix artefact tied to Borehole's
  cost ratio; "add random rollouts" is not a method.
- P-B47 fail → seed 46's +1.7 was the honest seed; the 4/5 was luck at n=5.
- All three hold → the first replicated deployable improvement in the program, mechanism
  still open (state coverage vs early RTG selection vs fidelity head), and the full-run
  question is reopened with a real candidate.

## Stage 0 (smoke, not a gate re-run)

Core unchanged since h207's gates. Smoke: (a) MIXR and NIR run on Hartmann under this
config for BUDGET=30 without error, with the expected batch composition (40/member for
NIR120, 20+20 for MIXR); (b) NIR120's batch is exactly 120 trajectories. ~10 min.

## Compute

25 runs, cap 15. Launch order longest-first: Hartmann arms (~159 real queries, 8:1) first,
then NIR120-B, then the Borehole-47 pair as slots free. Threads 1/worker. h208 (other
session) may still hold slots; `launch.sh` respects the cap.

## Evaluation

Frozen: final simple regret, rel% of |optimum| @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
