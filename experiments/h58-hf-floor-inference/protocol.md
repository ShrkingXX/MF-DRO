# H58 — the HF floor exists only in simulation; does enforcing it at inference remove the stall?

**CONFIRMATORY. Protocol committed before any run.**

## The defect, located in code

`minimum_hf_fraction` (default 0.25) is applied at exactly one place,
`mf_dro.py:1373`, inside `simulate_mf_trajectory`:

    hf_steps_so_far = sum(1 for e in actions_ell if e == 1)
    if (tau > 0 and hf_steps_so_far / tau < minimum_hf_fraction and ell_tau == 0):
        ell_tau = 1  # forced HF to ensure training diversity

`grep` over everything after the real run loop returns **nothing**. The real
query's fidelity is `ell ~ Bernoulli(p)` from `fidelity_head`, overridden only by
`real_hf_warmup=2` for the first two iterations. After that it is unconstrained.

So the DT is TRAINED on rollouts where HF is forced to at least 25%, and then
RUN with no floor at all. A train/inference mismatch, and the comment on the
enforcement line says plainly that its purpose was training diversity.

## Why this matters, measured not assumed

The incumbent updates ONLY on HF queries. `src/analysis/stall_diagnose.py` over
h57's live traces attributes each incumbent stall:

| benchmark | seed | stall | HF during stall | cause |
|---|---|---|---|---|
| Hartmann 6D | 46 | 17 | **0** | HF-STARVED |
| Currin 2D | 46 | 7 | **0** | HF-STARVED |
| Hartmann 6D | 44 | 19 | 13 (68%) | MISDIRECTED |
| Hartmann 6D | 48 | 9 | 8 (89%) | NEAR-MISS |
| Currin 2D | 44 | 11 | 3 (27%) | NEAR-MISS |

Two of five stalls have zero HF queries across the whole stall window. Those
incumbents were mathematically incapable of moving; no search-quality
explanation is needed or admissible for them.

This also reframes h45 seeds 49 and 50, which had ZERO improvements over 144 and
74 all-distinct queries and were read as an aimless-search pathology. If they
were LF-heavy, "aimless" was never the right description.

## Design

Hartmann 6D and Currin 2D, seeds 44, 46, 48, cost budget 200, regression head,
`initial_hf`/`initial_lf` per h57 (6/45 and 5/15). One variable:

| arm | real-query fidelity |
|---|---|
| **FREE** | current behaviour: `ell ~ Bernoulli(p)`, no floor (h57's MF-DRO arm) |
| **FLOOR** | mirror line 1373 at inference: if `n_HF/t < 0.25` and the DT chose LF, force HF |

12 jobs.

**h57 IS NOT TOUCHED.** `src/policy/mf_dro.py` stays frozen: the floor is applied
in H58's worker by wrapping `mf.dt.propose_mf`, the same monkeypatch pattern h31
and h57's MF-MES arm already use. Separate directory, separate results, its own
commit hash recorded per result.

## Locked predictions

1. **PRIMARY**: FLOOR has strictly fewer HF-STARVED stall windows than FREE
   (`stall_diagnose.py`, same classifier, same thresholds). This is close to
   mechanical and is the sanity check, not the result.
2. **THE ACTUAL QUESTION**: FLOOR's mean final HF simple regret is LOWER than
   FREE's, paired, on >= 4 of 6 (benchmark, seed) cells.
3. **HARMFUL**: FREE better on >= 4 of 6. Forcing HF spends 8x the cost per query
   on Hartmann, so a floor can buy fewer total queries and lose. This outcome is
   live and would say the fidelity head's LF preference is right and the stall is
   the price of a correct decision.
4. **NULL**: 3-3, or differences inside the seed spread.

Prediction 3 is the one I expect to be underrated: HF costs 8 units on Hartmann
against LF's 1, so a 25% floor is a large budget reallocation, not a free fix.

## What this cannot settle

n=3 per benchmark, no p-values. Two benchmarks, not three -- Borehole is omitted
because at c_H=2 its HF is comparatively cheap and its h57 cells are already
running 87-100% HF, so the floor would almost never bind there.
