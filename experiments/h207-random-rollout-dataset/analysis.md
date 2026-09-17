# h207 analysis — CONFIRMATORY on the arms; the registered mechanism rule decides the reading

15/15 finals, 0 tracebacks. Frozen metric imported from h83's `grid`; finals only.

## Result

| arm | final regret | vs NIR | LF frac |
|---|---|---|---|
| NIR  20 MES/member, `terminal_improvement` | 12.62 | — | 0.438 |
| **MIXR  20 MES + 20 random** | **6.22** | **−6.40** (se 2.33, 4/5) | 0.270 |
| MIXO  20 MES + 20 ORACLE (ceiling) | 0.00 | −12.62 (se 1.94, 5/5) | 0.217 |
| h206N (mes_entropy control) | 11.39 | | 0.462 |
| CTRL-K1 | 11.59 | | 0.261 |
| MF-MES (h83, strongest baseline) | 6.40 | | |

## Scoring

**P-NIR — SUPPORTED.** NIR − h206N = +1.23 (se 2.25, 2/5), inside ±1.26. The label alone
does not move the endpoint.

**P-MIXO — regret criterion MET (0.00, 5/5), mechanism criterion FAILED.** Registered
rule (Amendment 8): RTG-selection only if RTG-sensitivity ≥ 2× NIR AND > BTG-sensitivity.
Measured: RTG 1.40× (last 30 iters), BTG negligible (0.002–0.017). The early-iteration
probe explains the endpoint without any conditioning token:

| | |x − x\*|∞ at rtg = −1 | at rtg = +1 | (first 10 iterations, before the GP knows x\*) |
|---|---|---|---|
| NIR | 0.669 | 0.666 | |
| MIXO | **0.182** | **0.156** | |

MIXO's output is shifted 0.5 toward x\* *unconditionally*; RTG moves it 0.03. The oracle
half is a constant regression target at x\*; under `loc_loss='mse'` the location head's
mean is dragged toward it. On Borehole the midpoint between a MES action and a corner
optimum is an excellent query, so the drag reaches x\* within ~10 real queries and the
GP then learns it. **This is the loss-side analogue of h201A's positional shortcut.
MIXO is NOT evidence that the DT reads RTG.** Registered P-MIXO-blur had the wrong sign
(I predicted blur → worse); the registered mechanism rule is what caught it.

**P-MIXR — registered lean REFUTED on regret.** I registered "not selective, MIXR − NIR >
+1.26". Observed −6.40, MIXR better on 4/5 (per-seed −8.5 −4.8 −12.1 −8.2 +1.7). MIXR
sits at MF-MES's level, which no deployable MF-DRO arm has reached. Mechanism NOT
established:
- RTG-sensitivity 1.11× NIR (last 30; fails 2×); 3.5× in the first 30 (EXPLORATORY)
- BTG negligible
- |x − x\*| unchanged vs NIR (0.65 vs 0.67) — no location pull, unlike MIXO
- LF fraction 0.438 → 0.270 — the fidelity mix moved (h202's confound, in reverse); but
  CTRL-K1 has the same mix (0.261) at 11.59, so mix alone does not explain 6.22
- **MIXR trains on 120 trajectories/iteration vs NIR's 60 — an unregistered
  data-quantity confound.** MIXO shares it.

Hypotheses for MIXR, all open: (i) data quantity; (ii) state coverage — random rollouts
visit states MES never does, regularising the state encoder; (iii) early RTG selection
(the 3.5× first-30 reading); (iv) fidelity-head retraining by the random half's p_HF=0.5.

## What this RETRACTS and what it leaves

- **Retracted: h201A's 0.00 as evidence the window reads history.** With the positional
  shortcut removed (h206), the oracle still reaches 0.00 — by a different shortcut (loss
  mean toward a constant target). Neither oracle result says anything about RTG.
- **The single-teacher account of RTG inertness is NOT confirmed.** MIXO was the easiest
  possible selectivity test (identical τ=0 state, two actions, cleanly separated labels)
  and the DT did not select by RTG. Q2 reopens on the architecture/training side: the
  location head under MSE averages a bimodal target rather than conditioning on the
  token that disambiguates it.
- **What stands:** a deployable arm beat every control by more than the band on 4/5
  seeds. Whether it is real, and why, is the next experiment.

## Next (registered separately)

Replicate MIXR on fresh seeds (Borehole 47–51) and a second benchmark (Hartmann 42–46);
add NIR-120 (40 MES/member) as the data-quantity control. No mechanism claim before that.
