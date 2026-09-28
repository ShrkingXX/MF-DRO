# h213 analysis — CONFIRMATORY. 10/10 finals, 0 tracebacks. The first true one-factor label test.

Frozen metric imported from h83's `grid`; finals only; endpoint only.

## Result

| arm | bench | regret | vs CTRL-K1 |
|---|---|---|---|
| **TI-PAR-H** (`terminal_improvement`, RNG matched) | Hartmann | **8.99** | **+3.06** (se 1.55, **0/5**) |
| TI-PAR-B | Borehole | 10.53 | −1.06 (se 1.08, 3/5) — inside band |
| CTRL-K1 (`mes_entropy`) | H / B | 5.93 / 11.59 | — |
| K1-TI-H (same label, RNG **un**matched, h211) | Hartmann | 11.73 | +5.80 |

## Scoring

**P-LABEL-H — SUPPORTED. My registered lean was WRONG.** I predicted the label would be
**inert** once RNG was matched, reasoning from the measured RTG sensitivity (1.1–1.4×) that
a channel barely moving the decision head could not cost 5.8 points. The label costs
**+3.06 on 0/5** with the random stream held identical. The channel does matter more than
the sensitivity probe suggested.

**P-LABEL-B — SUPPORTED.** −1.06 (3/5), inside the band. Neutral on Borehole, reproducing
h207's finding — and now for the first time as a genuine one-factor comparison.

## What this settles about h211

h211's P-LABEL measured **+5.80**; the RNG-matched value is **+3.06**. So roughly **half of
h211's label penalty was random-stream divergence** and half is the label. h211's P-LABEL is
therefore **partially retracted**: its direction and existence stand, its magnitude does not,
and it was never a one-factor test.

By extension the same correction applies to every earlier cross-label comparison — h207's
P-NIR (+1.23, Borehole) and h210's CTRL-K1-H vs NIR-H — which were run without RNG matching.
h207's Borehole neutrality survives, since h213 reproduces neutrality there under matching.

## Method note worth keeping

Stage 0 v1's gate demanded the parity run reproduce the control's **first real query**. That
was wrong by construction: with RNG matched the labels generate identical rollouts and attach
different RTG values, which are training inputs, so the trained model *should* diverge — that
divergence is the effect under test. Gate v2 required bit-identical rollout data with
differing `rtg`, which is the one-factor condition stated correctly. Direct measurement
confirmed parity: 963 Thompson calls and RNG fingerprint +1.250309758 under both labels.
