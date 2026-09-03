# h199 — **P2. Even the ORACLE ceiling has no effect.**

**CONFIRMATORY**, 5/5 finals, readout committed before any run finished. Quality
compared ONLY by final simple regret (frozen rel% @ cost 200).

## Result

| arm | final regret |
|---|---|
| h199 ORACLE-lookahead (ceiling) | **10.59** |
| h194 CTRL-K1 (no window, MES) | 11.59 |

Paired **−1.00** (se 0.92), better on **3/5**. Threshold 1.26 → **P2, no effect.**
Per-seed h199: 11.88, 10.80, 7.66, 9.37, 13.23.

## Why this is the strong result, not a weak one

This is the FIRST arm to give the teacher access to the TRUE function (no GP fantasy
error) while avoiding h145's fatal confound. Two things were checked, not assumed:

1. **SC-DIVERSITY passed with room to spare** (τ=0 action SD 104.3% of MES's, gate was
   ≥25%). h145's oracle collapsed to a single endpoint and could not distinguish
   "quality hurt" from "degeneracy hurt". h199 does not collapse, so a null here is
   attributable to quality, not degeneracy.
2. **No fidelity-mix confound.** LF fraction 0.228 vs CTRL-K1's 0.261 — essentially
   matched. h199 has no window (K=1), so the sequence-length fidelity-saturation
   mechanism found this session (h202) does not apply here. Unlike h197's P3, this null
   cannot be explained by fewer/more-expensive queries per budget.

**Per the protocol registered before running:** P2 retracts nothing, but it makes the
mechanism's SUFFICIENCY claim unsalvageable — no teacher, however good, reaches the DT
through this channel, because giving it the TRUE function still didn't move the outcome.

## What remains

h198 (the same lookahead schema under the GP, both label variants) is still running.
Combined with this result: if h198 also nulls, the two-arm design (GP vs oracle) will
have shown the bottleneck was never fantasy quality -- since even perfect fantasies
don't help. If h198 does NOT null while h199 does, that would be the genuinely strange
outcome and would need its own explanation before being believed.
