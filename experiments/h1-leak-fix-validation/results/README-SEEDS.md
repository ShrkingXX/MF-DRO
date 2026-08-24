# Seed provenance in this directory — READ BEFORE ANALYSING

This directory now contains MF-DRO results for **two different purposes**:

| seeds | purpose | governed by |
|---|---|---|
| **42-51** | the **frozen evaluation** (10 seeds) | `PROTOCOL.md` — FROZEN |
| 52-71 | LIVE control arm for the H6 extension | `../../h6-frozen-dt/protocol-extension.md` |

`code/analyze.py` hard-codes `SEEDS = [42..51]` and must **keep** doing so. The
frozen-protocol result is:

    MF-DRO 0.5047 +/- 0.0395 | MI-Greedy 0.5091 +/- 0.1266 | GP-UCB 1.7934 +/- 0.1223
    success test: FAIL (0.5442 >= 0.3825)

Verified unchanged after seeds 53-55 landed in this directory.

**Do not widen `SEEDS` in `analyze.py`.** Extending the frozen evaluation's seed
count after seeing its result would be exactly the optional-stopping the H6
extension protocol was written to avoid, and `PROTOCOL.md` freezes n=10. The
extension seeds exist only to give H6 a paired LIVE arm at larger n.
