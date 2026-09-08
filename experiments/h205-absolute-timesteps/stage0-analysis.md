# h205 Stage 0 — PASS. Every flag verifiably does what it claims.

| config | seq_len | position indices exercised | loss-masked frac |
|---|---|---|---|
| CTRL (both off) | 8 | 0-7 (**8 distinct**) | 0.000 |
| A (prefix only) | **15** | 0-14 | **0.438** |
| B (absolute only) | 8 | **0-52 (53 distinct)** | 0.000 |
| C (both) | 15 | 0-45 | 0.440 |

CTRL confirms the defect as diagnosed: only 8 embedding rows ever receive
gradient, and index 7 is always the fragment end. Arm B exercises 53 rows on a
short BUDGET=70 probe (would reach ~118 at full budget). Arm A's 0.438 masked
fraction is 7/15 -- exactly the prefix, loss-excluded but still attended.

SC1a/SC1b/SC1c (no-op gates) PASS; SC3 no index overflow (52 < 256); SC5 cold
start handled. STAGE 0: PASS -- Stage 1 cleared to run.

```
========================================================================
  CTRL  (both off)         seq_len=  8  pos idx 0..7 (8 distinct)  loss-masked frac=0.000
  A     (prefix only)      seq_len= 15  pos idx 0..14 (15 distinct)  loss-masked frac=0.438
  B     (absolute only)    seq_len=  8  pos idx 0..52 (53 distinct)  loss-masked frac=0.000
  C     (both)             seq_len= 15  pos idx 0..45 (46 distinct)  loss-masked frac=0.440

========================================================================
SC1a absolute labelling exercises >8 embedding rows: 53 distinct, max 52  -> PASS
SC1b prefix lengthens the training sequence: 8 -> 15  -> PASS
SC1c prefix is loss-masked: frac=0.438 (CTRL 0.000)  -> PASS
SC3 no index overflow: max idx 52 < 256  -> PASS
SC5 cold start (n<K-1) handled: all four configs completed  -> PASS
========================================================================
STAGE 0: PASS
```
