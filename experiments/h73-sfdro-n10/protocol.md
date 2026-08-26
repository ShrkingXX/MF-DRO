# H73 — Does SF-DRO's Hartmann advantage over SF-MES survive n=10?

**CONFIRMATORY.** Locked before any h73 number exists. Bars name magnitudes.

## Why this is the test that matters

SF-DRO beating SF-MES on Hartmann is **the strongest pro-DRO result this project
has**: 11.5% vs 21.4% at n=3, winning 3/3, with lower variance (sd 1.76 vs
10.05). It is the one place the DRO machinery clearly outperforms its own MES
counterpart with the fidelity setting held fixed, and SF-DRO is the north star's
base method.

It is also n=3, and lesson 26 is now three for three: **every** exploratory n=3
direction taken to n=10 in this project has failed — h45 (regression head, had
already changed a shipped default), h64 (POOL600, was the north-star claim), h70
(KO-style GP, sign reversed). h72 then showed the Hartmann column specifically is
where n=3 misleads most: MI-Greedy's published entry moved **+12.7 points** at
n=10 and MF-GP-UCB's **+21.5**.

So the project's best remaining pro-DRO claim sits in its least reliable column
and has never been replicated. That is the highest-value thing left to test.

## Design

SF-DRO on Hartmann 6D at seeds **42-51 (n=10)**. Seeds 44/46/48 already exist in
h59 and are reused unchanged as a built-in reproduction control; the 7 new seeds
run here. Cost budget 200, identical config to h59, no LF queries.

Comparator: **SF-MES at n=10 from h72**, already measured (21.17%) with its own
reproduction control passed at +0.00.

~19.3 min/run; 7 workers alongside h71's 6 stays inside the 15 cap.

## Locked predictions

1. **PRIMARY.** SF-DRO beats SF-MES at n=10 by **>= 5.0 points** of relative
   regret **and** wins **>= 7/10** seeds. Both required. The n=3 gap was 9.9
   points at 3/3; requiring half of it plus a clear win count guards against the
   h64 failure, where a favourable mean came from a single seed.
2. **SECONDARY.** SF-DRO's sd stays below SF-MES's (n=3 was 1.76 vs 10.05).
   Reported but not decisive — the general "DRO buys variance" hypothesis was
   already refuted on worst-case across six pairs.
3. **NULL.** Gap < 5.0 points **or** wins <= 6/10. Then **the last pro-DRO result
   in this project is withdrawn**, lesson 26 goes four for four, and the honest
   summary becomes that no DRO variant has ever beaten its own MES counterpart
   under replication.
4. **REVERSED.** SF-MES ahead by >= 5.0 points.

## What this cannot settle

n=10 is better than n=3, not definitive. It tests Hartmann only — SF-DRO loses to
SF-MES on Currin and Borehole at n=3, and those are not re-run here (67.2 and
43.8 min per run). A win here would mean SF-DRO beats SF-MES on **one** of three
benchmarks under replication, not that it is the better method.
