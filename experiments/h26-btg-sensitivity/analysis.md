# H26 — BTG and extreme-RTG sensitivity: the last two unmeasured claims

## Part 1 — BTG (the claim that had never been measured)

An audit of all sixteen probes in this project found that **none had ever varied
BTG**, yet `paper/main.tex`'s abstract asserted invariance to it.

| sweep | argmax moved | mean pairwise corr |
|---|---|---|
| in-band, realised support $[22, 52]$ | **0/12 = 0.0%** | 1.000000 |
| OOD $\{5, 100, 500\}$ (reported separately) | **0/12 = 0.0%** | 0.999992 |

**PRED 1 PASS.** BTG is inert both inside and far outside its realised support.
The abstract's wording is now measured rather than inferred.

## Part 2 — "is it definite that *any* value of RTG wouldn't matter?"

A fair challenge: prior RTG evidence covered the realised band $[0.5,1.0]$ and a
flawed $0.1\times$--$10\times$ design. "Any value" was stronger than what had been
tested. Rather than soften the wording, we swept RTG over **nine orders of
magnitude**.

| sweep | argmax moved | mean pairwise corr |
|---|---|---|
| in-band $[0.5, 1.0]$ | **0/12 = 0.0%** | 0.999933 |
| extreme $10^{-3} \to 10^{6}$ | **1/12 = 8.3%** | 0.998046 |

So the honest, precise statement is:

- Within its realised support, RTG changes the decision on **zero** of twelve
  pools.
- Stretched across $10^{-3}$ to $10^{6}$ — values the system can never present —
  it changes the decision on **one** of twelve, with at most 2 distinct argmaxes
  in a 9-point sweep.

It is therefore **not literally true** that no RTG value can ever change the
decision. It is true that no value in-distribution does, and that even a
$10^{9}$-fold stretch moves it on 8.3% of pools, far below the 30% bar used
throughout. The paper should say this rather than claim absolute invariance.

## Note on the scripts' printed verdicts

Both probes were derived from H8's template and still print H8's labels
("IN-BAND BTG", "the 'RTG does not drive decisions' finding SURVIVES"). Those
strings are **stale**; the measurements above are what the code computed. This
is the fifth stale auto-verdict of the project, and it is recorded rather than
quietly ignored --- the standing lesson is that a printed conclusion must key on
the decisive quantity, and inherited templates carry inherited sentences.
