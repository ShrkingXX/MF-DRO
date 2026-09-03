# h194 Stage 1a — **P3. The window HURTS.** But it is the *broken* window.

**CONFIRMATORY** against the gate registered before launch, with the readout committed
before any run finished. 10/10 runs, no failures.

## Drift check first — and it came back perfectly clean

| | frozen rel% |
|---|---|
| fresh CTRL-K1 (today) | **11.59** |
| h84's ROI-Q10 (Aug 27, 17 commits earlier) | **11.59** |
| paired | **+0.00** |

Not rounding: **5/5 seeds are bit-identical**, same query counts (115/103/120/118/123) and
identical traces. The default ROI-Q10 path is genuinely unchanged across those 17 commits,
so **h84-era ROI controls remain quotable**. Running a contemporaneous control turned out
to be unnecessary — which could only be known by running it.

## The registered gate

| | frozen rel% |
|---|---|
| **WINDOW (K=8)** | **16.58** |
| CTRL-K1 (K=1) | **11.59** |

Paired per seed: **+2.83, +6.67, −0.03, +4.08, +11.38**
Mean **+4.99** (se 1.93), **window worse on 4/5**, against a ±1.26 threshold.

**P3 — the sliding window HURTS, by ~4× the noise floor and 2.6 se from zero.**

## Answering the question directly: no

The ask was whether this beats **default MF-DRO** (`use_roi=False` = 15.82):

| | frozen rel% | paired | better on |
|---|---|---|---|
| WINDOW (K=8) | 16.58 | **+0.76** vs default | 3/5 |
| ROI-Q10 alone | **11.59** | −4.23 vs default | — |

**No.** The window arm is slightly *worse* than default MF-DRO, and far worse than simply
turning ROI on. **And the expert-teacher half was never run** — Stage 1b is gated on 1a not
returning P3, precisely because a harmful shared component makes the combination
unattributable. That gate now binds.

## My prediction record here, stated plainly

I registered **P3** before Stage 1a. Then, two ticks later, I **withdrew it**: the
reasoning rested on equating "late-τ under a teacher whose τ=0 was destroyed" (TAIL-MES,
which fails) with "late-τ under a normal teacher" (ordinary acquisition choices), and a
preliminary measurement pointed the other way (K=8 sat 0.036 *further* from the box
centre, not closer). My stated expectation at run time was **P2**.

**P3 fired. My withdrawn prediction was right and my live one was wrong.** I claim no
credit for the first: the reasoning behind it was faulty, and a faulty argument reaching a
true conclusion is not a successful prediction. The readout script auto-labels the verdict
"as the mechanism predicted" — **that string is stale and should be disregarded.**

## The load-bearing caveat: this is the window with ZEROED past actions

h195 (the human-requested audit against DT Algorithm 1) established that our window
**zeroes every historical action token**, while the paper feeds the actions actually
executed and our own training fills them with real locations. With the causal mask, the
readout state token attends to those slots. So this arm fed the model step-tuples it never
saw in training — *state s followed by action 0*.

**So P3 here means "the window as implemented hurts", not "windows hurt".** That
qualifier must travel with the number.

**h196 fixes exactly this** and is registered, coded, identity-gated (122.29066752728207,
exact) and SC-verified — its first SC caught a silent no-op where `_hist` was rebuilt as
fresh dicts and dropped the recorded actions. It launches now.

## What this does and does not RETRACT

- **Nothing is retracted.** P3 was one of the three registered outcomes and it is the one
  that leaves the mechanism intact.
- **P1 would have forced an exception** into "input-side fixes cannot help" — which unifies
  three Phase-1 nulls (state, conditioning, history). That exception is not needed.
- **But the unification is not yet safe either**, because h196 could still produce P1 with
  a correctly-fed window. Until h196 reports, "input-side fixes cannot help" stands on
  h194's *defective* window plus the h185/h186 measurements, not on a clean window test.
