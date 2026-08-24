# H26 — BTG sensitivity: a headline claim that was never measured

## Why

`paper/main.tex`'s abstract states the proposed query "is invariant to the state,
the return-to-go, the **budget-to-go**, and its own interaction history."

An audit of every probe in this project shows **none of them ever varied BTG**.
RTG was swept (H8, H9, H10), state was swept (H5 audit, H21, H22), history was
tested (H11) --- BTG was held fixed in all sixteen. The claim rests on inference
from the general inertness, not on measurement. Given that three inferences in
this project have already failed re-measurement, it must be tested before it
ships.

## Design

Trained model held fixed. Sweep the BTG target across its **realised support**
--- observed `btg_now` values across runs span roughly 22 to 52, with the
structural floor at $2c_H + 6c_L = 22.0$. Sweep
$\{22, 27, 32, 37, 42, 47, 52\}$ and, for comparison, a deliberately
out-of-distribution extension $\{5, 100, 500\}$ reported separately.

Measure the fraction of 12 candidate pools on which the proposed argmax changes.

## Locked predictions

1. **PRIMARY**: in-band BTG movement is $<20\%$, consistent with every other
   input channel. This retains the abstract's wording.
2. **NULL**: if in-band BTG moves the argmax on $\ge 20\%$ of pools, the
   abstract is **wrong** and must be corrected --- BTG would then be the one
   live conditioning channel, which would be a materially different and more
   interesting finding than the current story.

## Guard

The OOD extension is reported separately and never used to support the in-band
claim --- the same discipline that made an early RTG sweep uninterpretable until
it was re-run within its realised band.

Single process, 1 thread. `PROTOCOL.md` untouched.
