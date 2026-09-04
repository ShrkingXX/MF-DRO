# h201 arm B — **P3, and it lands EXACTLY on the saturation floor.**

**CONFIRMATORY**, 5/5 finals. CEILING/DIAGNOSTIC (h145's oracle, not a method).

## Result

| arm | final regret |
|---|---|
| h201B oracle + K=1 (matched control) | **43.94** |
| h194 CTRL-K1 (MES, no window) | 11.59 |
| h201A oracle + K=8 window (already in hand) | 0.00 |

Per-seed h201B: 46.23, 47.35, 44.84, 44.68, 36.60 -- worse than CTRL-K1 on **5/5**.

**43.94 is not merely a bad number -- it is the SAME saturation floor value already
identified this session as "the initial design" and previously hit by TAIL-MES, a
completely different failing teacher.** Two independent teachers whose tau=0 carries no
usable signal converge on the identical number. That is strong corroboration that the
floor is a property of the READOUT POSITION carrying nothing useful, not of any one
teacher's idiosyncrasies.

## The clean ablation this arm exists to provide

| | paired difference | seeds |
|---|---|---|
| h201A (K=8) - h201B (K=1), SAME teacher | **-43.94** (se 1.90) | 5/5 |
| h201B (K=1) - CTRL-K1 (MES), SAME window | **+32.35** (se 1.87) | 0/5 |

**Neither the teacher alone nor the window alone helps.** The oracle teacher WITHOUT the
window is a disaster (worse than plain MES by 32.35). The window WITHOUT a teacher whose
late step is worth reading (h196/h197, this session) also hurts. Only the COMBINATION --
a teacher whose late-step action is the answer, paired with a window that exposes that
position -- produces h201A's 0.00. This is the isolation the two-arm design was built to
provide, and the prediction registered before either arm ran is confirmed exactly.
