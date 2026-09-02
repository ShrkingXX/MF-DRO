# h156 + h156b — CONFIRMATORY. The bridge exists, and it is not adaptivity.

9 states (Borehole seeds 42/43/44 × 3 cuts), N=100 trajectories per condition.

| harness condition | mean | sd | **MAX** | real arm | observed rtg_target | err |
|---|---|---|---|---|---|---|
| C1 closed-loop | +0.391 | 0.221 | **0.9060** | control | 0.9761 | −7.2% |
| C2 open, OWN greedy path frozen | +0.255 | 0.219 | **0.8238** | **h153 (running)** | — | **FORECAST** |
| C3 open, random path | +0.020 | 0.097 | 0.2739 | RANDOM-POOL | 0.2965 | −7.6% |
| C4 oracle path to x* | +0.008 | 0.113 | 0.3022 | ORACLE | 0.3113 | −2.9% |
| C5 diverse-good endpoints | +0.008 | 0.106 | 0.2650 | DIVERSE-GOOD | 0.3285 | −19.3% |

**The offline harness reproduces all four observed rtg_targets from trajectory
geometry alone**, three of them within 8%. It was not tuned to them: C4 and C5
copy h145's and h146's own path generators verbatim. h156b's R3 fires.

## The bridge findings.md was missing

`rtg_target = max(batch_max, 0.5·running_max)` (mf_dro.py:2056-2060) is a
**MAXIMUM over the batch**. Every penalty measured before this was a penalty on
the MEAN, which is why +0.16 looked far too small to explain 0.976 → 0.311. It
was the wrong statistic. The failing arms do not merely shift the mean down —
they flatten the distribution (sd 0.10-0.11 vs 0.22) and earn **essentially zero
information gain on average** (+0.008 to +0.020 against the control's +0.391).
A max has nothing left to find.

## R2 fires, and it costs me the framing I adopted last tick

C2 — freezing each rollout's own adaptively-derived path, which is *exactly*
h153's design — **retains 90.9% of C1's MAX** (below C1 in 8/9 states, but
barely). The three failing arms retain 29-33%.

**So freezing is not what separates them.** The open-loop penalty h152 measured
is real (+0.0845 Hartmann, +0.1594 Borehole, replicated) but it costs ~9% of the
tail where the failing arms lose ~70%. It is a genuine confound and NOT the
operative cause.

### Withdrawn

The framing adopted after h152 — that the substitute teachers fail *because*
they are open-loop / non-adaptive, and that this is what h145/h146/h149 share —
is **withdrawn**. It was an over-correction. h152 correctly identified an
uncontrolled variable; I then wrongly promoted that variable to the explanation
without checking whether it was large enough, which is the same error in the
opposite direction from the one h152 caught.

### Reinstated, now quantitative

h149's original mechanism survives and is vindicated: **the reward is
information gain, and trajectory quality (proximity to x*) is orthogonal to
it.** Walking toward an already-good point earns almost nothing — C4 and C5
confirm this directly, and C5 shows genuine endpoint DIVERSITY does not rescue
it (+0.008 mean, 29.3% of the tail). What h149 lacked was the reason a modest
mean penalty destroys the target: the target is a max, and these batches have no
tail.

## PRE-REGISTERED FORECAST for h153 (recorded while it is ~25% through)

h153 is C2. **Predicted rtg_target ≈ 0.82, not ≈0.31, and h153 should therefore
NOT reproduce the 43.94 failure — it should land near the control.**

If h153 fails anyway, this entire tail account is refuted and must be withdrawn,
including the reinstatement above. That is stated now, before its result exists
in any form.

## Honest weaknesses

- C5's error is −19.3%, much the worst of the four. The account fits
  DIVERSE-GOOD least well and that is not explained.
- All 9 states come from Borehole control traces, seeds 42/43/44. The states the
  FAILING arms actually visit are different (they never improve), so the harness
  matches their targets while running on the control's state distribution. That
  it does so is evidence the effect is trajectory geometry rather than state, but
  it is not a direct replication of their conditions.
- 9 states, 3 seeds. No p-values.
