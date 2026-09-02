# h179 — standardised conditioning. **P2 (unchanged), PROVISIONAL pending h181.**

CONFIRMATORY against the protocol committed before launch. 5/5 seeds, Borehole.

## Probe read FIRST, per protocol

| statistic | h179 STDCOND |
|---|---|
| RTG responsiveness, real state, pooled | **0.1010** |
| per-seed | 0.0961, 0.1070, 0.1277, 0.0640, 0.1104 |
| fidelity flips between rtg=0 and rtg=1 | **23% of iterations** |

The emitted action *does* move with the conditioning, and the fidelity choice
flips on nearly a quarter of iterations. Real-state and train0-state
responsiveness are identical to 4 d.p. — the inference state is bit-identical to
a training state, as already established.

## Regret (frozen metric)

| arm | n | mean rel% | sd | per-seed |
|---|---|---|---|---|
| h179 STDCOND | 5 | **16.66** | 2.33 | 18.02, 16.71, 12.82, 16.89, 18.88 |
| control (h83 MF-DRO) | 5 | 15.82 | 2.36 | 15.28, 14.77, 12.93, 16.90, 19.19 |

Paired (same seeds): **+0.85 rel% pts, se 0.62**, i.e. **+5.4%** relative.
**2 of 5 seeds clearly worse** (+2.74, +1.94); the other three tied to within 0.31.

### The gate was under-specified, and how that was resolved

The protocol wrote P1/P2/P3 as "improves / unchanged / degrades" with **no numeric
width for 'unchanged'**. That is a gate weakness of the kind `check_gate.py`
exists to catch, and it was not caught before launch. It is resolved here by
applying the project's **pre-existing** harness noise floor — 6.1% mean / 10.9%
worst, established well before this arm — rather than a threshold invented after
seeing the result. **+5.4% sits inside that floor → P2.**

## Verdict: P2 → R3, PROVISIONALLY

> R3 P2 holds -> the channel is genuinely irrelevant either way; the tau=0 account
> is strengthened and the "defect" framing is downgraded to a curiosity.

**Why provisional.** h179 is the only probe ever run on Borehole; every earlier
probe is Hartmann. So h179 varies standardisation *and* benchmark at once, and
0.1010 cannot yet be attributed to standardisation. **h181** (registered and
launched, 5 seeds, one-line diff) supplies the matched unstandardised control.

## How h181 will be read — pre-specified, before its results exist

- **h181 responsiveness < 0.051 (P1).** Standardisation genuinely made the channel
  responsive. Then h179's null means **"responsive but useless"**: the DT's action
  changes with the target — fidelity flips on 23% of iterations — and performance
  does not improve. R3 stands, strengthened: the target carries no information
  about what a good query is.
- **h181 responsiveness in [0.051, 0.152] (P2).** Borehole is simply more
  responsive than Hartmann and standardisation changed little. Then h179 **tested
  nothing**, its P2 is uninformative about standardisation, and the in-situ half
  of h178's z-scoring claim is unsupported — the 336× stays a module-level
  statement. R1's spirit applies to h178, not h179.
- **h181 responsiveness > 0.152 (P3).** Standardisation *reduced* responsiveness,
  which would invert the h178 reasoning and require re-examining that analysis.

Under two of the three branches h179's headline changes. Recording the branches
now is what makes the eventual read mechanical rather than authored after the fact.
