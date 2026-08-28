# H130 — analysis

**P1 PASS. P2 PASS.** Both locked predictions confirmed. Zero new runs.

Statistic: count-matched mean non-init HF y, K = min HF count across arms within
each seed. ROI-Q10 vs h83 `MF-DRO` control, seeds 42-46, paired.

| bench | K per seed | control | ROI | delta | sd | \|m\|/sd | up | verdict |
|---|---|---|---|---|---|---|---|---|
| **Borehole_8D** | 86, 98, 81, 82, 77 | 224.306 | 241.359 | **+17.053** | 4.811 | **3.54** | 5/5 | **SEPARATES** |
| Hartmann_6D | 8, 24, 12, 5, 8 | 2.266 | 2.191 | -0.075 | 0.228 | 0.33 | 2/5 | no |
| Currin_2D | 15, 34, 22, 21, 22 | 13.568 | 13.559 | -0.009 | 0.029 | 0.30 | 1/5 | no |
| Ackley_10D | 40, 40, 40, 39, 40 | -5.576 | -5.878 | -0.301 | 0.386 | 0.78 | 1/5 | no |

P1: Borehole reproduces h120's P3 exactly (+17.053, effect 3.54, 5/5) — as it
must, being the same statistic; it confirms the measure is correctly specified.

P2: **none of the other three separate.** All are below 1.0, and all three point
slightly NEGATIVE — but at effects of 0.30-0.78 that is not separable from no
effect and must not be read as the ROI harming quality.

Count-matching mattered only on Borehole (matched +17.053 vs unmatched +15.157,
a 1.9 gap). Elsewhere the two agree to within 0.08, because the ROI barely
changes the HF count on those benchmarks — itself consistent with the fidelity
mechanism being Borehole-specific.

## The conclusion this completes

Four mechanisms have now been tested for generality across the same four
benchmarks, and every one gives exactly one positive cell:

| mechanism | Borehole | Hartmann | Currin | Ackley |
|---|---|---|---|---|
| regret benefit | **-4.22%, 1.74** | 0.48 | — | — |
| fidelity mix | **1.65** | 0.78 | 0.41 | 0.49 (censored) |
| boundary mass | **54.2% vs 14.7%** | — | — | — |
| **query quality** | **+17.05, 3.54** | 0.33 | 0.30 | 0.78 |

**Every measured ROI effect is Borehole-specific. Nothing about the ROI
generalises.**

## And the sharpest form of h121's mismatch

The founding diagnosis measured bad HF query quality **on Hartmann** — "mean HF
query score 0.336 vs MF-MES's 0.747". Query quality is the channel it named.

On Hartmann, the ROI does not improve query quality: delta -0.075, effect 0.33,
2/5 seeds. **The ROI fails to move the diagnosed quantity on the benchmark where
it was diagnosed**, while moving it decisively on a benchmark where the
diagnosis was not made and where h121 showed the waste is smallest (3.2% median
vs Hartmann's 12.5%).

That is the same mismatch h121 found in the benefit, now established in the
mechanism the diagnosis itself named. It is the strongest statement of it
available, and it does not depend on any interpretation of what the ROI is
"really" doing.

## Limitations

- n=5 per benchmark, no p-values.
- Three benchmarks against one prediction; P3's fresh-seed requirement was the
  pre-registered handling and did not need to fire.
- Mean HF y on each benchmark's own scale, not the diagnosis's normalised score
  (that normalisation is not recoverable from the record — h121). Same construct,
  stated scale.
- Ackley is included here although excluded from the fidelity comparison: its
  one-sided censoring applies to a count fraction, not to a value statistic. The
  asymmetry is deliberate and recorded in the protocol.
