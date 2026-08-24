# H8 analysis — the RTG finding SURVIVES the OOD correction

## Result

Same trained model, same state, 12 resampled candidate pools.

| sweep | targets | argmax moved | distinct argmaxes | mean pairwise corr |
|---|---|---|---|---|
| **IN-BAND** [0.5, 1.0] | 0.50 ... 1.00 | **0/12 = 0.0%** | 1.00 | 0.999933 |
| OOD (original design) | 0.06 ... 6.2 | 1/12 = 8.3% | 1.08 | 0.998864 |

Both locked predictions PASS.

## Interpretation

The earlier RTG conclusion holds, and is now stated correctly:

> **Within the ~2x band the RTG target actually occupies, varying it changes the
> proposed candidate on 0% of trials, with score vectors correlated at 0.99993.**

Note the direction: the in-band sweep is *even less* sensitive than the OOD one
(0.0% vs 8.3%). That is the expected ordering — smaller input perturbation,
smaller output change — and it means the OOD confound was, if anything, making
RTG look *more* influential than it is, not less. My worry that the finding was
an OOD artefact was the right worry to have, and it resolves in the direction
that strengthens the original claim.

## Consequences for earlier conclusions

- **H4 (AdaLN) refutation stands.** I had flagged it as "unsafe" pending this
  check, because AdaLN cannot rescue a signal that is constant by construction.
  The signal is not merely under-used at the margins — it is inert across its
  entire realised support, so the refutation is safe.
- **H5 (score head) stands** for the same reason.
- The "MF-DRO re-fits rather than conditions" characterisation is now supported
  by a measurement taken on the distribution the model actually inhabits.

## What this does NOT resolve

It remains true that `rtg_target` is structurally clamped to [0.5, 1.0] by
`max(batch_max, 0.5*running_max)` over a batch-max-normalised rtg, with CV 0.232.
So two distinct claims are still confounded:

1. the network ignores RTG, and
2. RTG carries too little variation to be worth using.

H8 shows the network does not respond *within the support it is given*. It does
not show the network would ignore a signal with genuinely informative variation.
Testing that needs a schema change (widen `alpha_rtg`, or condition on an
un-normalised quantity) and a retrain — a real experiment, not a probe, and the
single most promising remaining lead on the conditioning question.
