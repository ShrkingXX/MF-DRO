# Protocol — H5: the bottleneck is h -> score, not rtg -> h

**Locked before running.** Results commit must be separate.

## Motivation

H4 refuted the conditioning-side explanation (AdaLN made RTG sensitivity worse:
0% vs 25% of sweeps moving the argmax). The residue of that experiment is a
sharper diagnosis:

    rtg -> h      works           reward embeddings differ (pairwise dist 6.8-13.3);
                                  score vectors shift with RTG (corr 0.986-0.9999)
    h   -> score  nearly severed  replacing h with the batch mean or a SHUFFLED h
                                  changes the argmax only ~8% of the time;
                                  argmax(score)==argmax(mu_H) on 67-75% of pools

If `h -> score` is close to a constant map, then *no* conditioning-side
intervention can matter, because everything RTG/BTG/state carry arrives through
`h`. That would explain the entire failed sequence of conditioning fixes.

## Hypothesis

**H5**: the score head has learned a fixed, largely `mu_H`-greedy acquisition
over the candidate features and is close to indifferent to `h`. Denying it the
GP features forces the decision through `h`, restoring state/RTG sensitivity.

## Change under test — no new code required

The `use_candidate_features` flag already implements exactly the contrast:

- **Arm A (`use_candidate_features=True`, current default)**: candidates carry
  `[x_norm(d), mu_H, sigma_H, mu_L, sigma_L, dist_inc]` = d+5.
- **Arm B (`use_candidate_features=False`)**: candidates carry bare coordinates
  `[x_norm(d)]` only. The score head has no GP quantity to rank by, so any
  non-arbitrary ranking must come through `h`.

Everything else identical, including `rtg_conditioning="token"` (H4 showed
`adaln` is not the lever, and this experiment must vary one thing).

## Locked predictions (confirmatory)

Measured with the H4 probe harness (12 resampled pools), Arm B vs Arm A:

1. **h-indifference falls**: fraction of positions where shuffling `h` changes
   the argmax rises from ~8% to **>30%**.
2. **The mu_H shortcut breaks**: argmax(score)==argmax(mu_H) agreement falls
   from 67-75% to **<30%**.
3. **RTG regains traction**: fraction of RTG sweeps moving the argmax rises
   above Arm A's 25%.

Prediction 2 is near-tautological if the feature is absent, so it is a
*manipulation check*, not evidence for H5. **Predictions 1 and 3 are the real
test** — they can fail even with the feature removed (the head could simply
become arbitrary rather than `h`-driven).

## What each outcome means

- 1 and 3 both hold -> H5 supported: the score head was the bottleneck, and the
  DT's conditioning can be made causally effective by denying the shortcut.
- 2 holds but 1 and 3 fail -> the head becomes *arbitrary*, not `h`-driven.
  H5 refuted in its useful form; the conditioning signal in `h` is genuinely
  too weak to rank candidates, which would be a substantive negative result
  about return-conditioned BO.
- Nothing changes -> the shortcut is being reconstructed from the coordinates
  themselves (the GP is smooth in x), which would itself be worth reporting.

## Cost caveat, stated in advance

Arm B removes information the policy legitimately needs. Regret may worsen even
if H5 is supported. H5 is a **diagnostic** about where the signal is lost, not a
proposed final method. Do not treat an Arm B regret regression as refuting H5,
and do not adopt Arm B as the method on the strength of a mechanism result.

## Compute

Probe only, single process, thread-capped. Do not launch while a grid holds the
pool.
