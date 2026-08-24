# H4 analysis — REFUTED, with a caveat, and a sharper diagnosis

## Result vs the locked prediction

Locked (350c0a3): *under `adaln`, an RTG sweep changes the argmax on >30% of
sweeps, vs a measured 0% under `token`.*

Measured, 12 independently resampled candidate pools, sweep {0.1,0.5,1,2,5,10}x:

| arm | sweeps where argmax moved | distinct argmaxes/sweep | argmax == argmax(mu_H) |
|---|---|---|---|
| `token` | **3/12 = 25.0%** | mean 1.25, max 2 | 9/12 (75%) |
| `adaln` | **0/12 = 0.0%** | mean 1.00, max 1 | 8/12 (67%) |

**H4 REFUTED.** AdaLN did not increase RTG sensitivity — it removed what little
existed. The prediction failed in the opposite direction.

## Two honest corrections to my own framing

1. **The prediction's premise was partly wrong.** It asserted "a measured 0%
   under token." Under *this* probe's protocol (12 resampled pools, argmax
   recovered from `propose_mf`'s returned coordinates), `token` measures **25%**,
   not 0%. The earlier 0% came from a differently-trained model and a single
   candidate pool. The comparison that stands is token 25% vs adaln 0%, both
   measured here under identical conditions.

2. **A confound limits how strongly this refutes the mechanism.** AdaLN-Zero
   initializes to gamma=1, beta=0 — deliberately identity at step 0, so the
   conditioning effect must be *learned*. This probe trains 10 epochs on a
   single rollout batch. That is almost certainly too little for the modulation
   layer to depart from its zero init, which would produce exactly the observed
   0%. So this refutes "AdaLN is a quick win at this training scale"; it does
   **not** cleanly refute the underlying RADT/DDT mechanism at full training
   scale.

## What this rules out, and what it points at

Ruled out: swapping *how* the return signal enters the network is not, by
itself, sufficient — at least not cheaply.

**Sharper diagnosis, which is the real value here.** Both arms show
argmax(score) == argmax(mu_H) on **67–75%** of pools. Combined with the earlier
measurement that replacing `h` with the batch mean or a *shuffled* `h` changes
the argmax only ~8% of the time, the picture is:

    rtg -> h    : works (embeddings differ; score vectors shift)
    h   -> score: nearly severed (score head is ~indifferent to which h it gets)

The bottleneck is **downstream of the conditioning**, in the score head, not in
how RTG reaches `h`. This explains why every conditioning-side intervention has
failed and predicts that further ones will too. No amount of improving `rtg -> h`
can matter while `h -> score` is close to a constant map and the head is largely
ranking candidates by `mu_H`.

## Next

Target `h -> score` directly, not the conditioning pathway. Candidates:
- Force score-head dependence on `h` (e.g. remove `mu_H` from candidate features
  so the head cannot shortcut to a fixed acquisition, mirroring the earlier MES
  column removal that took argmax(mu_H) agreement from 100% to 70%).
- Re-run this probe after full-length training to close the AdaLN confound
  before discarding the literature mechanism entirely.
