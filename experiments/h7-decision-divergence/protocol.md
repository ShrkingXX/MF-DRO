# Protocol — H7: measure the DT's contribution DIRECTLY, not through regret

**Locked before running.** Results commit must be separate.

## Why this experiment exists

H6 asks "does continued DT training contribute anything?" and answers it by
comparing **final regret** between a frozen and a live policy. That routes a
mechanistic question through the noisiest possible measurement, and it failed:
the paired estimate changed sign four times (n=1 -0.208 ... n=10 +0.098), the
95% CI straddles zero, and a post-hoc power analysis shows ~80 seeds would be
needed. The design cannot answer its own question.

Meanwhile, every *mechanistic* finding in this project has been solid, because
each was a near-deterministic measurement rather than a noisy average:

| finding | measurement | strength |
|---|---|---|
| target leakage | a_emb ablation, 15.6% -> 0.7% | decisive |
| MF-GP-UCB freeze is definitional | n_HF = 0 on 10/10 seeds | decisive |
| score head ignores `h` | argmax unchanged 12/12, both arms | decisive |
| RTG never moves the argmax | pinned in every configuration tried | decisive |
| H6 regret comparison | CI [-0.097, +0.292] | **useless** |

**The lesson: measure the mechanism, not the downstream metric.** H7 applies
that to H6's question.

## Hypothesis

**H7** (same claim as H6, better instrument): after ~5 iterations of training,
continued DT training barely changes *what the policy proposes*.

## Design

Run a single MF-DRO trajectory per seed. At iteration 5, take a **snapshot** of
the DT weights. Then at every subsequent iteration t > 5, on the **identical
state, RTG, BTG and candidate pool**, compute:

- `x_live` : proposal from the DT as it currently is (trained through t)
- `x_frozen`: proposal from the iteration-5 snapshot

and record:

1. `argmax_agree[t]` — do the two select the **same candidate**? (binary)
2. `dist[t]` — normalized L2 between the two proposed locations
3. `fid_agree[t]` — do they choose the same fidelity?

Both policies are evaluated on the same inputs, so this isolates the effect of
the extra training alone. Only the LIVE proposal is actually executed, so the
trajectory is a normal MF-DRO run and nothing about the evaluation changes.

## Why this is far better powered than H6

Each run yields **~50-200 paired decisions** instead of one regret scalar, and
agreement is near-deterministic rather than variance-dominated. 5 seeds give
several hundred paired comparisons — versus H6's 30 noisy scalars.

## Locked predictions

1. **Primary**: mean `argmax_agree` over t > 5 is **> 0.70**. (i.e. the extra
   training rarely changes the decision)
2. `dist` does not systematically grow with t (no progressive divergence).

## What each outcome means

- Agreement high -> H6's claim is supported by a properly powered instrument:
  continued training is near-inert at the level of decisions. This would make
  the "MF-DRO re-fits rather than conditions" story defensible on mechanism
  even though the regret comparison is underpowered.
- Agreement low but regret unchanged -> the policy changes a lot yet it does not
  matter for outcome. That is *also* a strong result: it would say the decision
  space is flat w.r.t. the policy, i.e. the DT is choosing among near-equivalent
  options and the GP/MES teacher is doing the real work.
- Agreement high early then falling -> there is a training horizon after which
  the DT does start to matter, and k=5 was simply too early. Actionable.

All three outcomes are informative, which is the point.

## Compute

Adds two extra `propose_mf` calls per iteration (cheap relative to rollout
generation, which dominates at ~41 s/iter). 5 seeds, single process each.
Must not launch while the H6 extension holds the pool.
