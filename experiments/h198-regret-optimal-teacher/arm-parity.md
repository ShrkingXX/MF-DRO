# h198 arm-parity check (run because the two arms launched from different code)

h198a launched at commit ff257d3 + the NameError fix; h198b at 36367fc. The delta
is the dead-work removal (max_horizon defaults to 0 and oracle_f to None, both
inert). I ARGUED that is bit-identical because CRN reseeds every replication, so
RNG skipped at the end of one replication cannot leak into the next. Argument is
not evidence, so it was tested: the two code versions were run side by side on 10
GP states x 3 horizons with matched seeds.

```

  h198a-code vs h198b-code identical on 30/30 (seed, steps_left) cases
  -> SAFE: the two arms differ ONLY by rollout_reward
```

**30/30 identical.** The arms differ ONLY by rollout_reward, which is what the
factorial requires.
