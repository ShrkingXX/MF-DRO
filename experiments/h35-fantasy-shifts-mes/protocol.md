# H35 — why does fantasy conditioning make MES prefer high fidelity?

## The gap H34 left

H34 established that the same teacher chooses HF **50.6%** of the time inside
rollouts but **4.2--11.9%** at real inference, and that the fidelity head
faithfully learns the rollout rate. It did not explain *why* the two regimes
differ. The position breakdown gave the clue: `tau=0` has the lowest HF label
rate (33.0%) and the rate climbs as fantasy observations accumulate.

## Candidate mechanism

The KO model's HF posterior variance is
`var_H = rho^2 * var_L + var_delta`. Inside a rollout, `make_fantasy_ko`
conditions on sampled observations **without refitting hyperparameters**. LF
fantasy observations shrink `var_L` but leave `var_delta` untouched, so `var_H`
stops falling while `var_L` keeps falling. Cost-normalised MES compares
`MES_H/c_H` against `MES_L/c_L`; if LF queries progressively stop buying HF
information, the HF branch wins more often as `tau` grows.

## Measurement

Within rollouts, at every step `tau`, over the candidate pool:

1. mean `sigma_H` and mean `sigma_L`
2. the ratio `max(MES_H)/c_H` to `max(MES_L)/c_L` --- the quantity whose sign
   decides the fidelity
3. the realised HF choice rate

and the same three quantities at a **real inference** state for comparison.

## Locked predictions

1. **PRIMARY**: the ratio `max(MES_H)/c_H : max(MES_L)/c_L` **increases with
   `tau`**, and is higher at `tau=7` than at a real-inference state. This would
   make fantasy conditioning the cause.
2. **SUPPORTING**: `sigma_L` falls faster than `sigma_H` across `tau`, i.e. the
   ratio `sigma_H/sigma_L` rises.
3. **NULL**: if the ratio is flat in `tau`, the HF preference is not driven by
   fantasy accumulation and the explanation lies elsewhere --- most likely in the
   candidate pool or the `y*` distribution differing between regimes.

Single process, 1 thread. `PROTOCOL.md` untouched; no regret claim.
