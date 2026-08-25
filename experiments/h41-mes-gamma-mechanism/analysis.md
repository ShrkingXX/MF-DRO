# H41 — the correct mechanism for MES's argmax (corrects h28's explanation)

**Status: EXPLORATORY** (prompted by a correct objection: MES has no
"posterior-mean term"; it maximises information gain about y*).

## What MES actually computes

From `_compute_mes_hf_vectorized`, with `gamma = (y* - mu_H)/sigma_H`:

    H0 = log(sigma_H) + ½log(2*pi*e)
    H1 = mean_k[ log(sigma_H) + ½log(2*pi*e) + log Phi(gamma)
                 - gamma*phi(gamma)/(2*Phi(gamma)) ]
    MES = H0 - H1 = mean_k[ gamma*phi/(2*Phi) - log Phi ]

**`log sigma_H` cancels.** MES is a function of `gamma` **only**, and it is
monotonically **decreasing** in `gamma`. There is no separate mean term and no
mu + beta*sigma tradeoff — the earlier description was a UCB-shaped account of an
acquisition that does not have that form.

## Measured (40 pools, Hartmann 6D)

| quantity | value |
|---|---|
| `gamma` at the chosen candidate | **+1.5893 ± 0.0341** |
| `gamma` averaged over the pool | **+2.7203 ± 0.0213** |
| chosen `mu_H` percentile | **99.5%** |
| chosen `sigma_H` percentile | **27.9%** |
| pools where `mu_chosen > mean(y*)` | **0.0%** |

The chosen point has **lower gamma than the pool**, exactly as a
decreasing-in-gamma acquisition requires. The acquisition is behaving correctly.

## The correct explanation of the low-sigma selection

Because `gamma > 0` at every chosen point (mu never exceeded the sampled y*),
minimising `gamma = (y*-mu)/sigma` pulls in **two** directions: raise `mu`
(shrink the numerator) *or* raise `sigma` (grow the denominator).

It resolves overwhelmingly through `mu`: chosen points sit at the **99.5th
percentile of mu**. And because `mu` and `sigma` are **anti-correlated in a GP**
(`corr = -0.4696`, measured in h30), pushing `mu` to the top of the pool drags
`sigma` **down** to the 27.9th percentile as a side effect.

So the low-uncertainty selection is a **consequence of gamma-minimisation under
mu–sigma anti-correlation**, not evidence that MES ignores information gain. MES
is doing exactly what it is defined to do.

## A caveat this exposes on h28's headline number

h28 reported the chosen-`sigma` percentile as **2.9%**. That run used the
oversized initial design (`initial_hf=36, initial_lf=60`). With the
literature-standard design (`6/45`) the same measurement gives **27.9%** — an
order of magnitude different.

**The "bottom 3% of uncertainty" figure is specific to the oversized
initialisation** and should not be quoted as a general property of MF-MES. The
qualitative direction (chosen sigma below the pool median, mu near the top)
survives; the magnitude does not.
