# H24 — if the policy is a fixed acquisition rule, which one?

## Why

H23 established that MF-DRO's policy is, to within $0.13\%$, a **fixed linear
acquisition function** $\langle \bar{w}, \mathit{cf}_k\rangle$. A reader will
immediately ask what that function *is*. Answering turns a negative result into
a positive characterisation: the transformer, trained on multi-fidelity MES
rollouts, converges to a namable rule.

## Method

Report the **signed** $\bar{w}$ against the known feature layout

    [x_norm(0..5), mu_H, sigma_H, mu_L, sigma_L, dist_inc]

and measure how often $\arg\max_k \langle \bar{w}, \mathit{cf}_k\rangle$ agrees
with each of a set of standard rules over the same 12 candidate pools:

- `mu_H` alone (pure exploitation)
- `mu_H + beta * sigma_H` for `beta` swept over {0.5, 1, 2, 3, 5}
- `mu_L` alone (does it just track the cheap proxy?)
- `mu_H + mu_L` (naive fidelity pooling)
- the cost-normalised MF-MES teacher score itself

## Locked predictions

1. **PRIMARY**: some standard rule agrees with the learned argmax on **>= 75%**
   of pools. If one does, the paper can name what MF-DRO reduces to.
2. **NULL**: if no rule reaches 75%, the learned rule is a linear combination
   with no simple interpretation. That is still reportable --- "a fixed but
   uninterpretable linear rule" --- and the coefficient table becomes the result.

## Guard

Agreement on 12 pools of 200 candidates is a **descriptive** measure, not a
claim that MF-DRO *implements* the matched rule. If the best match is a
`mu_H + beta*sigma_H` family member, we report the best `beta` and its agreement
rate, not that MF-DRO "is UCB". Chance agreement is ~0.5% (1/200), so any rate
above a few percent is meaningful, but only a high rate licenses naming.

Single process, 1 thread. No regret claim; `PROTOCOL.md` untouched.
