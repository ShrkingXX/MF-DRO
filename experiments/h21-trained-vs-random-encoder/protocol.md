# H21 — is the attenuation architectural, or does TRAINING create it?

## Why this is the right last question

The paper's Limitations section states the gap honestly:

> "We show that an unbottlenecked scoring head does not restore conditioning,
> but we did not attempt an encoder-side remedy, so *the conditioning channel in
> this architecture is unusable* is better supported than *no architecture in
> this family could condition*."

There is a sharper, cheaper test than any encoder-side remedy: compare the
**trained** encoder against a **randomly initialised** one. Both share the exact
architecture, so the comparison isolates *learning* from *architecture*.

- If a random encoder passes state variation through at roughly the same
  $0.346\times$, the contraction is **architectural** — a property of the
  transformer's shape on these inputs.
- If a random encoder passes it through far better, then **training actively
  destroys the signal**, and the cause is the loss/targets, not the shape. That
  materially changes the paper's claim and its future-work direction.

## Design

Identical seed, identical states, identical candidate pools. One variable:
whether `_train_dt` is called.

| arm | encoder |
|---|---|
| T | trained (current result: $s\!\to\!h$ $0.346\times$, argmax $0/12$) |
| R | randomly initialised, never trained |

Measured on the 10 genuinely distinct $\tau{=}0$ states (different ensemble
blocks — **not** H5's broken same-state swap):

1. relative spread of $s$, $h$, $w$ and the resulting attenuations
2. argmax movement under a genuine state swap
3. pairwise cosine of the coefficient vector

## Locked predictions

1. **PRIMARY**: arm R's $s\!\to\!h$ attenuation exceeds arm T's $0.346\times$ by
   at least $1.5\times$ (i.e. $\ge 0.52$). This is the test of "training
   destroys it".
2. **SECONDARY**: arm R moves the argmax on $>30\%$ of probes. An untrained
   policy's *choices* are meaningless for performance — this measures only
   whether the **channel is open**, and is reported as such.
3. **NULL**: if arm R matches arm T, the contraction is architectural, the
   paper's stronger phrasing becomes defensible, and the Limitations sentence
   above should be revised accordingly.

## Interpretation guard, fixed in advance

A random network scoring differently across states is **not** evidence that
random is better. Arm R is a channel-capacity probe only. No regret claim, no
suggestion that training be skipped, and `PROTOCOL.md` is untouched.

## Compute

Single process, 1 thread.
