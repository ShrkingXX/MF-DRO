# h167 — **R2 fires.** The collapse is to the box centre, not to each teacher's mean.

CONFIRMATORY, zero compute. R2 was named in the protocol as the likeliest outcome
and as **NOT support** for the mechanism as stated. It is what happened.

| arm | d(centroid, CENTRE) | d(centroid, MIDPOINT) | nearer |
|---|---|---|---|
| control (works) | 0.7604 | 0.4870 | midpoint |
| h155 UCB-LOC (works) | 0.7788 | 0.4574 | midpoint |
| h153 FROZEN (works) | 0.7426 | 0.4770 | midpoint |
| h159 EXPLOIT (works) | 0.7375 | 0.4814 | midpoint |
| **ORACLE (fails)** | **0.0394** | 0.6766 | **centre** |
| **DIVERSE-GOOD (fails)** | **0.0409** | 0.6689 | **centre** |
| **RANDOM-POOL (fails)** | **0.0239** | 0.6577 | **centre** |

- **P1 HOLDS.** RANDOM-POOL's centroid is 0.0239 from the domain centre — nearest
  of all seven arms.
- **P2 FAILS.** ORACLE's centroid is 0.0394 from the **centre**, not 0.6766 from
  its own teacher's mean. It does not collapse toward *its* distribution's mean.
- **P3 HOLDS**, and enormously: working arms sit 0.7375–0.7788 from the centre,
  failing arms 0.0239–0.0409. A factor of ~20.

## What this changes

All three failing arms land within **0.04** of the exact centre of the
normalised box, in 8 dimensions, regardless of what their teachers were aiming
at. ORACLE's teacher averages to the midpoint of centre and x* (0.66 away) and
DIVERSE-GOOD's to something similar; neither student goes there.

**So the collapse is a property of the network, not of the target
distribution.** "The DT predicts the mean of an unlearnable target" is
**wrong as stated** — it predicts the centre of its own output box. That is a
different claim, it points at the output parameterisation rather than at the
target statistics, and it must not be reported as the mechanism surviving.

## And it puts the L_loc puzzle back in play

The resolution offered in h162 was: low L_loc on the failing arms means the
network sits at the targets' mean, where loss is minimal. If it sits at the box
centre instead, that is **not** the loss-minimising constant for ORACLE or
DIVERSE-GOOD, whose targets average elsewhere. The resolution does not follow.

Registered follow-up, not yet run: compare the MSE of a constant-centre
predictor against a constant-at-target-mean predictor on each teacher's action
distribution, and set both against the observed L_loc (0.018–0.022 failing,
0.040 control). If the observed loss matches the target-mean constant but the
real queries sit at the centre, then training-time fit and inference-time output
disagree — which would point at the inference conditioning (rtg_target outside
the training support, the subject of h148) rather than at learnability.

## What survives

The dispersion split (h162, h164) is untouched: it is a fact about spread and it
replicates on two benchmarks. What is withdrawn is the *explanation* offered for
it. This is the third time an explanation has outrun its evidence on this front.

---

# h167b — the DT is NOT collapsed at training time. The failure is at INFERENCE.

The follow-up registered above was run immediately. MSE of a constant predictor
against each teacher's own action distribution, in the same normalised space and
averaging convention as `L_loc`:

| teacher | target mean (first 3 dims) | MSE @ box centre | MSE @ target mean | **observed L_loc** |
|---|---|---|---|---|
| RANDOM | 0.50, 0.50, 0.50 | 0.0834 | 0.0834 | **0.018–0.022** |
| ORACLE | 0.75, 0.25, 0.56 | 0.1085 | 0.0533 | **0.018–0.022** |
| DIVERSE-GOOD | 0.73, 0.50, 0.50 | 0.0710 | 0.0544 | **0.018–0.022** |

**The observed training loss is 2.5–4× LOWER than the best possible constant
predictor.** The DT is not sitting at the box centre during training, and it is
not sitting at the target mean either. It is fitting the mapping, and fitting it
well.

## This retracts the learnability framing as stated

The framing said the failing teachers' actions are not a function of the
observable state, so the network cannot fit them. **It fits them** — better than
any constant, by a wide margin, on all three failing teachers. The actions are
demonstrably learnable.

Yet the same network's real queries land within 0.04 of the box centre (h167),
while every working arm sits 0.74–0.78 away.

**So the failure is a disagreement between training and inference, not a failure
to learn.** That relocates the whole question: it is not about what the teacher
teaches, it is about what happens when the trained network is asked for an
action under the inference conditioning.

## The obvious suspect, and it is already on the books

At inference the DT is conditioned on `rtg_target` = max(batch_max,
0.5·running_max) — by construction the **extreme upper tail** of the training
returns. h156 measured those distributions: the failing arms have mean rtg[0] of
+0.008 to +0.020 with s.d. ~0.10, and a target of ~0.30. The network is being
asked "what action yields a return three standard deviations above anything
typical?" — and in these arms the trajectories that got there did so by lucky
fantasy draws, not by doing anything systematic. There is no action that answers
the question, and the network returns its box centre.

`h148-rtg-target-outside-support` exists in this repo and is exactly this
question. It should be re-read before anything new is designed.

## Caveats

The constant-predictor baselines are computed from the arms' own path generators
(ORACLE and DIVERSE-GOOD reproduce h145's and h146's code); they are not the
literal serialised `actions_x`, which was never stored. Bayesian early stopping
may truncate some trajectories. Neither affects a 2.5–4× gap. The control's
targets are model-dependent so no analytic constant exists for it; it is
excluded rather than guessed.

**Fourth explanation to fall on this front.** The pattern is consistent: each
account fitted the evidence available and outran it. The dispersion split
(h162/h164) and the 2×2 remain facts; their explanations keep failing.
