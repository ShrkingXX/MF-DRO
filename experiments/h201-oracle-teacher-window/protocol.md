# h201 — h145's interpolating teacher READ AT TIMESTEP 7 instead of 0

**CONFIRMATORY**, and it measures a **CEILING/DIAGNOSTIC**, not a method (x* is not
available at run time). **Human-proposed.** Locked before any code is run.

## The argument

h145's teacher walks `x_tau = x_start + (x* - x_start)*tau/(T-1)` with
`x_start ~ U(domain)`. Measured this session (400 random starts, Borehole): the path
improves y at **every single step, 400/400** — it is exactly the "every step improves the
best observed y" design. It failed at **+28.13 rel%, 0/5**.

The established mechanism says why: the DT is read at **timestep 0**, and h145's tau=0
action is a uniform random point (measured y at tau=0: mean 75.7, sd 45.4 — noise). It is
structurally TAIL-MES, which h171 measured at **43.94, 0/5** (the saturation floor)
against HEAD-MES's **16.96, 5/5**. The trajectory is excellent from tau=1 on and
worthless at the only step that reaches inference.

**The window changes which step reaches inference.** Verified in code, not assumed:
`decisionTransformer.py` feeds `ts = torch.arange(T)` and reads out the LAST state token
(`h_full[0, ...][-1]`), so with a full K=8 window the readout sits at **position 7**.
`step_norm` is constant within a training rollout, so the positional embedding is the
ONLY channel encoding tau.

Under h145's teacher the tau=7 action is **x* exactly**. So the mechanism does not merely
permit this arm to work — **it predicts it**.

## Arms

| arm | teacher | window | note |
|---|---|---|---|
| **A** | h145 interpolating (oracle) | **K=8** | the test |
| **B** | h145 interpolating (oracle) | K=1 | matched control, same code state |

B is run rather than cited: h145's +28.13 came from an older code state, and this session
has already found one defect (zeroed action tokens) and one confound (fidelity mix) in
the window path. A control from the same build is the only honest comparison.

## SCs, registered before running

1. **READOUT POSITION (GATE).** Instrument inference and record the readout index per
   real iteration. It must reach **7** once >=8 real queries exist. If it never exceeds 0,
   the premise is false and the arm is void — report as a GATE MISS, not a result.
2. The teacher's tau=7 action must be x* (bit-level) on every rollout, and its tau=0
   action must have the uniform-start variance h145 intends.

## Prediction

- **P1 — large improvement, approaching the optimum.** The DT emits ~x* because that is
  its teacher's action at the READ position. This would establish that (i) the window
  genuinely moves the readout, (ii) the DT emits the teacher's action at the read
  position, and (iii) h145's failure was **purely positional**. It also makes the window
  a working INSTRUMENT even though it has not yet been a working METHOD.
- **P2 — no better than arm B.** The window does NOT change what the DT emits. This
  **RETRACTS my claim** — asserted above from code inspection — that the readout moves to
  timestep 7, and would undercut the whole sliding-window schema including h197.
- **P3 — worse than B.** Would point at the h146 degeneracy: the tau=7 action is x* for
  EVERY trajectory, i.e. a zero-variance target, and a DT fitting a constant map may be
  harmed even when the constant is good.

## The caveat that must travel with any P1

At tau=7 the teacher's action is x* with **zero across-rollout variance**. A P1 therefore
does NOT show that "a good teacher helps"; it shows the DT can copy a constant when the
constant is read at the right position — and that constant is the answer. This is a
CEILING and will be labelled one wherever it is quoted. Its value is diagnostic: it
separates "the window cannot move the readout" from "the readout moves but the teachers
we tried had nothing useful at the read position".

## Cost

Ordinary teacher cost (no lookahead) — h197 measured ~80 min/seed. 10 workers for both
arms. Currently 15/15 in use (h198a, h198b, h199), so this queues until slots free.
