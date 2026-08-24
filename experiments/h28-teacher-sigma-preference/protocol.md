# H28 — is the uncertainty aversion INHERITED from the teacher?

## The question

H24/H25 established the student's learned rule penalises `sigma_H`
(`w[sigma_H] < 0` on 9/10 seeds). The obvious reading is that training corrupted
something: the teacher is cost-normalised MF-MES, an *information-seeking*
acquisition, so the student appears to have inverted its teacher.

There is a competing explanation that has never been checked. **MES is not
monotone in `sigma`.** MES(x) depends on both the posterior spread *and* the gap
to the sampled `y*`: a point with huge variance far from the optimum carries
little information about `y*`. So the teacher's *realised choices* on this
benchmark might systematically sit at **low** `sigma_H`, in which case the
student is imitating faithfully and the aversion is inherited, not learned-wrong.

These have very different implications:

- **Student's fault**: the training procedure is broken and fixing it should
  recover exploration.
- **Teacher's fault**: the acquisition itself behaves exploitatively in this
  regime, and no amount of better distillation helps.

## Measurement

Over a real 200-trajectory rollout batch, at every step, compare the **chosen**
candidate's `sigma_H` against the distribution of `sigma_H` in its own candidate
pool. Report:

1. the mean **percentile** of the chosen candidate's `sigma_H` within its pool
2. Spearman correlation between the teacher's score and `sigma_H`, per pool
3. the same two quantities for `mu_H`, as a positive control (the teacher should
   clearly prefer high `mu_H`)

## Locked predictions

1. **PRIMARY**: if the mean chosen-`sigma_H` percentile is **< 40%**, the teacher
   demonstrably prefers low-uncertainty points and the student's negative weight
   is **inherited**. The paper's "the student inverted the sign of its teacher's
   defining term" must then be **retracted and rewritten**.
2. **ALTERNATIVE**: if the percentile is **> 60%**, the teacher prefers
   high-uncertainty points, the student genuinely inverted it, and the current
   wording stands.
3. **AMBIGUOUS**: 40--60% means the teacher is roughly indifferent to `sigma_H`;
   the student's negative weight is then neither inherited nor an inversion, and
   the paper should say only that the teacher provides no uncertainty-seeking
   signal to imitate.

## Control

`mu_H` percentile must be clearly above 50%. If it is not, the extraction is
wrong and nothing else in this experiment is interpretable.

Single process, 1 thread. `PROTOCOL.md` untouched.
