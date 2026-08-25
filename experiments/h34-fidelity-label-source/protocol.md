# H34 — why is the fidelity head miscalibrated? Check its training labels.

## The hypothesis

H33 found the fidelity head emits `p ~ 0.557` while the teacher's unconstrained
preference on the same pools was `4.2%` HF, and H31's running log shows the
teacher's realised rate over full runs is about `11.9%`. The head is trained by
BCE on the fidelity labels in the rollout batch, so **the miscalibration should
be visible in those labels**.

`simulate_mf_trajectory` applies `minimum_hf_fraction = 0.25`
(`mf_dro.py:1247`): after joint MES selects `(x_tau, ell_tau)`, a step can be
overridden to HF to maintain a floor. If that override fires often, the student
is trained on labels whose HF rate is far above what the teacher actually wants —
and it would be learning exactly what it was shown.

## Measurement

On a real 200-trajectory rollout batch:

1. HF rate of the fidelity labels the student is trained on (`actions_ell`)
2. HF rate the teacher would choose **without** the floor, on the same steps
3. how often the `minimum_hf_fraction` override changes the label
4. the same broken down by rollout position `tau`

## Locked predictions

1. **PRIMARY**: the training labels' HF rate exceeds the teacher's unconstrained
   rate by at least 10 percentage points. If so, the head's miscalibration is
   **inherited from its labels**, not a training failure, and the remedy is a
   one-line change to how the floor is applied.
2. **NULL**: if the label rate matches the teacher's preference, the floor is not
   responsible and the miscalibration originates in the head or its loss.

## Note on what this does and does not excuse

Even if the labels explain the *level*, they cannot explain the *uninformativeness*
(`p` sd = 2.4e-4, `corr` = 0.155 with the teacher's choice). A well-calibrated
constant is still a constant. These are separate defects and this experiment
addresses only the first.

Single process, 1 thread. `PROTOCOL.md` untouched.
