## BES Diagnostic Finding
BES from DRO paper (Section D.3) uses EI as stopping
signal. EI naturally decays toward 0 as GP identifies
the optimum (bounded below, monotone). MES measures
global HF entropy via Thompson sampling -- noisy and
non-monotonic within 8-step rollouts (observed floor
ratio 0.159, mean floor 0.346 across 105 steps on
smooth synthetic). BES threshold at any fixed fraction
of signal_0 would either fire spuriously at noisy
low points or never fire at all.
Decision: disable BES (bes_delta=0.0). Variable-length
training infrastructure kept as insurance.
If rollout_length is later increased to 20+, revisit
BES with threshold ~0.2-0.3 and EI (not MES) as signal.
