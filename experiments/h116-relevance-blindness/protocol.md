# H116 — Is MF-DRO blind to per-dimension relevance?

STATUS: LOCKED before any statistic was computed.
TYPE: CONFIRMATORY (prediction stated below, before computation).
COMPUTE: zero new runs. Re-analysis of stored h83-main-comparison queries.

## Motivation (code facts, verified 2026-08-28)

1. `src/models/ko_gp.py:312` — the KO GP uses `RBFKernel(ard_num_dims=self.d)`.
   Per-dimension lengthscales ARE learned. (`src/model/exactGP.py` is an
   isotropic legacy single-fidelity module and is NOT used by the KO path.)
2. `src/policy/mf_dro.py:251` — `ls = scale_kernel.base_kernel.lengthscale.mean().item()`.
   The ARD vector is collapsed to its mean before entering the DT state.
3. `grep -n lengthscale src/policy/mf_dro.py` — this is the ONLY place
   lengthscales enter the pipeline. Per-dimension relevance is computed by
   the surrogate and then discarded everywhere.

So the DT policy has no per-dimension relevance signal. The founding
diagnosis ("proposals 3x more dispersed", confirmed: 3.65x Hartmann /
3.73x Borehole, sensitivity-weighted) prescribes reducing dispersion where
it matters -- and no intervention in this project has ever reduced weighted
dispersion. This protocol tests whether relevance-blindness is the reason,
BEFORE spending compute on an intervention.

## Benchmark anisotropy (cached binned first-order S1, `tools/perdim.py`)

| bench | d | top S1 | PR/d (1/d=concentrated, 1=isotropic) |
|---|---|---|---|
| Borehole_8D | 8 | 0.858 | 0.168 |
| Hartmann_6D | 6 | 0.347 | 0.553 |
| Currin_2D   | 2 | 0.782 | 0.759 |
| Ackley_10D  | 10| 0.103 | 1.000 |

## Hypothesis

MF-DRO's HF-query dispersion profile is uncorrelated with per-dimension
relevance; MF-MES's is negatively correlated (it tightens the dimensions
that matter). MF-MES is the natural control: its acquisition is computed on
the GP posterior and therefore consumes the ARD lengthscales directly.

## Measure (per benchmark x method x seed)

1. X = non-init HF queries (`fid==1 and not is_init`), shape [n, d].
2. s_j = sd(X[:, j]) for each dimension j.
3. Profile p_j = s_j / sum_j s_j  -- normalising removes overall tightness,
   isolating the SHAPE of dispersion across dimensions. This is required:
   the two methods differ ~3.7x in overall weighted dispersion, and the
   claim under test is about relevance-awareness, not magnitude.
4. rho = Spearman(p, S1shares) over the d dimensions.

Robustness (declared now, not chosen post hoc): repeat 2-4 with MAD in
place of sd. Both are reported.

## Prediction (locked)

- Borehole (PRIMARY, most anisotropic): median rho(MF-MES) < 0;
  median rho(MF-DRO) ~ 0.
- Separation criterion: seeds are matched (same seed = same initial design),
  so pair them. GATE: |mean(rho_MES - rho_DRO)| / sd(rho_MES - rho_DRO) >= 1.0.
- Hartmann (SECONDARY, mildly anisotropic): same sign predicted, weaker.
  Reported, does not gate.
- Currin (d=2, Spearman ranges over {-1,+1} only) and Ackley (S1 near
  uniform, PR/d=1.000, ranking ill-conditioned) are DEGENERATE for this
  measure. Reported for completeness, EXCLUDED from inference.

## Falsification and its consequence

If MF-MES also shows rho ~ 0 on Borehole, relevance-awareness does NOT
distinguish the two methods. The code facts above would remain true but
would not explain the dispersion gap, and the proposed ARD-weighted-L_loc
intervention would be unmotivated. This protocol exists to kill that
intervention cheaply if it deserves killing.

## Stated limitations

- n=5 seeds. NO p-values (project rule).
- S1 shares are binned first-order estimates: they ignore interactions.
  Borehole's 0.858 top share is robust; Hartmann's flatter profile is
  noisier and its ranking correspondingly less reliable.
- Observational re-analysis of existing runs. Establishes association
  between method and relevance-awareness. It does NOT show that supplying
  relevance would improve regret.
- Data source is h83-main-comparison ONLY (4 bench x 5 method x 5 seeds,
  seeds 42-46). No cross-experiment globbing.

---

## AMENDMENT 1 (2026-08-28) — coordinate scale. Filed AFTER first computation.

DISCLOSURE: the first run of this protocol was executed and its numbers are
reported in analysis.md as SUPERSEDED. This amendment was written after
seeing them. It is filed because the measure was confounded, not because
the numbers were unwelcome.

The protocol said "X = non-init HF queries" without fixing the coordinate
scale. Stored `x` is NOT on a common scale across benchmarks:

  Borehole_8D  domain widths [0.1, 4.99e4, 5.25e4, 120, ...]  -- RAW units
  Hartmann_6D  domain widths [1, 1, 1, 1, 1, 1]               -- unit cube

So on the PRIMARY benchmark, s_j = sd(X[:,j]) is dominated by each
dimension's box width, which is a constant of the benchmark, identical for
every method and every seed. The normalised profile p therefore measured
the domain box, not the policy. This explains the otherwise implausible
stability of the superseded result (rho = -0.405, sd 0.02 across seeds, for
BOTH methods): both were largely reading off the same fixed box.

CORRECTION: normalise to the unit cube before computing dispersion,
  Z = (X - domain_min) / (domain_max - domain_min),
using `get_benchmark(f"{bench}_HF")["domain_min"/"domain_max"]`, the same
source `tools/perdim.py` uses for S1. All of steps 2-4 then run on Z.
This makes Borehole and Hartmann commensurate and removes the box constant.

The hypothesis, prediction, gate (>=1.0), primary/secondary/degenerate
assignments and limitations are UNCHANGED. Only the coordinate scale changes.

## AMENDMENT 2 (2026-08-28) — sample-size floor. Filed at the same time.

The superseded run also exposed that non-init HF query counts on Hartmann
are small and highly variable (as few as 8 for MF-DRO seed42, vs 25 for
MF-MES seed42). A per-dimension sd over n=8 points in d=6 is too noisy to
rank. Runs with fewer than 15 non-init HF queries are now reported with
their n and EXCLUDED from the paired statistic; the exclusion count is
reported. Borehole (n ~ 79-99) is unaffected.
