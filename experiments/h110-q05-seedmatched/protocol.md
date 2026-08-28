# H110 — give the BEST setting the seed-matched n=10 treatment h106 gave the weaker one

LOCKED BEFORE ANY RUN. 5 runs: Borehole ROI-Q05 at seeds 52-56.

## Why this is the missing measurement

h106 established that q=0.10 removes **57%** of MF-DRO's one real deficit, at
n=10 seed-matched to the comparison's own seed set (42-46 + 52-56).

q=0.05 is separably better -- h97 (-1.52, 4/5 at 47-51) and h107 (-1.57, 4/5 at
42-46), agreeing to 0.05 across independent seed sets, pooled -5.40 at 10/10
against no-ROI. But **q=0.05 has never been measured on the comparison's seed
set**: it exists at 42-46 and 47-51, and Borehole MF-MES does not exist at 47-51
(verified across every experiment directory).

On 42-46 alone -- the only seed set with all four arms -- q=0.05 closes 62%
against q=0.10's 45%. Five runs at 52-56 completes the n=10 picture, because
both comparators already exist there:

    MF-MES at 52-56    h92, 5/5 verified present
    no-ROI at 52-56    h89 CONTROL, 5/5
    Q05 at 52-56       MISSING -> this experiment

## THIS EXPERIMENT IS CONDITIONAL ON h109

h109 is running now and decides whether the working-tree patches perturb
`use_roi=True` runs. These 5 runs execute the same patched code. **If h109's P1
fails, h110 is contaminated exactly as h106/h107/h108 are, and its result must
be discarded and re-run on clean code.**

I am launching before h109 reports because the slots are otherwise idle, both
finish in roughly the same window, and h106/h107/h97/h102 already ran on this
code -- so five more does not change the contamination picture, only its size.
**That is a compute argument, not an evidential one**, and the dependency is
registered here so no h110 number is ever reported without h109's verdict
attached.

## Predictions (effect sizes stated, per the rule four bars failed today)

**R1 (PRIMARY). Q05 beats no-ROI at 52-56 with paired mean <= -3.0 and
|mean| >= 0.5 sd.** Registered POSITIVE. Prior: -5.01 and -5.79 on the two
existing seed sets, both 5/5.

**R2. Q05 closes MORE of the MF-MES gap at n=10 than q=0.10's 57%.**
Registered POSITIVE but WEAKLY -- 62% vs 45% on 42-46 is one seed set, and the
same setting's closure moved 45% -> 57% between seed sets, so the quantity is
noisier than one decimal suggests.

**R3 (NEGATIVE). Q05 does NOT close the gap: its n=10 mean stays above MF-MES's
8.24 at these seeds.** Registered NEGATIVE so a real gain cannot be inflated. A
refutation would be the most important result of the run and needs re-verifying
before belief.

**R4. Paired, Q05 still loses to MF-MES on >= 6 of 10 seeds.** Registered so the
mean-based percentage cannot stand alone -- on 42-46 the paired count is 2/5
against MF-MES while the mean-based figure reads 62%.

## Gate

7 workers running. 5 more takes it to 12, inside 15.
