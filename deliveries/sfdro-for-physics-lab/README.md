# SF-DRO — Bayesian optimization for the battery campaign

Two optimizers, same job: *given everything measured so far, which recipe next?*

- **basic BO** (`naive_bo.py`) — Gaussian process + expected improvement. The reference point.
- **SF-DRO** (`src/policy/dro.py`) — our method. Same GP surrogate, but the "what next" step plans several steps ahead instead of using an acquisition function.

Both were written for benchmark functions the code calls itself. A cell that takes days to cycle is not that, so `sfdro.py` wraps SF-DRO in an **ask/tell** interface — the two sections below are the whole API.

Snapshot of commit `df0d92f`, 2026-08-27.

## Install

Python 3.9, CPU only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch==2.0.0" "botorch==0.8.4" "gpytorch==1.10" "numpy<2" \
    scipy omegaconf hydra-core matplotlib tqdm
```

Then `python example_asktell.py` — eight fake cells through the full loop in about half a minute. If it prints recipes and cycle counts, you're set up.

## 1. Initialize with your dataset

A campaign needs two things: the **box** you may search, and the **measurements you already have**.

```python
import numpy as np
from sfdro import SFDRO

BOUNDS = np.array([          # one row per bound, one column per knob, your own units
    [0.0, 0.2, 4.0],         # lower: additive wt%, C-rate, cutoff voltage
    [5.0, 3.0, 4.5],         # upper
])

opt = SFDRO(bounds=BOUNDS, minimize=False, seed=42)
opt.initialize(X_init, y_init)   # X_init (n, 3) recipes built; y_init (n,) measured
```

`initialize` records the points and fits the GP ensemble. From a spreadsheet:

```python
import pandas as pd
df = pd.read_csv("history.csv")
opt.initialize(df[["additive_wt_pct", "charge_C_rate", "cutoff_voltage_V"]].to_numpy(),
               df["cycles"].to_numpy())
```

- `minimize=False` = larger is better (cycle life). Use `True` for resistance, fade rate. Either way you pass and read your own numbers in your own sign.
- Every point must lie inside `BOUNDS`, and every proposal will too. Make it the range you're actually willing to build.
- Two points is the technical minimum; 5–10 is what our runs use. Spread them out (Latin hypercube or Sobol) — a one-factor-at-a-time sweep teaches the GP nothing about the other knobs.
- One number per recipe. Average replicate cells. Leave failed cells out entirely — there's no way to say "tried and failed".

## 2. Update with a new measurement

```python
x_next = opt.ask()            # (3,) array — the recipe to build next
y_next = measure(x_next)      # <-- your experiment, however long it takes
opt.tell(x_next, y_next)

x_best, y_best = opt.best     # best so far
X_all, y_all = opt.history    # everything, for your records
```

`tell` is instant; the refit happens inside the next `ask`. So you can record a number the moment it comes off the cycler and worry about the next proposal later.

- **`ask` is slow on purpose** — one call refits ten GPs, simulates 200 campaigns inside them, and trains a small transformer. About **80 s** on a laptop at default settings, growing with dataset size and dimension. Nothing next to a week-long cell, but don't assume it's hung.
- **One point at a time.** Calling `ask` repeatedly without telling it anything gives you near-identical recipes — nothing changed between calls. For a parallel batch, ask once and vary the other channels by hand.
- **You can tell it things it didn't ask for.** Any in-bounds point works: side experiments, old cells, a recipe someone insisted on. More data is strictly better.
- **Resuming.** Save `opt.history` to CSV and rebuild with `initialize(X, y)` on the full history. One caveat: the transformer is warm-started within a session, so a rebuild costs you roughly the first proposal after a restart.

## How it works

Basic BO fits a GP to the data, scores candidates by expected improvement, and builds the winner. Its weakness is that expected improvement is **myopic** — it optimizes the gain from the *next single* measurement, and cannot express "this looks mediocre, but measuring it tells me which half of the space to discard."

SF-DRO replaces that scoring step. Each iteration it fits **ten** GPs with different starting length scales (so they disagree about smoothness — that's honest model uncertainty), then plays out 200 full optimization campaigns *inside* those GPs, several steps deep and costing no cells. Each simulated step is labelled with its **return-to-go**: the total improvement still to come before the end of that simulated campaign, so a step that sets up a large gain three moves later scores high. A small Decision Transformer learns state → action from those labelled trajectories, and the proposal is what it predicts when asked for the best return-to-go seen. Everything is rebuilt from scratch each iteration, so it adapts as data arrives.

**Caveat worth stating plainly:** SF-DRO is a research method, not a settled tool. It costs ~100× more computation per proposal than basic BO — a good trade only when the experiment is expensive, which is exactly your case — and its advantage isn't uniform across problems. The toy comparison in `example_closed_loop.py` is one seed on one smooth function; it's a smoke test, not evidence. For a real answer on your system, run both from the same initialization dataset over several seeds.

## Files and knobs

| file | |
|---|---|
| `sfdro.py` | ask/tell wrapper + every hyperparameter in one place — **start here** |
| `example_asktell.py` | the lab loop, with a stand-in measurement to replace |
| `example_closed_loop.py` | the other mode: a simulated objective the code calls itself |
| `naive_bo.py` | basic BO, standalone and readable in 140 lines |
| `src/policy/dro.py` | SF-DRO itself |
| everything else | imports of the above; `src/objectives/synthetic.py` has benchmark functions for testing |

Keyword arguments to `SFDRO(...)`, defaults as used for the paper:

- `rollouts_per_iter=200`, `rollout_length=8` — simulated campaigns per proposal and how far each looks ahead. The main cost *and* quality knobs.
- `gp_num_models=10` — GPs in the ensemble.
- `noise_constraint=1e-2` — floor on fitted measurement noise; raise it if your cells are noisy.
- `seed=42` — change for a different run, fix to reproduce one.

For a fast first pass: `gp_num_models=5, rollouts_per_iter=20, rollout_length=4` brings a proposal down to a few seconds. Both examples use those. The remaining knobs in `build_config` (`rtg_schema`, `use_mes_reward`, …) are research ablations pinned to the tested configuration — changing them puts you off that path.

## Rough edges

- Prints `Using device: cpu` and PyTorch attention-mask warnings on every proposal regardless of `verbose`. Harmless; `python -W ignore` quiets them.
- `known_optimal_value` only exists to print "regret" against a known benchmark optimum. Meaningless for a real experiment — read `opt.best`.
- No discrete or categorical knobs (electrolyte A vs B), and no constraints beyond the box (e.g. "components sum to 100%"). For a few discrete levels, run one optimizer per combination.

Anything behaves oddly — send `campaign.csv` and the console output.
