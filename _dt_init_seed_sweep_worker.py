"""
DT weight-initialization seed sweep: does the incumbent freeze depend on the
DT's random weight initialization specifically, independent of everything
else config.seed controls (LHS initial sampling, KO-ensemble diversity
draws, GP fitting)?

Motivation: seed=44 on Hartmann_6D has landed on the EXACT SAME regret
(1.6928, frozen from iteration 0/1, never improving again) across seven
independent conditions this session -- Stage 2 v3 baseline, both Hartmann
cost-ratio overrides (c_H=4, c_H=2), both the original (buggy) and now
state-matching-fixed (v2) synthetic-expert-trajectory ablations, and both
GP-refinement test BASELINEs (EI and UCB variants) -- varying the rollout
generator, RTG scheme, cost ratio, and real-query refinement strategy, none
of which moved it. The one thing every one of those conditions shares:
torch.manual_seed(config.seed+2000) fixes the DT's initial weights
identically every time, since config.seed=44 in all of them.

Held FIXED (identical to the v2 synthetic-expert ablation): benchmark=
Hartmann_6D, training data = expert v2 (clean, state-matched -- see
_synthetic_expert_worker.py's Bug 1/2 fix), all other config. The outer
seed is pinned at OUTER_SEED=44 (the most extensively cross-validated
frozen case this session) for every run in this sweep -- LHS initial
sampling and KO-ensemble diversity draws are therefore IDENTICAL across
every dt_init_seed value (see mf_dro.py's _sample_initial_points, which
unconditionally calls torch.manual_seed(self.config.seed) regardless of
config.dt_init_seed, and the ensemble-diversity draws at the top of
__init__, which run BEFORE config.dt_init_seed's reseed and are therefore
untouched by it).

Varied ONLY: config.dt_init_seed in [0..9] -- see mf_dro.py's __init__,
which reseeds torch.manual_seed(dt_init_seed) immediately before
constructing self.dt = DecisionTransformer(...), and ONLY when this
attribute is explicitly set (a no-op otherwise, so every existing config's
behavior is unchanged).
"""
import sys
import os
import json
import types

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

dt_init_seed = int(sys.argv[1])

OUTER_SEED = 44
BENCHMARK = "Hartmann_6D"
X_STAR = [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]
N_ITERS = 100
N_TRAJ = 70  # matches M=10 * rollouts_per_model=7
EXP_NAME = "mfdro_dt_init_seed_sweep"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"seed{OUTER_SEED}__dtinit{dt_init_seed}.json")
tag = f"[dt_init_seed={dt_init_seed} outer_seed={OUTER_SEED}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, _extract_mf_state, _get_mf_state_dim

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)
X_STAR_T = torch.tensor(X_STAR, dtype=torch.float64)


def _make_expert_trajectory(mf, rng):
    """Identical to _synthetic_expert_worker.py's v2 (Bug 1/2-fixed) version."""
    T = mf.config.rollout_length
    d = mf.d
    x_start = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(d, generator=rng, dtype=torch.float64)

    xs = [x_start + (X_STAR_T - x_start) * (tau / (T - 1)) for tau in range(T)]
    ys = [f_hf(x.unsqueeze(0)).reshape(-1)[0].item() for x in xs]

    sim_hf_x, sim_hf_y = list(mf.data_hf_x), list(mf.data_hf_y)
    states = []
    for tau in range(T):
        s_tau = _extract_mf_state(
            sim_hf_x, sim_hf_y, [], [],
            mf.ko_ensemble,
            len(mf.data_hf_y), mf.config.bo_iterations,
            mf.c_L, mf.c_H, "cpu", torch.float64,
            recent_hf_frac=1.0, bounds=mf.bounds,
            # Bug A fix -- see _synthetic_expert_worker.py's identical note.
            ref_grid=mf.state_ref_grid, ref_model=mf.ko_ensemble[0],
        )
        states.append(s_tau)
        sim_hf_x.append(xs[tau])
        sim_hf_y.append(ys[tau])

    rtg_raw = torch.tensor([ys[-1] - ys[tau] for tau in range(T)], dtype=torch.float64)
    rtg = rtg_raw / max(rtg_raw[0].item(), 1e-8)
    costs = torch.full((T,), mf.c_H, dtype=torch.float64)
    btg = costs.flip(0).cumsum(0).flip(0)
    actions_x = torch.stack([
        ((x - mf.bounds[0]) / (mf.bounds[1] - mf.bounds[0])).clamp(0.0, 1.0) for x in xs
    ])
    actions_ell = torch.ones(T, dtype=torch.long)

    return {
        "states": torch.stack(states),
        "actions_x": actions_x,
        "actions_ell": actions_ell,
        "rtg": rtg,
        "btg": btg,
        "y_values": torch.tensor(ys, dtype=torch.float64),
        "lf_fraction": 0.0,
        "neg_rtg_frac": float((rtg < 0).float().mean()),
    }


def _expert_generate_rollout_batch(self):
    rng = torch.Generator().manual_seed(hash((self.config.seed, len(self.data_hf_y))) % (2 ** 31))
    return [_make_expert_trajectory(self, rng) for _ in range(N_TRAJ)]


print(f"{tag} Starting (n_iters={N_ITERS}, n_traj={N_TRAJ})", flush=True)

torch.manual_seed(OUTER_SEED)
np.random.seed(OUTER_SEED)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, f"dtinit{dt_init_seed}", OUTER_SEED,
    bo_iterations=N_ITERS, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=1e9, initial_hf=36, initial_lf=60,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    known_optimal_x=X_STAR,
)
config.seed = OUTER_SEED
config.dt_init_seed = dt_init_seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
mf._generate_rollout_batch = types.MethodType(_expert_generate_rollout_batch, mf)

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
frozen = n_improved == 0
out = dict(result, outer_seed=OUTER_SEED, dt_init_seed=dt_init_seed,
           incumbent_improved_count=n_improved, distinct_regret_values=distinct,
           final_regret=rc[-1], frozen=frozen)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE n_improved={n_improved} final_regret={rc[-1]:.4f} frozen={frozen} "
      f"distinct={distinct}/{len(rc)}", flush=True)
