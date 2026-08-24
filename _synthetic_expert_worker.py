"""
Synthetic-expert-demonstration ablation: does MF-DRO's DT collapse/drift
pathology persist even when trained on hand-designed, unambiguously-correct
trajectories, or is it specific to the KO-GP+MES rollout generator?

Design (see conversation for full rationale):
  - Keeps the real pipeline's state extraction, _train_dt, _propose_next_query,
    and real benchmark evaluation entirely UNCHANGED.
  - Replaces ONLY _generate_rollout_batch: instead of KO-GP fantasy-sampled
    MES rollouts, generates N_TRAJ=70 (matching M*rollouts_per_model)
    deterministic "expert" trajectories per BO iteration:
      * x_start ~ Uniform([0,1]^d), independent per trajectory (wide state
        coverage so the DT must learn a state-conditioned rule, not
        memorize one path)
      * x_tau = lerp(x_start, x_star, tau/(T-1)) over rollout_length steps
        -- arrives exactly at x_star on the last step
      * fidelity: always HF (disclosed simplification -- recent_hf_frac is
        always 1.0 in this synthetic training data, vs ~0.97 LF at real
        Hartmann_6D inference; accepted since this ablation targets
        location-policy quality, not fidelity-mix)
      * y_tau = f_hf(x_tau), the TRUE benchmark value, no GP sampling --
        the demonstration is genuinely correct, not just GP-believed-correct
      * RTG[tau] = (y_final - y_tau) / (y_final - y_0) -- deterministic,
        zero trajectory-level luck by construction: two trajectories at the
        same relative position get the same RTG, always
      * BTG: backward cumsum of true HF cost, same formula as the real
        pipeline
      * state vector: _extract_mf_state using the REAL, currently-fitted
        self.ko_ensemble for the raw-KO-hyperparameter block (not a
        synthetic ensemble), AND best_value_HF/best_position_HF seeded from
        the REAL accumulated mf.data_hf_x/mf.data_hf_y (fixed -- an earlier
        version seeded these from a single synthetic point, [x_start], which
        made this half of the state vector describe a tiny, unrealistic
        history while the KO-hyperparameter half described the real,
        36+-point one; the synthetic trajectory's own steps are appended on
        top of that real starting point as the loop progresses). recent_hf_frac
        stays fixed at 1.0 (this trajectory's own steps are always HF; see
        the fidelity note above).

If the DT trained on this data STILL shows the drift-away-from-x*/value-
collapse pattern at real inference, the problem is in the DT's own
architecture/training dynamics, not the rollout generator. If it doesn't,
the rollout generator (KO-GP+MES) is implicated instead.

NOTE: RTG here (normalized remaining improvement in true y, always in
[0,1], monotonic at the endpoints) is a DIFFERENT quantity than the real
pipeline's MES-entropy RTG (log(b_tau/b_T), often negative, no monotonic
structure) -- rtg[0]==1.0 by construction here, which happens to land in
the same rough range as real rtg_target (empirically ~0.6-1.6, mean ~1.17
on a real Hartmann_6D run), but the per-step shape genuinely differs. Not
fixed in this version -- a separate, more fundamental design question than
the state-seeding bug above.
"""
import sys
import os
import json
import types

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

seed = int(sys.argv[1])
BENCHMARK = sys.argv[2] if len(sys.argv) > 2 else "Hartmann_6D"

# Sanity-check control (per user request): if Hartmann_6D's frozen-despite-
# clean-data result is a bug in THIS ablation's own plumbing, Ackley_10D
# should show the same spurious freeze. If it's a genuine finding, Ackley
# (the "healthy" benchmark throughout this session -- structurally
# unimodal, no competing local optima, unlike Hartmann's 4-bump landscape)
# should converge close to perfectly under a clean expert teacher, since
# every prior diagnostic this session found Ackley's DT training dynamics
# meaningfully healthier than Hartmann's.
BENCHMARK_CFG = {
    "Hartmann_6D": dict(x_star=[0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573],
                         initial_hf=36, initial_lf=60),
    "Ackley_10D":  dict(x_star=[0.5] * 10, initial_hf=60, initial_lf=100),
}
cfg_bm = BENCHMARK_CFG[BENCHMARK]

N_ITERS = 100
N_TRAJ = 70  # matches M=10 * rollouts_per_model=7
# v2: state-seeding fix (Bug 1/2/3 -- see module docstring). Separate
# exp_name so the original (buggy) results at mfdro_synthetic_expert stay
# intact for a direct before/after comparison, rather than being
# overwritten or silently skipped by the existing-file check below.
EXP_NAME = "mfdro_synthetic_expert_v2"

X_STAR = torch.tensor(cfg_bm["x_star"], dtype=torch.float64)

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__synthetic_expert__seed{seed}.json")
tag = f"[synthetic-expert {BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, _extract_mf_state, _get_mf_state_dim

torch.set_default_dtype(torch.float64)
# Ackley_10D's HF/LF pair uses "Ackley_10D_HF"/"Ackley_10D_LF" -- no bare-name
# collision issue here (that only affects SF-DRO's lookup and the
# _KNOWN_OPTIMAL_X auto-dict, both bypassed below via the explicit
# known_optimal_x override).
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)


def _make_expert_trajectory(mf, rng):
    T = mf.config.rollout_length
    d = mf.d
    x_start = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(d, generator=rng, dtype=torch.float64)

    xs = [x_start + (X_STAR - x_start) * (tau / (T - 1)) for tau in range(T)]
    ys = [f_hf(x.unsqueeze(0)).reshape(-1)[0].item() for x in xs]

    # Bug 1/3 fix: seed from the REAL accumulated HF data (list(...) copies
    # so appending below never mutates mf.data_hf_x/mf.data_hf_y themselves),
    # matching simulate_mf_trajectory's own sim_hf_x = list(real_hf_x_list)
    # (mf_dro.py) -- previously seeded from a single synthetic point
    # ([x_start], [ys[0]]), making best_value_HF/best_position_HF describe a
    # completely different (and far smaller) history than what real
    # inference presents alongside the SAME mf.ko_ensemble features (which
    # were always fit on the real data regardless of this bug). Bug 2 fix:
    # no longer pre-seeding with [x_start] separately from the loop's own
    # append -- xs[0] (== x_start exactly) is now appended exactly once, at
    # tau=0, like every other tau, instead of twice.
    sim_hf_x, sim_hf_y = list(mf.data_hf_x), list(mf.data_hf_y)
    states = []
    for tau in range(T):
        s_tau = _extract_mf_state(
            sim_hf_x, sim_hf_y, [], [],
            mf.ko_ensemble,
            len(mf.data_hf_y), mf.config.bo_iterations,
            mf.c_L, mf.c_H, "cpu", torch.float64,
            recent_hf_frac=1.0, bounds=mf.bounds,
            # Bug A fix's ref_grid/ref_model params: mf.state_ref_grid is
            # the same fixed grid the real pipeline uses. ref_model uses
            # mf.ko_ensemble[0] (matching real inference's own convention)
            # rather than a per-trajectory current_ko -- this synthetic
            # ablation has no analogue of the real rollout's per-ensemble-
            # member/per-step-conditioned model, so every one of this
            # iteration's 70 synthetic trajectories shares the same
            # reference-grid features (a disclosed simplification, same
            # spirit as recent_hf_frac always being 1.0 here).
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

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, "synthetic_expert", seed,
    bo_iterations=N_ITERS, num_epochs=10,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=1e9, initial_hf=cfg_bm["initial_hf"], initial_lf=cfg_bm["initial_lf"],
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
    known_optimal_x=cfg_bm["x_star"],
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
mf._generate_rollout_batch = types.MethodType(_expert_generate_rollout_batch, mf)

result = mf.run()

rc = result["hf_regret_curve"]
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
distinct = len(set(f"{r:.6f}" for r in rc))
out = dict(result, seed=seed, incumbent_improved_count=n_improved,
           distinct_regret_values=distinct, final_regret=rc[-1])
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE final_regret={rc[-1]:.4f} n_iters={len(rc)} n_improved={n_improved} "
      f"distinct={distinct}/{len(rc)} lf_fraction={result['lf_fraction']:.3f}", flush=True)
