"""
Phase 4 Step 3 diagnosis: why did DRO-MES-Joint go flat (0-1 improvement in
30 iterations) on 2-3 seeds while DRO-MES-PerStep improved continuously?

Hypothesis to test: the floored-target formula
    rtg_target = max(batch_max_rtg, alpha_floor * running_max_rtg)
picks the MAXIMUM of 75 stochastic rollouts' RTG[0] each iteration, then
ratchets running_max_rtg upward forever (never decays). For PerStep, RTG[0]
is a bounded sum of ~4 small positive MES rewards (~0.01-0.05 each) -- a
well-behaved, low-variance quantity, so the batch max is a mild, learnable
extrapolation. For Joint, RTG[0] = log(b_0/b_T) is a log-ratio of two
independently Thompson/Gumbel-MLE-fit scale parameters (K=100 samples each)
-- a noisier quantity with a heavier tail, so the batch max of 75 draws can
be a rare, largely uninformative outlier relative to the bulk of the
training distribution the DT actually learned from. If the resulting
rtg_target sits many population-std above the batch's own mean, the DT is
being asked to extrapolate to an out-of-distribution RTG at inference time,
which plausibly produces a degenerate/stuck action.

This script reproduces ONE real iteration's rollout batch (matching
_propose_next_candidate's exact logic, via the same production
_simulate_trajectory_joint) for both variants at a comparable point in the
optimization, and reports:
  - full batch_rtg0_list summary (mean, std, max, z-score of max)
  - the resulting rtg_target and its z-score relative to the batch
  - how many std above the batch mean the actual inference-time target sits

Read-only diagnostic: does not modify dro.py.

Usage:
    python validation/phase4_step3_diagnosis.py
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import get_benchmark
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

BENCHMARK = "Currin_2D"
SEED = 42
N_INITIAL = 5
NUM_ROLLOUTS = 75
ROLLOUT_LENGTH = 4


def _make_dro(rtg_schema, seed=SEED, n_initial=N_INITIAL):
    benchmark_spec = get_benchmark(BENCHMARK)
    cfg = _build_dro_config(
        exp_name="joint_rtg_validation_diag", benchmark_name=BENCHMARK, variant_name=f"diag-{rtg_schema}", seed=seed,
        use_mes_reward=True, rtg_schema=rtg_schema, alpha_floor=0.5, alpha_inference=None,
        lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=benchmark_spec,
        gp_num_models=5, rollouts_per_iter=NUM_ROLLOUTS, rollout_length=ROLLOUT_LENGTH,
        bo_iterations=30, initial_points=n_initial,
        dt_hidden=128, dt_layers=4, dt_heads=4, dt_lr=1e-4, gp_kernel="rbf", gp_ard=True, verbose=False,
    )
    objective = benchmark_spec["make_objective"]()
    dro = DirectRegretOptimization(cfg, objective)
    dro.sample_initial_points()
    dro._initialize_models()
    dro._update_models()
    return dro


def collect_batch_rtg0(dro, use_joint):
    """Reproduces _propose_next_candidate's rollout-generation loop exactly
    (same GP-ensemble iteration, same rollouts_per_gp split), just also
    returns the full batch_rtg0_list instead of discarding it after
    _compute_rtg_target."""
    num_rollouts = dro.simulation_config.num_rollouts
    rollout_length = dro.simulation_config.max_rollout_length
    batch_rtg0_list = []
    for gp_idx in range(len(dro.gp_ensemble)):
        rollouts_per_gp = max(1, num_rollouts // len(dro.gp_ensemble))
        for _ in range(rollouts_per_gp):
            state = dro._extract_state(dro.data_x, dro.data_y, dro.data_x.shape[0])
            if use_joint:
                traj = dro._simulate_trajectory_joint(gp_idx, state, rollout_length)
                batch_rtg0_list.append(traj["joint_rtg"][0].item())
            else:
                traj = dro._simulate_trajectory(gp_idx, state, rollout_length)
                batch_rtg0_list.append(traj["rewards"].sum().item())
    return batch_rtg0_list


def analyze(label, dro, use_joint):
    print(f"\n=== {label} (iteration 1, n_initial={N_INITIAL}) ===")
    batch = np.array(collect_batch_rtg0(dro, use_joint))
    rtg_target, batch_max_rtg = dro._compute_rtg_target(list(batch))

    mean, std, mx = batch.mean(), batch.std(), batch.max()
    z_max = (mx - mean) / std if std > 0 else float("nan")
    z_target = (rtg_target - mean) / std if std > 0 else float("nan")

    print(f"  batch_rtg0_list (n={len(batch)}): mean={mean:.4f} std={std:.4f} min={batch.min():.4f} max={mx:.4f}")
    print(f"  batch_max z-score (how many std above mean): {z_max:.2f}")
    print(f"  rtg_target (fed to DT at inference) = {rtg_target:.4f}")
    print(f"  rtg_target z-score relative to this batch's own distribution: {z_target:.2f}")
    print(f"  fraction of batch >= rtg_target: {np.mean(batch >= rtg_target):.4f} "
          f"({int(np.sum(batch >= rtg_target))}/{len(batch)})")
    return dict(mean=mean, std=std, max=mx, z_max=z_max, rtg_target=rtg_target, z_target=z_target)


def main():
    from checkpoint import setup_dirs
    setup_dirs("joint_rtg_validation_diag")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dro_joint = _make_dro("joint")
    r_joint = analyze("DRO-MES-Joint", dro_joint, use_joint=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dro_perstep = _make_dro("floored")
    r_perstep = analyze("DRO-MES-PerStep", dro_perstep, use_joint=False)

    print("\n=== COMPARISON ===")
    print(f"  PerStep: rtg_target z-score = {r_perstep['z_target']:.2f} std above batch mean "
          f"(coefficient of variation std/mean = {r_perstep['std']/max(r_perstep['mean'],1e-9):.3f})")
    print(f"  Joint:   rtg_target z-score = {r_joint['z_target']:.2f} std above batch mean "
          f"(coefficient of variation std/mean = {r_joint['std']/max(r_joint['mean'],1e-9):.3f})")
    print()
    print("  If Joint's rtg_target sits many more std above its own batch's mean than")
    print("  PerStep's does (and/or Joint's batch has much higher relative variance),")
    print("  that's direct evidence the DT is asked to extrapolate to a target far")
    print("  outside what it was actually trained on, for the Joint schema specifically.")


if __name__ == "__main__":
    main()
