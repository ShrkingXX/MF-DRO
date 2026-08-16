"""
Phase 4 Step 2 follow-up: isolate whether the 50% neg_frac observed in the
production rtg_schema="joint" debug run (n=20, Currin_2D early-stage,
seed=42, 5 rollouts) is a small-sample fluke or a genuine early-stage
problem, by running 30 rollouts (not 5) at seed=42 directly (matching what
Step 3's real experiment will use, NOT Phase 2/3's benchmark-index-offset
seed=46) at both early-stage (5 initial points) and mid-stage (25 initial
points).

Uses the validated simulate_with_gumbel_b/compute_both_rtg from
phase2_rtg_correlation.py (post roi_candidates bugfix) rather than the
production _simulate_trajectory_joint, per the user's explicit request --
this isolates whether the phenomenon is seed-specific to the underlying
Gumbel/Thompson mechanics themselves (same in both), independent of the
production wiring already confirmed correct in the other 6 Step 2 checks.

Reports, for each stage:
  - neg_frac across all 30*4=120 step-level joint_rtg values
  - mean/std of joint_rtg[0] across 30 rollouts
  - mean/std of b_0 and b_T (Gumbel scale before/after the 4-step rollout)
    across 30 rollouts

Read-only diagnostic: does not modify any existing code.

Usage:
    python validation/phase4_step2_neg_frac_check.py
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_gumbel_quality import _make_dro
from phase2_rtg_correlation import simulate_with_gumbel_b, compute_both_rtg

BENCHMARK = "Currin_2D"
SEED = 42 # direct, matching Step 3's planned seeds 42/43/44 -- NOT Phase 2/3's benchmark-offset seed=46
N_ROLLOUTS = 30
MAX_LENGTH = 4
K_THOMPSON = 100

STAGES = [("early", 5), ("mid", 25)]


def _run(n_initial, stage_name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dro = _make_dro(BENCHMARK, n_initial, seed=SEED)
    dro._update_models()

    joint_rtg0_list = []
    all_step_joint = []
    b0_list, bT_list = [], []

    for r in range(N_ROLLOUTS):
        state = dro._extract_state(dro.data_x, dro.data_y, dro.data_x.shape[0])
        traj = simulate_with_gumbel_b(dro, gp_idx=0, initial_state=state,
                                       max_length=MAX_LENGTH, K_thompson=K_THOMPSON)
        _, joint_rtg = compute_both_rtg(traj)
        joint_rtg0_list.append(joint_rtg[0].item())
        all_step_joint.extend(joint_rtg.tolist())

        b_values = traj["gumbel_b_values"].tolist() # [T+1]: b_0, ..., b_T
        b0_list.append(b_values[0])
        bT_list.append(b_values[-1])

    all_step_joint = np.array(all_step_joint)
    neg_frac = float(np.mean(all_step_joint < 0))

    print(f"\n=== {BENCHMARK} {stage_name} (n_initial={n_initial}, seed={SEED}, N_ROLLOUTS={N_ROLLOUTS}) ===")
    print(f"  neg_frac (n={len(all_step_joint)} step-level values): {neg_frac:.3f} ({neg_frac*100:.1f}%)")
    print(f"  joint_rtg[0]: mean={np.mean(joint_rtg0_list):.4f}, std={np.std(joint_rtg0_list):.4f}")
    print(f"  b_0: mean={np.mean(b0_list):.4f}, std={np.std(b0_list):.4f}")
    print(f"  b_T: mean={np.mean(bT_list):.4f}, std={np.std(bT_list):.4f}")

    return dict(stage=stage_name, n_initial=n_initial, neg_frac=neg_frac,
                joint_rtg0_mean=float(np.mean(joint_rtg0_list)), joint_rtg0_std=float(np.std(joint_rtg0_list)),
                b0_mean=float(np.mean(b0_list)), b0_std=float(np.std(b0_list)),
                bT_mean=float(np.mean(bT_list)), bT_std=float(np.std(bT_list)))


def main():
    from checkpoint import setup_dirs
    setup_dirs("joint_rtg_validation")

    print("=" * 100)
    print(f"PHASE 4 STEP 2 FOLLOW-UP: neg_frac at seed={SEED}, early vs mid stage, {BENCHMARK}")
    print("=" * 100)

    results = {}
    for stage_name, n_initial in STAGES:
        results[stage_name] = _run(n_initial, stage_name)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    early_nf = results["early"]["neg_frac"]
    mid_nf = results["mid"]["neg_frac"]
    print(f"  early-stage neg_frac: {early_nf:.1%}")
    print(f"  mid-stage neg_frac:   {mid_nf:.1%}")
    print()
    if early_nf > 0.30 and mid_nf < 0.15:
        print("  INTERPRETATION: problem confined to the first handful of BO iterations.")
        print("  Early noise should be overwhelmed by the 30-iteration signal in Phase 4;")
        print("  the DT should recover as data accumulates. Proceed to Phase 4.")
    elif mid_nf > 0.30:
        print("  INTERPRETATION: mid-stage neg_frac is also high -- the joint formulation")
        print("  is unreliable throughout for this seed, not just early on. Phase 4 would")
        print("  likely show DRO-MES-Joint degrading relative to DRO-MES-PerStep.")
        print("  Consider whether Phase 4 is still informative before running the full experiment.")
    else:
        print("  INTERPRETATION: neither clearly early-confined nor clearly persistent --")
        print("  review the numbers directly before deciding.")


if __name__ == "__main__":
    main()
