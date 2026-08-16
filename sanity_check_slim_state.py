"""
Sanity checks for the slim-state redesign:
  state_hyperparams_enabled (default True) -- ablates slots 0-9 when False.
  use_roi_std_quantiles (default False) -- adds [min,p25,median,p75,max] of
    sigma_roi per GP ensemble member.

Checks:
  1. Default config (both flags at default): state shape unchanged at 18-dim
     (regression check -- every prior experiment must be unaffected).
  2. Step 1 config (state_hyperparams_enabled=False, both ROI flags False):
     8-dim state.
  3. Step 2/3 config (state_hyperparams_enabled=False, use_roi_std_quantiles=True):
     33-dim state, with roi_candidates -> last 25 values finite.
  4. Same Step 2/3 config, roi_candidates=None -> last 25 values all 0.0.
  5. use_roi_state and use_roi_std_quantiles BOTH True (not part of this
     experiment's plan, but should compose without error) -> 18+10+25=53-dim
     with full hyperparams, or 8+10+25=43-dim slim.
  6. End-to-end smoke test through run_single_seed for the Step 1 and Step
     2/3 configs.

Usage:
    python sanity_check_slim_state.py
"""
import torch

from benchmarks import get_benchmark
from checkpoint import setup_dirs
from dro_runner import _build_dro_config
from src.policy.dro import DirectRegretOptimization

SEED = 42


def make_dro(state_hyperparams_enabled=True, use_roi_state=False, use_roi_std_quantiles=False,
             use_roi_sigma_iqr=False, gp_num_models=5):
    torch.manual_seed(SEED)
    spec = get_benchmark("Hartmann_6D")
    objective = spec["make_objective"]()
    cfg = _build_dro_config(
        exp_name="sanity_slim_state", benchmark_name="Hartmann_6D", variant_name="check", seed=SEED,
        use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5, alpha_inference=None,
        lambda_rtg=1.0, rtg_warmup=3, benchmark_spec=spec,
        gp_num_models=gp_num_models, rollouts_per_iter=5, rollout_length=4, bo_iterations=10,
        initial_points=5,
        dt_hidden=32, dt_layers=1, dt_heads=1, dt_lr=1e-4, gp_kernel="rbf", gp_ard=True, verbose=False,
        rollout_acq_function="ei",
        use_roi_state=use_roi_state, use_roi_std_quantiles=use_roi_std_quantiles,
        state_hyperparams_enabled=state_hyperparams_enabled, use_roi_sigma_iqr=use_roi_sigma_iqr,
    )
    dro = DirectRegretOptimization(cfg, objective)
    dro.sample_initial_points()
    dro._initialize_models()
    return dro


def main():
    setup_dirs("sanity_slim_state")

    # === Check 1: default config regression ===
    print("=== Check 1: default config (backward compat) ===")
    dro1 = make_dro()
    state1 = dro1._extract_state(dro1.data_x, dro1.data_y, dro1.data_x.shape[0])
    print(f"  shape={tuple(state1.shape)}  EXPECT (18,)  PASS={tuple(state1.shape)==(18,)}")

    # === Check 2: Step 1 slim base (8-dim) ===
    print("\n=== Check 2: Step 1 config -- state_hyperparams_enabled=False, no ROI features ===")
    dro2 = make_dro(state_hyperparams_enabled=False)
    state2 = dro2._extract_state(dro2.data_x, dro2.data_y, dro2.data_x.shape[0])
    print(f"  shape={tuple(state2.shape)}  EXPECT (8,)  PASS={tuple(state2.shape)==(8,)}")
    print(f"  values: {[round(v,4) for v in state2.tolist()]}")

    # === Check 3: Step 2/3 slim + quantiles (33-dim), with roi_candidates ===
    print("\n=== Check 3: Step 2/3 config -- slim base + use_roi_std_quantiles, WITH roi_candidates ===")
    dro3 = make_dro(state_hyperparams_enabled=False, use_roi_std_quantiles=True)
    torch.manual_seed(SEED + 1)
    roi = dro3.bounds[0] + (dro3.bounds[1] - dro3.bounds[0]) * torch.rand(500, dro3.bo_config.input_dim, dtype=dro3.dtype)
    state3 = dro3._extract_state(dro3.data_x, dro3.data_y, dro3.data_x.shape[0], roi_candidates=roi)
    print(f"  shape={tuple(state3.shape)}  EXPECT (33,)  PASS={tuple(state3.shape)==(33,)}")
    last25 = state3[-25:].tolist()
    print(f"  last 25 (quantile) values: {[round(v,5) for v in last25]}")
    print(f"  all finite: {all(v==v for v in last25)}")
    # verify per-model quantile monotonicity: min <= p25 <= median <= p75 <= max
    monotone_ok = True
    for m in range(5):
        q = last25[m*5:(m+1)*5]
        if not (q[0] <= q[1] <= q[2] <= q[3] <= q[4]):
            monotone_ok = False
    print(f"  per-model quantiles monotone (min<=p25<=median<=p75<=max): {monotone_ok}")

    # === Check 4: Step 2/3 config, roi_candidates=None -> zero-padded ===
    print("\n=== Check 4: Step 2/3 config, roi_candidates=None ===")
    state4 = dro3._extract_state(dro3.data_x, dro3.data_y, dro3.data_x.shape[0], roi_candidates=None)
    last25_none = state4[-25:].tolist()
    print(f"  shape={tuple(state4.shape)}  EXPECT (33,)")
    print(f"  all zero: {all(v==0.0 for v in last25_none)}")

    # === Check 5: both ROI flags True, composability ===
    print("\n=== Check 5: use_roi_state AND use_roi_std_quantiles both True (slim base) ===")
    dro5 = make_dro(state_hyperparams_enabled=False, use_roi_state=True, use_roi_std_quantiles=True)
    state5 = dro5._extract_state(dro5.data_x, dro5.data_y, dro5.data_x.shape[0], roi_candidates=roi)
    print(f"  shape={tuple(state5.shape)}  EXPECT (43,) [8 base + 10 roi_state + 25 quantiles]  "
          f"PASS={tuple(state5.shape)==(43,)}")

    # === Check 7: DRO-ROISigmaIQR config (28-dim: 18 base + 10 sigma_iqr) ===
    print("\n=== Check 7: use_roi_sigma_iqr=True, full 18-dim base kept ===")
    dro7 = make_dro(use_roi_sigma_iqr=True)
    state7 = dro7._extract_state(dro7.data_x, dro7.data_y, dro7.data_x.shape[0], roi_candidates=roi)
    print(f"  shape={tuple(state7.shape)}  EXPECT (28,)  PASS={tuple(state7.shape)==(28,)}")
    last10 = state7[-10:].tolist()
    print(f"  last 10 (sigma_iqr) values: {[round(v,5) for v in last10]}")
    print(f"  all finite: {all(v==v for v in last10)}")
    # each model contributes [mean_sigma_norm, iqr_norm] -- iqr must be >= 0
    iqr_nonneg = all(last10[2*m+1] >= 0.0 for m in range(5))
    print(f"  IQR values non-negative: {iqr_nonneg}")

    print("\n=== Check 8: use_roi_sigma_iqr=True, roi_candidates=None -> zero-padded ===")
    state8 = dro7._extract_state(dro7.data_x, dro7.data_y, dro7.data_x.shape[0], roi_candidates=None)
    last10_none = state8[-10:].tolist()
    print(f"  shape={tuple(state8.shape)}  EXPECT (28,)")
    print(f"  all zero: {all(v==0.0 for v in last10_none)}")

    # === Check 6: end-to-end smoke tests ===
    print("\n=== Check 6: end-to-end smoke tests via run_single_seed ===")
    from dro_runner import run_single_seed
    r1 = run_single_seed(
        exp_name="sanity_slim_state", benchmark_name="Hartmann_6D", variant_name="Step1-slim", seed=42,
        use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
        gp_num_models=2, rollouts_per_iter=5, rollout_length=3, bo_iterations=3, initial_points=5,
        dt_hidden=32, dt_layers=1, dt_heads=1, dt_lr=1e-4, gp_kernel="rbf", gp_ard=True, verbose=False,
        rollout_acq_function="ei", state_hyperparams_enabled=False,
    )
    print(f"  Step1-slim regret_curve: {r1['regret_curve']}")
    r2 = run_single_seed(
        exp_name="sanity_slim_state", benchmark_name="Hartmann_6D", variant_name="Step23-quantiles", seed=42,
        use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
        gp_num_models=2, rollouts_per_iter=5, rollout_length=3, bo_iterations=3, initial_points=5,
        dt_hidden=32, dt_layers=1, dt_heads=1, dt_lr=1e-4, gp_kernel="rbf", gp_ard=True, verbose=False,
        rollout_acq_function="ei", state_hyperparams_enabled=False, use_roi_std_quantiles=True,
    )
    print(f"  Step23-quantiles regret_curve: {r2['regret_curve']}")
    r3 = run_single_seed(
        exp_name="sanity_slim_state", benchmark_name="Hartmann_6D", variant_name="ROISigmaIQR", seed=42,
        use_mes_reward=True, rtg_schema="floored", alpha_floor=0.5,
        gp_num_models=2, rollouts_per_iter=5, rollout_length=3, bo_iterations=3, initial_points=5,
        dt_hidden=32, dt_layers=1, dt_heads=1, dt_lr=1e-4, gp_kernel="rbf", gp_ard=True, verbose=False,
        rollout_acq_function="ei", use_roi_sigma_iqr=True,
    )
    print(f"  ROISigmaIQR regret_curve: {r3['regret_curve']}")
    print("  OK, no crashes")


if __name__ == '__main__':
    main()
