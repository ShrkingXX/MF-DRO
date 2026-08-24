import math
import torch
from types import SimpleNamespace

from benchmarks import get_benchmark
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)

BENCHMARK = "Hartmann_6D"
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)


def make_config(**overrides):
    base = dict(
        exp_name="unit_check", benchmark_name=BENCHMARK, variant_name="check",
        seed=42, M=3, rollout_length=8, rollouts_per_model=2, num_epochs=1,
        initial_points=5, dt_hidden=16, dt_layers=1, dt_heads=1, dt_lr=1e-4,
        lambda_fid=1.0, alpha_rtg=0.5, alpha_btg=0.5, max_seq_length=80,
        minimum_hf_fraction=0.25, real_hf_warmup=2, cost_budget=None,
        initial_hf=18, initial_lf=30, use_sequential_init=False,
        use_rtg_grounding=False, dkl_threshold=9999, bes_delta=0.0,
        c_L=1.0, c_H=8.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# Check 1: flag OFF -> bit-for-bit identical to no flag at all.
cfg_no_flag = make_config()
cfg_off = make_config(use_lf_screened_init=False)

mf_no_flag = DirectMFRegretOptimization(cfg_no_flag, f_hf, f_lf, bounds)
mf_no_flag._sample_initial_points()

mf_off = DirectMFRegretOptimization(cfg_off, f_hf, f_lf, bounds)
mf_off._sample_initial_points()

assert torch.allclose(torch.stack(mf_no_flag.data_hf_x), torch.stack(mf_off.data_hf_x))
assert mf_no_flag.data_hf_y == mf_off.data_hf_y
assert torch.allclose(torch.stack(mf_no_flag.data_lf_x), torch.stack(mf_off.data_lf_x))
assert mf_no_flag.data_lf_y == mf_off.data_lf_y
assert mf_no_flag.cumulative_cost == mf_off.cumulative_cost
print("Check 1 PASSED: use_lf_screened_init unset/False -> bit-for-bit identical")

# Check 2: flag ON -> correct counts and cost accounting.
cfg_on = make_config(use_lf_screened_init=True)
mf_on = DirectMFRegretOptimization(cfg_on, f_hf, f_lf, bounds)
mf_on._sample_initial_points()

assert len(mf_on.data_hf_x) == cfg_on.initial_hf == 18
assert len(mf_on.data_lf_x) == cfg_on.initial_lf == 30
expected_cost = cfg_on.initial_hf * cfg_on.c_H + cfg_on.initial_lf * cfg_on.c_L
assert abs(mf_on.cumulative_cost - expected_cost) < 1e-9, (mf_on.cumulative_cost, expected_cost)
assert mf_on.initial_hf_values == mf_on.data_hf_y
print(f"Check 2 PASSED: n_hf={len(mf_on.data_hf_x)} n_lf={len(mf_on.data_lf_x)} "
      f"cumulative_cost={mf_on.cumulative_cost} (expected {expected_cost})")

# Check 3: screened HF points are NOT the same as plain random LHS at the
# same seed (i.e. the new code path actually did something different).
from src.utils.init_design import lhs_design
X_hf_plain_lhs = lhs_design(bounds, mf_on.d, cfg_on.initial_hf, cfg_on.seed, seed_offset=0)
X_hf_screened = torch.stack(mf_on.data_hf_x)
assert not torch.allclose(X_hf_plain_lhs, X_hf_screened)
print("Check 3 PASSED: screened HF locations differ from plain random LHS")

# Check 4: screened HF points should, on average, land at meaningfully
# higher true-f_lf values than a random LHS draw of the same size (the
# whole point of screening -- weak statistical check, not a hard guarantee
# for any single point, but should hold in aggregate over 18 points).
lf_vals_screened = torch.tensor([f_lf(x.unsqueeze(0)).reshape(-1)[0].item() for x in X_hf_screened])
lf_vals_random = torch.tensor([f_lf(x.unsqueeze(0)).reshape(-1)[0].item() for x in X_hf_plain_lhs])
print(f"Check 4 (informational): mean f_lf at screened HF points = {lf_vals_screened.mean().item():.4f} "
      f"vs random LHS = {lf_vals_random.mean().item():.4f}")
assert lf_vals_screened.mean().item() > lf_vals_random.mean().item(), \
    "screened HF points should on average find higher LF values than random"
print("Check 4 PASSED: screened HF points average higher f_lf than random LHS")

print("\nALL CHECKS PASSED")
