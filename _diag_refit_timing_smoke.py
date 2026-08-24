import time
import torch
import numpy as np

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)
BENCHMARK = "Hartmann_6D"
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

torch.manual_seed(44)
np.random.seed(44)
config = _build_mf_dro_config(
    "timing_smoke", BENCHMARK, "refit_test", 44,
    bo_iterations=1, num_epochs=2,
    M=2, rollouts_per_model=1, rollout_length=2,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=9999, initial_hf=10, initial_lf=15,
    dkl_threshold=9999, bes_delta=0.0,
    use_real_rollout_queries=True, refit_hyperparams_in_rollout=True,
)
config.seed = 44
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
mf._sample_initial_points()
mf._update_ko_ensemble()

t0 = time.time()
batch = mf._generate_rollout_batch()
t1 = time.time()
print(f"_generate_rollout_batch (M=2, rollouts_per_model=1, rollout_length=2, refit=True): "
      f"{t1-t0:.2f}s for {len(batch)} rollouts")
print(f"Per rollout-step estimate: {(t1-t0)/(len(batch)*2):.3f}s/step")
