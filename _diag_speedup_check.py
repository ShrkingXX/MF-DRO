import time
import torch

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, compute_joint_mf_mes

BENCHMARK = "Hartmann_6D"
SEED = 42

config = _build_mf_dro_config(
    "diag_speedup_check", BENCHMARK, "MF-DRO", SEED,
    bo_iterations=1, num_epochs=100,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
)

hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor(
    [hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64
)

mf_dro = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)
mf_dro._sample_initial_points()
mf_dro._update_ko_ensemble()

print("=== Timing check: _generate_rollout_batch() ===")
t0 = time.time()
batch = mf_dro._generate_rollout_batch()
elapsed = time.time() - t0
print(f"Rollout generation: {elapsed:.1f}s")
print(f"EXPECT < 300s: {'PASS' if elapsed < 300 else 'FAIL'}")

print("\n=== Correctness check: compute_joint_mf_mes ===")
ko = mf_dro.ko_ensemble[0]
from src.policy.mf_dro import _compute_mf_roi_candidates
roi_cands = _compute_mf_roi_candidates(ko, bounds)
x, ell, scores = compute_joint_mf_mes(ko, roi_cands, c_H=mf_dro.c_H, c_L=mf_dro.c_L)

print(f"roi_cands.shape: {tuple(roi_cands.shape)}")
print(f"scores.shape: {tuple(scores.shape)}  EXPECT (<=50, 2): "
      f"{'PASS' if scores.shape[0] <= 50 and scores.shape[1] == 2 else 'FAIL'}")
all_nonneg = bool((scores >= 0).all().item())
print(f"all scores >= 0: {all_nonneg}  {'PASS' if all_nonneg else 'FAIL'}")
print(f"ell in {{0,1}}: {ell}  {'PASS' if ell in (0, 1) else 'FAIL'}")
