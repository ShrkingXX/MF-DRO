import math
from dro_runner import run_mf_single_seed

SMOKE_OVERRIDES = dict(
    initial_points=3, M=10, rollouts_per_model=7, num_epochs=10,
    minimum_hf_fraction=0.0,
)

print("########## Hartmann_6D smoke test: Takeno MF-MES formula, minimum_hf_fraction=0.0 (formula only) ##########")
result = run_mf_single_seed(
    "mfdro_takeno_sidecheck", "Hartmann_6D", "MF-DRO-TakenoMES", seed=42,
    bo_iterations=5, **SMOKE_OVERRIDES
)
trace = result["fidelity_trace"]
print(f"fidelity_trace: {trace}")
print(f"lf_fraction: {result['lf_fraction']:.3f}")
print(f"fid_mean_per_iter: {result['fid_mean_per_iter']}")
has_hf = 1 in trace
print(f"At least 1 HF query without override: {has_hf}")
print("PASS" if has_hf else "FAIL (no HF query appeared naturally)")
