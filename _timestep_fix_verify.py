from dro_runner import run_mf_single_seed

print("########## Timestep fix verification: Hartmann_6D, 6 iterations, smoke-scale ##########")
result = run_mf_single_seed(
    "mfdro_timestepfix_check", "Hartmann_6D", "MF-DRO-TimestepFix", seed=42,
    bo_iterations=6,
    initial_points=3, M=10, rollouts_per_model=7, num_epochs=10,
    minimum_hf_fraction=0.25,
)
print(f"\nfidelity_trace: {result['fidelity_trace']}")
print(f"fid_mean_per_iter: {result['fid_mean_per_iter']}")
