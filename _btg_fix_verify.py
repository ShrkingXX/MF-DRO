import sys
from dro_runner import run_mf_single_seed

which = sys.argv[1]  # "nofloor" or "withfloor"

if which == "nofloor":
    exp_name = "mfdro_btgfix_check_nofloor"
    minimum_hf_fraction = 0.0
    title = "Step 1/2: BTG fix only, minimum_hf_fraction=0.0"
else:
    exp_name = "mfdro_btgfix_check_withfloor"
    minimum_hf_fraction = 0.25
    title = "Step 3: BTG fix + minimum_hf_fraction=0.25 restored"

print(f"########## {title} ##########")
result = run_mf_single_seed(
    exp_name, "Hartmann_6D", "MF-DRO-BTGfix", seed=42,
    bo_iterations=5,
    initial_points=3, M=10, rollouts_per_model=7, num_epochs=10,
    minimum_hf_fraction=minimum_hf_fraction,
)
trace = result["fidelity_trace"]
print(f"\nfidelity_trace: {trace}")
print(f"lf_fraction: {result['lf_fraction']:.3f}")
print(f"fid_mean_per_iter: {result['fid_mean_per_iter']}")
has_hf = 1 in trace
print(f"At least 1 HF query: {has_hf}")
print("PASS" if has_hf else "FAIL")
