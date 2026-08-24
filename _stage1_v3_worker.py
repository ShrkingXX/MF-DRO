import sys
import statistics as st

from dro_runner import run_mf_single_seed

benchmark = sys.argv[1]
seed = int(sys.argv[2])
EXP_NAME = "mfdro_stage1_v3"

tag = f"[MF-DRO-v3 {benchmark} seed{seed}]"

print(f"{tag} Starting (real_hf_warmup=2, minimum_hf_fraction=0.25, "
      f"bo_iterations=30, num_epochs=100)", flush=True)
result = run_mf_single_seed(
    EXP_NAME, benchmark, "MF-DRO", seed,
    bo_iterations=30,
    num_epochs=100,
    minimum_hf_fraction=0.25,
    real_hf_warmup=2,
)

lf_frac = result["lf_fraction"]
trace = result["fidelity_trace"]
n_hf = sum(1 for e in trace if e == 1)
fid_means = result.get("fid_mean_per_iter") or []
mean_fid_mean = sum(fid_means) / len(fid_means) if fid_means else float('nan')
fid_mean_std = st.stdev(fid_means) if len(fid_means) > 1 else float('nan')
final_regret = result["hf_regret_curve"][-1] if result["hf_regret_curve"] else None

print(f"{tag} DONE seed={seed} | benchmark={benchmark} | "
      f"final_regret={final_regret:.4f} | lf_fraction={lf_frac:.3f} | "
      f"n_hf_queries={n_hf} | mean_fid_mean={mean_fid_mean:.4f} | "
      f"fid_mean_std={fid_mean_std:.4f}", flush=True)
