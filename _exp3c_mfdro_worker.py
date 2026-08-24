import sys
from dro_runner import run_mf_single_seed

seed = int(sys.argv[1])
EXP_NAME = "mfdro_exp3c"
BENCHMARK = "Hartmann_6D"

tag = f"[MF-DRO-3C {BENCHMARK} seed{seed}]"
print(f"{tag} Starting (rollout candidates: 200 unfiltered; "
      f"rho optimizer: momentum-persisting)", flush=True)

result = run_mf_single_seed(
    EXP_NAME, BENCHMARK, "MF-DRO", seed,
    bo_iterations=30,
    num_epochs=100,
    minimum_hf_fraction=0.25,
    real_hf_warmup=2,
)

final_regret = result["hf_regret_curve"][-1]
lf_frac = result["lf_fraction"]
n_hf = sum(1 for e in result["fidelity_trace"] if e == 1)
print(f"{tag} DONE final_regret={final_regret:.4f} lf_fraction={lf_frac:.3f} "
      f"n_hf_queries={n_hf}", flush=True)
