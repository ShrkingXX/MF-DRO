import os
import sys

from dro_runner import run_mf_single_seed

STOP_SENTINEL = "/tmp/stage1_B_STOP"

benchmark = sys.argv[1]
seed = int(sys.argv[2])
EXP_NAME = "mfdro_stage1"

tag = f"[MF-DRO {benchmark} seed{seed}]"

if os.path.exists(STOP_SENTINEL):
    print(f"{tag} SKIPPED -- STOP sentinel present (pathology detected in another seed)", flush=True)
    sys.exit(0)

print(f"{tag} Starting", flush=True)
result = run_mf_single_seed(
    EXP_NAME, benchmark, "MF-DRO", seed,
    bo_iterations=30,
    num_epochs=100,
    minimum_hf_fraction=0.25,
)

lf_frac = result["lf_fraction"]
fid_means = result.get("fid_mean_per_iter") or []
mean_fid_mean = sum(fid_means) / len(fid_means) if fid_means else float('nan')
final_regret = result["hf_regret_curve"][-1] if result["hf_regret_curve"] else None

print(f"{tag} DONE lf_fraction={lf_frac:.3f} mean_fid_mean={mean_fid_mean:.4f} final_regret={final_regret}", flush=True)

if lf_frac > 0.9:
    print(f"{tag} WARNING: lf_fraction={lf_frac:.3f} > 0.9 -- PATHOLOGICAL ALL-LF RUN. "
          f"Writing STOP sentinel so not-yet-started seeds skip. Report to user before continuing.", flush=True)
    with open(STOP_SENTINEL, "w") as f:
        f.write(f"{tag} lf_fraction={lf_frac:.3f}\n")
