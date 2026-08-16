from dro_runner import run_mf_single_seed

EXP_NAME = "mfdro_stage1"
BENCHMARKS = ["Currin_2D", "Hartmann_6D"]
SEEDS = [42, 43, 44]

stopped = False
for benchmark in BENCHMARKS:
    if stopped:
        break
    for seed in SEEDS:
        print(f"[MF-DRO] Starting {benchmark} seed{seed}", flush=True)
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
        print(f"[MF-DRO] DONE {benchmark} seed{seed} lf_fraction={lf_frac:.3f} "
              f"mean_fid_mean={mean_fid_mean:.4f} final_regret={final_regret}", flush=True)

        # PATHOLOGICAL LF MONITOR (seed-level, per Stage 1 spec)
        if lf_frac > 0.9:
            print(f"[MF-DRO] WARNING: {benchmark} seed{seed} lf_fraction={lf_frac:.3f} > 0.9 "
                  f"-- PATHOLOGICAL ALL-LF RUN. STOPPING MF-DRO orchestration. "
                  f"Report to user before continuing.", flush=True)
            stopped = True
            break

if not stopped:
    print("[MF-DRO] ALL RUNS COMPLETE", flush=True)
else:
    print("[MF-DRO] STOPPED EARLY due to LF pathology", flush=True)
