from dro_runner import run_single_seed

EXP_NAME = "mfdro_stage1"
BENCHMARKS = ["Currin_2D", "Hartmann_6D"]
SEEDS = [42, 43, 44]

for benchmark in BENCHMARKS:
    for seed in SEEDS:
        print(f"[SF-DRO] Starting {benchmark} seed{seed}", flush=True)
        result = run_single_seed(
            EXP_NAME, benchmark, "SF-DRO-rotate-MES", seed,
            use_mes_reward=True,
            rtg_schema="floored",
            alpha_floor=0.5,
            rollout_acq_function="rotate",
            gp_num_models=5,
            rollouts_per_iter=75,
            rollout_length=4,
            bo_iterations=30,
        )
        final_regret = result["regret_curve"][-1] if result["regret_curve"] else None
        print(f"[SF-DRO] DONE {benchmark} seed{seed} final_regret={final_regret}", flush=True)

print("[SF-DRO] ALL RUNS COMPLETE", flush=True)
