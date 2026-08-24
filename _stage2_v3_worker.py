"""
Stage 2 v3: full comparison with the confirmed fixes applied --
lognormal-prior lengthscale (ko_gp.py default), noise_lb=1e-2 (default),
diverse ensemble init (mf_dro.py default), raw KO hyperparameter state
(mf_dro.py default) -- plus dkl_threshold=9999 (explicit, since
initial_hf for Hartmann/Borehole is below the class default of 30 but
would cross it mid-run without this) and num_epochs=10 for MF-DRO
(updated from 100 based on the num_epochs ablation: lower epochs gives
~5x more incumbent-improvement events, though it's a dosage effect, not
a full fix for benchmarks with structurally multimodal landscapes).

SF-DRO's num_epochs is left at its own native default (100, hardcoded in
dro_runner.py's _build_dro_config, not exposed as an override) -- the
ablation only tested MF-DRO, so lowering SF-DRO's epochs would be an
unvalidated, unrequested change.

initial_hf/initial_lf: 2.0x the original per-benchmark 3*d/5*d asymmetric
design (Currin 12/20, Hartmann 36/60, Borehole 48/80), per the init-size
tuning sweep (_init_size_tuning_worker.py, 27 runs: 0.5x/1x/2x across all 3
benchmarks x 3 seeds) -- 1x (the original sizing) was the WORST multiplier
on both benchmarks it wasn't already saturated on. Applied UNIFORMLY to
every method's initial-data budget, matching this codebase's "same
initialization across methods" requirement.

TERMINATION: fixed N_ITERS=100 real BO iterations for every method/
benchmark/seed, NOT a cost_budget. A cost_budget stopping condition makes
final iteration count depend on how much LF vs HF a given policy/seed
happens to choose (a heavily-LF run gets far more iterations than a
heavily-HF run within the same budget), which risks under-converged
regret curves for HF-heavy runs and isn't controlled across methods/seeds.
N_ITERS=100 is not an arbitrary choice: it exactly reproduces what the
OLD cost_budget values (300/800/200) already implied for SF-DRO, which is
HF-only (cost_budget/c_H = 100 for all three benchmarks: 300/3, 800/8,
200/2) -- so this keeps the same effective scale that was already
implicitly the design target, just applied as the actual stopping
condition for every method instead of only working out that way for one
of them. Cost is NOT a stopping condition anymore, but is still tracked
per iteration (cost_curve, already existing per-iteration output) for
cost-weighted regret / log-regret-vs-cost analysis after the fact --
final total cost now varies by method/seed (depends on realized LF/HF
mix) instead of being fixed, which the analysis needs to account for
(e.g. step-interpolating onto a shared cost grid, as _stage2_plots.py's
regret-vs-cost plot already does).

cost_budget is passed as a large non-binding value (1e9) for MF-DRO/
MF-DRO+LFScreen (DirectMFRegretOptimization.run()'s cost_budget check
crashes on None -- getattr's default only applies when the attribute is
absent, and _build_mf_dro_config always sets it) and as None for the
mf_baselines.py optimizers (their own run() loops handle
cost_budget=None natively as unlimited) -- in both cases N_ITERS is the
only thing that can actually stop the loop.

Ackley_10D: HF:LF cost ratio = 10:1 (HF cost=10.0, LF cost=1.0, set in
benchmarks.py). Tests whether MF-DRO's incumbent-freeze pathology is
specific to narrow-basin/multimodal benchmarks (Hartmann_6D) or
architectural -- Ackley is smooth and radially symmetric around its
optimum, unlike Hartmann's several competing local-optima bumps. Init
sizing follows the same 3*d/5*d-then-2x convention as the other three
(d=10 here: base 30/50, doubled to 60/100).

known_optimal_x=[0.5]*10 passed EXPLICITLY (not via dro_runner.py's
_KNOWN_OPTIMAL_X auto-lookup) for Ackley_10D only: that dict's own
"Ackley_10D" entry ([0.0]*10) is for a DIFFERENT, pre-existing SF-only
benchmark with the same base name but a totally different domain
([-32.768,32.768]^10 vs this MF pair's [0,1]^10) and a different true
optimum location -- using the auto-lookup here would silently feed the
wrong x* into query_dist_to_xstar_per_iter. See _build_mf_dro_config's
known_optimal_x param docstring in dro_runner.py.
"""
import sys
import os
import json

N_ITERS = 100

BENCHMARK_CONFIG = {
    "Currin_2D":   dict(d=2,  initial_hf=12, initial_lf=20),
    "Hartmann_6D": dict(d=6,  initial_hf=36, initial_lf=60),
    "Borehole_8D": dict(d=8,  initial_hf=48, initial_lf=80),
    "Ackley_10D":  dict(d=10, initial_hf=60, initial_lf=100),
}

# Explicit known_optimal_x overrides -- see docstring above for why
# Ackley_10D can't use dro_runner.py's own _KNOWN_OPTIMAL_X auto-lookup.
KNOWN_OPTIMAL_X_OVERRIDE = {
    "Ackley_10D": [0.5] * 10,
}

EXP_NAME = "mfdro_stage2_v3"

method = sys.argv[1]      # SF-DRO | MF-GP-UCB | MF-MI-Greedy | Greedy-MES | MF-DRO | MF-DRO+LFScreen
benchmark = sys.argv[2]
seed = int(sys.argv[3])

cfg = BENCHMARK_CONFIG[benchmark]
EXP_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(EXP_DIR, exist_ok=True)
out_path = os.path.join(EXP_DIR, f"{method}__{benchmark}__seed{seed}.json")

tag = f"[{method} {benchmark} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

print(f"{tag} Starting (n_iters={N_ITERS}, "
      f"initial_hf={cfg['initial_hf']}, initial_lf={cfg['initial_lf']})", flush=True)

if method in ("MF-DRO", "MF-DRO+LFScreen"):
    from dro_runner import run_mf_single_seed
    result = run_mf_single_seed(
        EXP_NAME, benchmark, method, seed,
        bo_iterations=N_ITERS,
        num_epochs=10,
        minimum_hf_fraction=0.25,
        real_hf_warmup=2,
        cost_budget=1e9,  # non-binding; N_ITERS is the only stop condition
        initial_hf=cfg["initial_hf"],
        initial_lf=cfg["initial_lf"],
        dkl_threshold=9999,
        bes_delta=0.0,
        rollout_length=8,
        use_lf_screened_init=(method == "MF-DRO+LFScreen"),
        known_optimal_x=KNOWN_OPTIMAL_X_OVERRIDE.get(benchmark),
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    final_regret = result["hf_regret_curve"][-1]
    lf_frac = result["lf_fraction"]
    n_iters = len(result["fidelity_trace"])
    print(f"{tag} DONE final_regret={final_regret:.4f} lf_fraction={lf_frac:.3f} "
          f"n_iters={n_iters} final_cost={result['cost_curve'][-1]:.1f}", flush=True)

elif method == "SF-DRO":
    from checkpoint import setup_dirs
    setup_dirs(EXP_NAME)
    from dro_runner import run_single_seed
    hf_spec_cost = {"Currin_2D": 3.0, "Hartmann_6D": 8.0, "Borehole_8D": 2.0,
                     "Ackley_10D": 10.0}[benchmark]
    # run_single_seed looks up a BARE benchmark name (no _HF/_LF suffix).
    # For Ackley_10D that bare key is already taken by an older, unrelated
    # benchmark (see benchmarks.py's Ackley_10D_MF comment) -- redirect the
    # lookup to the correctly-aliased "Ackley_10D_MF" entry while keeping
    # this worker's own out_path/tag (and Stage 2's checkpoint naming) as
    # "Ackley_10D", matching every other method's file naming.
    sf_benchmark_name = "Ackley_10D_MF" if benchmark == "Ackley_10D" else benchmark
    result = run_single_seed(
        EXP_NAME, sf_benchmark_name, "SF-DRO-rotate-MES", seed,
        use_mes_reward=True,
        rtg_schema="floored",
        alpha_floor=0.5,
        rollout_acq_function="rotate",
        gp_num_models=5,
        rollouts_per_iter=75,
        rollout_length=4,
        bo_iterations=N_ITERS,
        initial_points=cfg["initial_hf"],
    )
    rc = result["regret_curve"]
    # SF-DRO is single-fidelity (always pays c_H per query, no LF/HF choice),
    # so its post-init cost is deterministic: cost_curve[i] = (i+1)*c_H.
    cost_curve = [hf_spec_cost * (i + 1) for i in range(len(rc))]
    out = {"regret_curve": rc, "cost_curve": cost_curve}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{tag} DONE final_regret={rc[-1]:.4f} n_iters={len(rc)} "
          f"final_cost={cost_curve[-1]:.1f}", flush=True)

else:
    from src.baselines.mf_baselines import (
        MultiFidelityBenchmark, MFGPUCBOptimizer, MFMIGreedyOptimizer, GreedyMFMESOptimizer
    )
    bench = MultiFidelityBenchmark(benchmark)
    common = dict(
        n_initial_hf=cfg["initial_hf"], n_initial_lf=cfg["initial_lf"],
        seed=seed, cost_budget=None,  # None -> mf_baselines.py's own run()
        # loops treat this as unlimited (budget = cost_budget if not None
        # else float('inf')) -- N_ITERS (passed to .run() below) is the
        # only stop condition.
    )
    if method == "MF-GP-UCB":
        opt = MFGPUCBOptimizer(bench, **common)
    elif method == "MF-MI-Greedy":
        opt = MFMIGreedyOptimizer(bench, **common)
    elif method == "Greedy-MES":
        opt = GreedyMFMESOptimizer(bench, **common)
    else:
        raise ValueError(f"unknown method {method}")

    result = opt.run(bo_iterations=N_ITERS)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    rc, cc = result["regret_curve"], result["cost_curve"]
    print(f"{tag} DONE final_regret={rc[-1]:.4f} n_iters={len(rc)} "
          f"final_cost={cc[-1]:.1f}", flush=True)
