import math
import torch
import numpy as np
from dro_runner import run_mf_single_seed

SMOKE_OVERRIDES = dict(
    initial_points=3, M=10, rollouts_per_model=7, num_epochs=10,
)

def check_result(benchmark, result):
    print(f"\n=== SMOKE TEST CHECKS: {benchmark} ===")
    ok = True

    # 1. No NaN/Inf in any logged quantity
    flat_vals = []
    for k in ["hf_regret_curve", "cost_curve", "rtg_target", "btg_target",
              "fid_mean_per_iter", "fid_std_per_iter", "L_loc_per_iter",
              "L_fid_per_iter", "neg_rtg_frac_per_iter"]:
        flat_vals.extend(result[k])
    finite = all(math.isfinite(v) for v in flat_vals)
    print(f"1. No NaN/Inf: {finite}")
    ok = ok and finite

    # 2. fid_mean in [0.05, 0.95] all iters
    fid_ok = all(0.05 <= v <= 0.95 for v in result["fid_mean_per_iter"])
    print(f"2. fid_mean in [0.05,0.95] all iters: {fid_ok}  values={result['fid_mean_per_iter']}")
    ok = ok and fid_ok

    # 3. L_loc > 0 and L_fid > 0 all iters
    loc_fid_ok = all(v > 0 for v in result["L_loc_per_iter"]) and all(v > 0 for v in result["L_fid_per_iter"])
    print(f"3. L_loc>0 and L_fid>0 all iters: {loc_fid_ok}")
    ok = ok and loc_fid_ok

    # 4. cumulative_cost strictly increases
    cc = result["cost_curve"]
    strictly_inc = all(cc[i] < cc[i+1] for i in range(len(cc)-1))
    print(f"4. cumulative_cost strictly increasing: {strictly_inc}  curve={cc}")
    ok = ok and strictly_inc

    # 5. At least 1 LF and 1 HF query
    trace = result["fidelity_trace"]
    has_both = (0 in trace) and (1 in trace)
    print(f"5. Both fidelities present: {has_both}  trace={trace}")
    ok = ok and has_both

    print(f"lf_fraction={result['lf_fraction']:.3f}")
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok

overall_pass = True
for benchmark in ["Currin_2D", "Hartmann_6D"]:
    print(f"\n########## Running MF-DRO smoke test on {benchmark} ##########")
    try:
        result = run_mf_single_seed(
            "mfdro_smoketest", benchmark, "MF-DRO", seed=42,
            bo_iterations=5, **SMOKE_OVERRIDES
        )
        p = check_result(benchmark, result)
        overall_pass = overall_pass and p
    except RuntimeError as e:
        print(f"CRASHED (RuntimeError): {e}")
        overall_pass = False
    except Exception as e:
        print(f"CRASHED (unexpected {type(e).__name__}): {e}")
        overall_pass = False

print(f"\n\n===== SMOKE TEST FINAL: {'ALL PASS' if overall_pass else 'FAILURE -- DO NOT RUN STAGE 1'} =====")
