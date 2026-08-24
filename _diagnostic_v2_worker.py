"""
Diagnostic v2: re-run of scripts/test_changes.py's exact 4-variant x 3-seed
design, updated for config that changed after the original was authored --
rollout_length default 4->8, BES added then disabled (bes_delta=0.0), DKL
architecture fixed, GP warm-starting added.

Parallelized across variants (4 independent processes, one per variant) and
resumable (each (variant,seed) run's diagnostics are saved to a sidecar
JSON; a run already on disk is loaded back instead of recomputed):

    python _diagnostic_v2_worker.py BASELINE    # this variant's pre-pass + 3 seeds
    python _diagnostic_v2_worker.py INIT_ONLY
    python _diagnostic_v2_worker.py INIT_RTG
    python _diagnostic_v2_worker.py FULL
    python _diagnostic_v2_worker.py SUMMARIZE   # after all 4 finish: table + interpretation only

Per-variant runs and SUMMARIZE are separate processes/invocations so the
summary table and A/B/C/D interpretation (which need all 4 variants
together) can be produced once, after the parallel per-variant runs finish,
without re-running anything. Section 1/2/3 diagnostics, the summary table
layout, and the interpretation logic are otherwise byte-for-byte the same
as scripts/test_changes.py (the original, un-parallelized, un-resumable
design this replaces) and _diagnostic_v2_worker.py's first (killed, no
progress lost) sequential-single-process attempt: EXP_NAME
(results/mfdro_diagnostic_v2) and COMMON_CFG['rollout_length']=8 are the
only config changes from the original script; bes_delta is not set here at
all, inheriting _build_mf_dro_config's own default (0.0, disabled).
"""
import os
import sys
import json
import time
import statistics as st

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

torch.set_default_dtype(torch.float64)
# Intra-op thread cap (4 variant-processes run concurrently on a 15-core
# machine) is set via OMP_NUM_THREADS/MKL_NUM_THREADS env vars at launch,
# not torch.set_num_threads() here -- the latter warns and no-ops once any
# parallel work has already started (triggered by something in the
# botorch/gpytorch import chain above), so it must be set before the
# process's native thread pool initializes.

X_STAR = torch.tensor([0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64)
TRUE_HF_AT_XSTAR = 3.322
BENCHMARK = "Hartmann_6D"
SEEDS = [42, 43, 44]
EXP_NAME = "mfdro_diagnostic_v2"
CHECKPOINTS = [80, 240, 400]  # 10x/30x/50x c_H=8

COMMON_CFG = dict(
    M=5, rollout_length=8, rollouts_per_model=4,
    bo_iterations=500, initial_hf=18, initial_lf=30,
    num_epochs=50, cost_budget=400,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
)

VARIANTS = {
    "BASELINE":  dict(use_sequential_init=False, use_rtg_grounding=False, dkl_threshold=9999),
    "INIT_ONLY": dict(use_sequential_init=True,  use_rtg_grounding=False, dkl_threshold=9999),
    "INIT_RTG":  dict(use_sequential_init=True,  use_rtg_grounding=True,  dkl_threshold=9999),
    "FULL":      dict(use_sequential_init=True,  use_rtg_grounding=True,  dkl_threshold=30),
}
ORDER = ["BASELINE", "INIT_ONLY", "INIT_RTG", "FULL"]
LABELS = {"BASELINE": "BASELINE", "INIT_ONLY": "INIT", "INIT_RTG": "INIT+RTG", "FULL": "FULL"}

out_dir = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(out_dir, exist_ok=True)


def result_path(variant, seed):
    return os.path.join(out_dir, f"{variant}__{BENCHMARK}__seed{seed}.json")


def diag_path(variant, seed):
    return os.path.join(out_dir, f"{variant}__{BENCHMARK}__seed{seed}.diag.json")


def build_mf_dro(variant_name, seed):
    cfg = _build_mf_dro_config(
        EXP_NAME, BENCHMARK, variant_name, seed,
        **COMMON_CFG, **VARIANTS[variant_name]
    )
    hf_spec = get_benchmark(BENCHMARK + "_HF")
    lf_spec = get_benchmark(BENCHMARK + "_LF")
    f_hf = hf_spec["make_objective"]()
    f_lf = lf_spec["make_objective"]()
    bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)
    return DirectMFRegretOptimization(cfg, f_hf, f_lf, bounds)


def regret_at_cost(cost_curve, regret_curve, c_ref):
    idx = None
    for i, c in enumerate(cost_curve):
        if c <= c_ref:
            idx = i
        else:
            break
    return regret_curve[idx] if idx is not None else regret_curve[0]


def run_one_variant(variant):
    print("=" * 100)
    print(f"PRE-PASS (variant={variant}): identifying best-init seed")
    print("=" * 100)
    init_stats = {}
    best_max, best_seed = -1e9, None
    for seed in SEEDS:
        mf = build_mf_dro(variant, seed)
        mf._sample_initial_points()
        max_hf = max(mf.data_hf_y)
        mean_hf = float(np.mean(mf.data_hf_y))
        n_above = sum(1 for y in mf.data_hf_y if y > 1.5)
        init_stats[seed] = dict(max_hf=max_hf, mean_hf=mean_hf, n_above=n_above)
        if max_hf > best_max:
            best_max, best_seed = max_hf, seed
    print(f"{variant}: best-init seed = {best_seed} (max_hf={best_max:.4f})")

    run_data = {}
    for seed in SEEDS:
        rpath, dpath = result_path(variant, seed), diag_path(variant, seed)
        if os.path.exists(rpath) and os.path.exists(dpath):
            print(f"\nSKIP variant={variant} seed={seed}: already completed "
                  f"({rpath} + {dpath} exist), loading from disk.")
            with open(rpath) as f:
                result = json.load(f)
            with open(dpath) as f:
                diag = json.load(f)
            run_data[seed] = dict(result=result, **diag)
            continue

        print("\n" + "=" * 100)
        print(f"RUN: variant={variant} seed={seed}")
        print("=" * 100)

        mf = build_mf_dro(variant, seed)
        mf._sample_initial_points()

        stats = init_stats[seed]
        print(f"=== INIT seed{seed} variant={variant} ===")
        print(f"max_hf={stats['max_hf']:.4f}  mean_hf={stats['mean_hf']:.4f}  "
              f"n_above_1.5={stats['n_above']}")

        mf._update_ko_ensemble()
        with torch.no_grad():
            mu0, var0 = mf.ko_ensemble[0].hf_posterior(X_STAR.unsqueeze(0))
        mu0, sigma0 = mu0.item(), var0.clamp_min(0).sqrt().item()
        print(f"GP at x*: mu={mu0:.4f}  sigma={sigma0:.4f}  (true={TRUE_HF_AT_XSTAR})")

        is_best_init_seed = (seed == best_seed)

        cost_budget = mf.config.cost_budget
        n_incumbent_improved = 0
        last_best_hf = None
        dkl_activated_iter = None
        neg_rtg_fracs = []
        t0_run = time.time()

        for t in range(mf.config.bo_iterations):
            if mf.post_init_cost >= cost_budget:
                print(f"iter {t}: cost budget reached "
                      f"(post_init_cost={mf.post_init_cost:.1f} >= {cost_budget:.1f}), stopping.")
                break

            mf._update_ko_ensemble()
            batch = mf._generate_rollout_batch()
            rtg_target = mf.schemas.update_and_get_rtg_target(batch)
            btg_target = mf.schemas.update_and_get_btg_target(batch)
            mf._last_rtg_target = rtg_target
            if mf.btg_target_base is None:
                mf.btg_target_base = btg_target

            L_loc, L_fid, fid_mean, fid_std = mf._train_dt(batch)
            x_t, ell_t = mf._propose_next_query()
            mf._last_p_pred = mf.dt.last_p_pred

            real_hf_warmup = getattr(mf.config, 'real_hf_warmup', 2)
            if t < real_hf_warmup:
                ell_t = 1

            if ell_t == 1:
                y_t = mf.f_hf(x_t.unsqueeze(0)).reshape(-1)[0].item()
                mf.data_hf_x.append(x_t.double())
                mf.data_hf_y.append(y_t)
            else:
                y_t = mf.f_lf(x_t.unsqueeze(0)).reshape(-1)[0].item()
                mf.data_lf_x.append(x_t.double())
                mf.data_lf_y.append(y_t)
            step_cost = mf.c_H if ell_t else mf.c_L
            mf.cumulative_cost += step_cost
            mf.post_init_cost += step_cost
            mf.recent_ell_history.append(ell_t)

            best_hf = max(mf.data_hf_y)
            regret = -best_hf - mf.config.true_opt
            neg_rtg_frac_batch = float(np.mean([tr['neg_rtg_frac'] for tr in batch]))
            neg_rtg_fracs.append((t, neg_rtg_frac_batch))

            if last_best_hf is None:
                incumbent_improved = "N/A(first)"
            else:
                incumbent_improved = best_hf > last_best_hf + 1e-9
                if incumbent_improved:
                    n_incumbent_improved += 1
            last_best_hf = best_hf

            use_dkl_now = mf.ko_ensemble[0].use_dkl
            if use_dkl_now and dkl_activated_iter is None:
                dkl_activated_iter = t

            mf.iteration_log.append({
                'iter': t, 'ell_t': ell_t, 'y_t': y_t, 'x_t': x_t.tolist(),
                'cumulative_cost': mf.cumulative_cost, 'post_init_cost': mf.post_init_cost,
                'regret': regret, 'rtg_target': rtg_target, 'btg_target': btg_target,
                'fid_mean': fid_mean, 'fid_std': fid_std, 'L_loc': L_loc, 'L_fid': L_fid,
                'neg_rtg_frac': neg_rtg_frac_batch,
            })

            print(f"iter {t:3d} | cost={mf.post_init_cost:7.1f} | ell_t={ell_t} | "
                  f"p_pred={mf._last_p_pred:.4f} | regret={regret:.4f} | "
                  f"best_hf={best_hf:.4f} | rtg_target={rtg_target:.4f} | "
                  f"neg_rtg_frac_batch={neg_rtg_frac_batch:.4f} | use_dkl={use_dkl_now} | "
                  f"incumbent_improved={incumbent_improved}")

            if is_best_init_seed and t % 10 == 0:
                with torch.no_grad():
                    mu_t, var_t = mf.ko_ensemble[0].hf_posterior(X_STAR.unsqueeze(0))
                sigma_t = var_t.clamp_min(0).sqrt().item()
                print(f"  [SECTION 3] iter {t}: GP at x*: mu={mu_t.item():.4f} "
                      f"sigma={sigma_t:.4f} use_dkl={use_dkl_now}")

        elapsed = time.time() - t0_run
        result = mf._build_result()
        with open(rpath, "w") as f:
            json.dump(result, f, indent=2)

        diag = dict(
            init_mu_at_xstar=mu0,
            init_max_hf=stats['max_hf'],
            init_mean_hf=stats['mean_hf'],
            init_n_above=stats['n_above'],
            n_incumbent_improved=n_incumbent_improved,
            dkl_activated_iter=dkl_activated_iter,
            neg_rtg_fracs=neg_rtg_fracs,
            wall_time=elapsed,
        )
        with open(dpath, "w") as f:
            json.dump(diag, f, indent=2)

        run_data[seed] = dict(result=result, **diag)
        print(f"DONE variant={variant} seed={seed} final_regret={result['hf_regret_curve'][-1]:.4f} "
              f"n_iters={len(result['fidelity_trace'])} wall_time={elapsed:.1f}s")

    print(f"\nVARIANT {variant} COMPLETE ({len(run_data)}/{len(SEEDS)} seeds).")
    return run_data


def neg_rtg_frac_at_iter(fracs, target_iter):
    exact = [v for (i, v) in fracs if i == target_iter]
    if exact:
        return exact[0]
    before = [v for (i, v) in fracs if i <= target_iter]
    return before[-1] if before else float('nan')


def summarize():
    run_data = {}
    missing = []
    for variant in ORDER:
        for seed in SEEDS:
            rpath, dpath = result_path(variant, seed), diag_path(variant, seed)
            if not (os.path.exists(rpath) and os.path.exists(dpath)):
                missing.append((variant, seed))
                continue
            with open(rpath) as f:
                result = json.load(f)
            with open(dpath) as f:
                diag = json.load(f)
            run_data[(variant, seed)] = dict(result=result, **diag)
    if missing:
        print(f"ERROR: {len(missing)} (variant,seed) runs not yet completed: {missing}")
        print("Run the missing variant(s) first, then re-run SUMMARIZE.")
        sys.exit(1)

    print("\n" + "=" * 100)
    print("=== DIAGNOSTIC SUMMARY: Hartmann_6D ===")
    print("=" * 100)

    summary = {}
    for variant in ORDER:
        seeds_data = [run_data[(variant, s)] for s in SEEDS]
        mean_max_hf_init = st.mean([d['init_max_hf'] for d in seeds_data])
        n_above_15 = sum(1 for d in seeds_data if d['init_max_hf'] > 1.5)
        mean_gp_mu_init = st.mean([d['init_mu_at_xstar'] for d in seeds_data])
        mean_neg_rtg_iter1 = st.mean([neg_rtg_frac_at_iter(d['neg_rtg_fracs'], 1) for d in seeds_data])
        mean_neg_rtg_iter20 = st.mean([neg_rtg_frac_at_iter(d['neg_rtg_fracs'], 20) for d in seeds_data])
        dkl_seeds = sum(1 for d in seeds_data if d['dkl_activated_iter'] is not None)
        mean_incumbent_improved = st.mean([d['n_incumbent_improved'] for d in seeds_data])
        regrets = {}
        for c in CHECKPOINTS:
            vals = [regret_at_cost(d['result']['cost_curve'], d['result']['hf_regret_curve'], c)
                    for d in seeds_data]
            regrets[c] = st.mean(vals)
        summary[variant] = dict(
            mean_max_hf_init=mean_max_hf_init, n_above_15=n_above_15,
            mean_gp_mu_init=mean_gp_mu_init,
            mean_neg_rtg_iter1=mean_neg_rtg_iter1, mean_neg_rtg_iter20=mean_neg_rtg_iter20,
            dkl_seeds=dkl_seeds, mean_incumbent_improved=mean_incumbent_improved,
            regrets=regrets,
        )

    print(f"\n{'Metric':<28}" + "".join(f"{LABELS[v]:>10}" for v in ORDER))
    print("-" * (28 + 10 * len(ORDER)))
    print(f"{'mean max_hf_at_init':<28}" + "".join(f"{summary[v]['mean_max_hf_init']:>10.2f}" for v in ORDER))
    print(f"{'seeds with max_hf>1.5':<28}" + "".join(f"{summary[v]['n_above_15']:>7}/3  " for v in ORDER))
    print(f"{'mean GP mu at x* (init)':<28}" + "".join(f"{summary[v]['mean_gp_mu_init']:>10.2f}" for v in ORDER))
    print(f"{'mean neg_rtg_frac(iter1)':<28}" + "".join(f"{summary[v]['mean_neg_rtg_iter1']:>10.2f}" for v in ORDER))
    print(f"{'mean neg_rtg_frac(iter20)':<28}" + "".join(f"{summary[v]['mean_neg_rtg_iter20']:>10.2f}" for v in ORDER))
    print(f"{'DKL activated?':<28}" + "".join(f"{str(summary[v]['dkl_seeds'])+'/3':>10}" for v in ORDER))
    print(f"{'mean iters incumbent_impr':<28}" + "".join(f"{summary[v]['mean_incumbent_improved']:>10.2f}" for v in ORDER))
    for c in CHECKPOINTS:
        print(f"{'regret at cost=' + str(c):<28}" + "".join(f"{summary[v]['regrets'][c]:>10.4f}" for v in ORDER))

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    delta_init = summary["INIT_ONLY"]["mean_max_hf_init"] - summary["BASELINE"]["mean_max_hf_init"]
    print("\nA) SEQUENTIAL INIT IMPACT:")
    if delta_init > 0.5:
        print(f"Sequential init improves initialization coverage. (delta={delta_init:.3f})")
    else:
        print(f"Sequential init did not reliably improve init on Hartmann_6D. "
              f"Basin still too narrow. (delta={delta_init:.3f})")

    delta_rtg = summary["INIT_ONLY"]["mean_neg_rtg_iter20"] - summary["INIT_RTG"]["mean_neg_rtg_iter20"]
    print("\nB) RTG GROUNDING IMPACT:")
    if delta_rtg > 0.05:
        print(f"RTG grounding reduces neg_rtg_frac meaningfully. (delta={delta_rtg:.3f})")
    else:
        print(f"RTG grounding has minimal effect on neg_rtg_frac. "
              f"GP miscalibration still dominates RTG bias. (delta={delta_rtg:.3f})")

    final_checkpoint = CHECKPOINTS[-1]
    print("\nC) DKL ACTIVATION:")
    if summary["FULL"]["dkl_seeds"] >= 1:
        print(f"DKL activated. Compare FULL vs INIT+RTG at cost={final_checkpoint}.")
        delta_dkl = summary["INIT_RTG"]["regrets"][final_checkpoint] - summary["FULL"]["regrets"][final_checkpoint]
        if delta_dkl > 0.3:
            print(f"DKL provides measurable GP quality improvement. (delta={delta_dkl:.3f})")
        else:
            print(f"DKL did not improve regret after activation. Feature extractor may not "
                  f"have learned useful structure at this data scale. (delta={delta_dkl:.3f})")
    else:
        print(f"DKL did not activate in any seed at cost={final_checkpoint}. "
              f"HF query rate too low to reach threshold=30.")
        print("Recommendation: lower dkl_threshold to 25 or increase minimum_hf_fraction.")

    print("\nD) INCUMBENT FREEZE:")
    any_resolved = False
    for v in ORDER:
        if summary[v]["mean_incumbent_improved"] > 3:
            print(f"Incumbent freeze partially resolved for {v}. "
                  f"(mean_incumbent_improved={summary[v]['mean_incumbent_improved']:.2f})")
            any_resolved = True
    if not any_resolved:
        print("Incumbent freeze persists across all variants. Changes are insufficient without "
              "location architecture fix (Path 1 / candidate scoring).")

    print("\n" + "=" * 100)
    print("Diagnostic test complete.")
    print("=" * 100)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in list(VARIANTS) + ["SUMMARIZE"]:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(list(VARIANTS) + ['SUMMARIZE'])}>")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "SUMMARIZE":
        summarize()
    else:
        run_one_variant(mode)
