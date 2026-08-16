"""
Report + plots for DRO-MES-EntropyJoint on Hartmann_6D, from the completed
entropy_joint_validation run (5 seeds, 50 iterations, rollout_length=8).

NOTE on "entropy"/"mean RTG[0]": only rtg_target (== batch_max_rtg for this
schema's dynamic-style dispatch -- the MAX of RTG[0] across the 75 rollouts
each iteration) was logged per iteration; the raw per-rollout RTG[0]
distribution was never saved, so a literal mean-across-rollouts isn't
reconstructable without re-running. batch_max_rtg is used as the reported/
plotted quantity below, labeled accordingly rather than silently called
"mean".

Usage:
    python analyze_entropy_joint_h6d.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXP_NAME = "entropy_joint_validation"
VARIANT = "DRO-MES-EntropyJoint"
BENCHMARK = "Hartmann_6D"
SEEDS = [42, 43, 44, 45, 46]

OUT_DIR = "results/entropy_joint_rtg_h6d/plots"


def load_all():
    results = {}
    for seed in SEEDS:
        path = f"results/{EXP_NAME}/checkpoints/{BENCHMARK}__{VARIANT}__seed{seed}.json"
        results[seed] = json.load(open(path))
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_all()

    final_regrets = {s: results[s]["regret_curve"][-1] for s in SEEDS}
    mean_final = np.mean(list(final_regrets.values()))
    se_final = np.std(list(final_regrets.values())) / np.sqrt(len(SEEDS))
    best_seed = min(final_regrets, key=final_regrets.get)
    worst_seed = max(final_regrets, key=final_regrets.get)

    all_rtg_targets = [v for s in SEEDS for v in results[s]["rtg_target"]]
    neg_rtg_frac = np.mean([v < 0 for v in all_rtg_targets])

    batch_max_iter1 = [results[s]["batch_max_rtg"][0] for s in SEEDS]
    batch_max_iter50 = [results[s]["batch_max_rtg"][-1] for s in SEEDS]
    rtg_target_iter1 = [results[s]["rtg_target"][0] for s in SEEDS]
    rtg_target_iter50 = [results[s]["rtg_target"][-1] for s in SEEDS]

    print(f"=== RESULTS: {VARIANT}, {BENCHMARK}, N={len(SEEDS)} seeds ===")
    print(f"Final regret (iter 50): mean={mean_final:.4f} +/- SE={se_final:.4f}")
    print(f"Best seed:  seed{best_seed}  regret={final_regrets[best_seed]:.4f}")
    print(f"Worst seed: seed{worst_seed}  regret={final_regrets[worst_seed]:.4f}")
    print(f"Mean neg_rtg_frac (rtg_target < 0) across all {len(SEEDS)*50} (iteration, seed) pairs: {neg_rtg_frac:.4f}")
    print()
    print("NOTE: 'entropy'/RTG[0] below is batch_max_rtg -- the MAX of RTG[0] across")
    print("75 rollouts each iteration, not a mean (raw per-rollout RTG[0] was not logged).")
    print(f"Mean batch_max_rtg (RTG[0], max-of-rollouts) at iter 1:  {np.mean(batch_max_iter1):.4f} "
          f"(+/- {np.std(batch_max_iter1)/np.sqrt(len(SEEDS)):.4f})")
    print(f"Mean batch_max_rtg (RTG[0], max-of-rollouts) at iter 50: {np.mean(batch_max_iter50):.4f} "
          f"(+/- {np.std(batch_max_iter50)/np.sqrt(len(SEEDS)):.4f})")
    print(f"  EXPECT decreasing -- observed direction: "
          f"{'DECREASED' if np.mean(batch_max_iter50) < np.mean(batch_max_iter1) else 'DID NOT decrease'}")
    print()
    print(f"Mean rtg_target at iter 1:  {np.mean(rtg_target_iter1):.4f} "
          f"(+/- {np.std(rtg_target_iter1)/np.sqrt(len(SEEDS)):.4f})")
    print(f"Mean rtg_target at iter 50: {np.mean(rtg_target_iter50):.4f} "
          f"(+/- {np.std(rtg_target_iter50)/np.sqrt(len(SEEDS)):.4f})")
    print(f"  (rtg_target == batch_max_rtg exactly for this schema's dynamic-style dispatch, "
          f"so these numbers are identical to the pair above.)")

    # --- Plot 1: log10(simple regret) vs BO iteration ---
    regret_matrix = np.array([results[s]["regret_curve"] for s in SEEDS]) # [5, 50]
    log_regret = np.log10(regret_matrix)
    mean_log_regret = log_regret.mean(axis=0)
    se_log_regret = log_regret.std(axis=0) / np.sqrt(len(SEEDS))
    iters = np.arange(1, regret_matrix.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, mean_log_regret, color="tab:blue", linewidth=2)
    ax.fill_between(iters, mean_log_regret - se_log_regret, mean_log_regret + se_log_regret,
                     color="tab:blue", alpha=0.2)
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("log10(simple regret)")
    ax.set_title("Hartmann_6D: DRO-MES-EntropyJoint (rollout_len=8, dynamic RTG)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "regret_curve.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"\nSaved {p1}")

    # --- Plot 2: batch_max_rtg (proxy for "entropy H(y*|D_0)") vs BO iteration ---
    bm_matrix = np.array([results[s]["batch_max_rtg"] for s in SEEDS]) # [5, 50]
    mean_bm = bm_matrix.mean(axis=0)
    se_bm = bm_matrix.std(axis=0) / np.sqrt(len(SEEDS))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, mean_bm, color="tab:red", linewidth=2)
    ax.fill_between(iters, mean_bm - se_bm, mean_bm + se_bm, color="tab:red", alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("batch_max_rtg = max RTG[0] over 75 rollouts\n(H(y*|D_0), max-of-rollouts, NOT a mean)")
    ax.set_title("Hartmann_6D: Mean H(y* | D_0) over BO iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "entropy_evolution.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved {p2}")


if __name__ == '__main__':
    main()
