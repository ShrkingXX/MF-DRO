"""
MF-DRO vs Greedy-MES basin-width sensitivity comparison: does Greedy-MES
(DT-free, no training, same KO-GP/MES machinery) show similarly dramatic
improvement as the basin widens, or is MF-DRO disproportionately more
sensitive? Ratio of w10 (narrowest) to w01 (widest) quantifies relative
sensitivity for each method -- a bigger ratio means more sensitive to
basin width.
"""
import json
import os
import numpy as np

CKPT_DIR = "results/mfdro_basin_width_sweep/checkpoints"
VARIANTS = ["Hartmann6D_w10", "Hartmann6D_w03", "Hartmann6D_w01"]
SEEDS = [42, 43, 44]


def load(prefix, variant, seed):
    path = os.path.join(CKPT_DIR, f"{prefix}{variant}__seed{seed}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def summarize(prefix, label):
    print(f"\n=== {label} ===")
    print(f"{'variant':<16}{'alpha':<8}{'mean_final_regret':<20}{'mean_n_improved':<18}"
          f"{'mean_n_hf_q':<14}{'mean_hf_gap':<14}")
    rows = {}
    for variant in VARIANTS:
        runs = [load(prefix, variant, s) for s in SEEDS]
        runs = [r for r in runs if r is not None]
        if not runs:
            print(f"{variant:<16} MISSING")
            continue
        alpha = runs[0]["alpha_basin"]
        mean_regret = np.mean([r["final_regret"] for r in runs])
        mean_improved = np.mean([r["incumbent_improved_count"] for r in runs])
        if prefix.startswith("GreedyMES"):
            gaps = [r["mean_hf_gap"] for r in runs if r.get("mean_hf_gap") is not None]
            n_hf = np.mean([r["n_hf_queries"] for r in runs])
        else:
            # MF-DRO JSONs don't carry mean_hf_gap/n_hf_queries fields --
            # recompute the same way _basin_width_greedy_mes_worker.py does.
            gaps, n_hf_list = [], []
            for r in runs:
                true_opt = r["true_optimal_value"]
                ell_t = r["fidelity_trace"]
                y_t = r["y_t_trace"]
                hf_gaps = [true_opt - y for y, e in zip(y_t, ell_t) if e == 1]
                if hf_gaps:
                    gaps.append(np.mean(hf_gaps))
                n_hf_list.append(len(hf_gaps))
            n_hf = np.mean(n_hf_list)
        mean_gap = np.mean(gaps) if gaps else float("nan")
        rows[variant] = dict(alpha=alpha, final_regret=mean_regret, n_improved=mean_improved,
                              n_hf=n_hf, hf_gap=mean_gap)
        print(f"{variant:<16}{alpha:<8}{mean_regret:<20.4f}{mean_improved:<18.2f}"
              f"{n_hf:<14.2f}{mean_gap:<14.4f}")
    return rows


mf_rows = summarize("", "MF-DRO (basin-width sweep)")
gm_rows = summarize("GreedyMES__", "Greedy-MES (control)")

print("\n=== Sensitivity ratio: w10 (narrowest) / w01 (widest) ===")
print(f"{'metric':<20}{'MF-DRO ratio':<16}{'Greedy-MES ratio':<18}")
for metric in ["final_regret", "hf_gap"]:
    mf_ratio = mf_rows["Hartmann6D_w10"][metric] / mf_rows["Hartmann6D_w01"][metric]
    gm_ratio = gm_rows["Hartmann6D_w10"][metric] / gm_rows["Hartmann6D_w01"][metric]
    print(f"{metric:<20}{mf_ratio:<16.2f}{gm_ratio:<18.2f}")

print("\nInterpretation: a much larger MF-DRO ratio than Greedy-MES's ratio means "
      "MF-DRO is disproportionately more sensitive to basin width -- evidence the "
      "effect is doing something specific to MF-DRO's own mechanism, not just "
      "'narrower basin = harder problem for anyone.' Similar ratios mean the earlier "
      "finding is mostly a generic difficulty confound.")
