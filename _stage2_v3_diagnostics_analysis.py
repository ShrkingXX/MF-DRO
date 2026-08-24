"""
Analyze the 5 new diagnostic metrics (action_reward_corr_per_iter,
rtg_frac_between_traj_var_per_iter, rtg_gpbelief_corr_per_iter,
grad_coherency_per_iter, query_dist_to_xstar_per_iter) across the 20
completed MF-DRO Stage 2 v3 runs, split by frozen vs not-frozen, to test:
  - trajectory-level-luck hypothesis (high rtg_frac_between_traj_var,
    low action_reward_corr / rtg_gpbelief_corr)
  - state-only-minimum-collapse hypothesis (grad_coherency RISES over the
    run, especially for frozen runs)
  - whether real queries approach x* on Hartmann_6D (only benchmark with a
    registered known_optimal_x)
"""
import json
import glob
import numpy as np
from collections import defaultdict

METRICS = ["action_reward_corr_per_iter", "rtg_frac_between_traj_var_per_iter",
           "rtg_gpbelief_corr_per_iter", "grad_coherency_per_iter"]

rows = []
for f in sorted(glob.glob("results/mfdro_stage2_v3/checkpoints/MF-DRO__*.json")):
    d = json.load(open(f))
    rc = d["hf_regret_curve"]
    n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
    frozen = n_improved == 0
    fname = f.split("/")[-1]
    bm = fname.split("__")[1]
    seed = int(fname.split("__")[2].replace("seed", "").replace(".json", ""))

    row = dict(bm=bm, seed=seed, frozen=frozen, n_improved=n_improved,
               final_regret=rc[-1])
    for m in METRICS:
        vals = np.array([v for v in d[m] if v is not None and not np.isnan(v)])
        row[f"{m}_mean"] = vals.mean() if len(vals) else float("nan")
        if len(vals) >= 20:
            row[f"{m}_early"] = vals[:20].mean()
            row[f"{m}_late"] = vals[-20:].mean()
        else:
            row[f"{m}_early"] = row[f"{m}_late"] = float("nan")

    xdist = d.get("query_dist_to_xstar_per_iter") or []
    if xdist:
        xdist = np.array(xdist)
        row["xstar_dist_early"] = xdist[:20].mean()
        row["xstar_dist_late"] = xdist[-20:].mean()
        row["xstar_dist_min"] = xdist.min()
    rows.append(row)

print(f"Total MF-DRO runs analyzed: {len(rows)}\n")

# === Frozen vs not-frozen comparison ===
frozen_rows = [r for r in rows if r["frozen"]]
active_rows = [r for r in rows if not r["frozen"]]
print(f"Frozen: {len(frozen_rows)}/{len(rows)}   Active (improved >=1x): {len(active_rows)}/{len(rows)}\n")

print(f"{'metric':<42}{'frozen mean':<16}{'active mean':<16}{'diff'}")
for m in METRICS:
    key = f"{m}_mean"
    fz = np.mean([r[key] for r in frozen_rows if not np.isnan(r[key])])
    ac = np.mean([r[key] for r in active_rows if not np.isnan(r[key])])
    print(f"{m:<42}{fz:<16.4f}{ac:<16.4f}{fz-ac:+.4f}")

print()
print("=== grad_coherency: early (first 20 iters) vs late (last 20 iters), by frozen status ===")
for label, group in [("frozen", frozen_rows), ("active", active_rows)]:
    early = np.mean([r["grad_coherency_per_iter_early"] for r in group if not np.isnan(r["grad_coherency_per_iter_early"])])
    late = np.mean([r["grad_coherency_per_iter_late"] for r in group if not np.isnan(r["grad_coherency_per_iter_late"])])
    print(f"{label:<10} early={early:.4f}  late={late:.4f}  delta={late-early:+.4f}")

print()
print("=== Per-benchmark breakdown (mean across seeds) ===")
by_bm = defaultdict(list)
for r in rows:
    by_bm[r["bm"]].append(r)
print(f"{'bench':<14}{'frozen':<9}{'act_rew_corr':<14}{'frac_btw_var':<14}{'rtg_gp_corr':<14}{'grad_coh(early->late)'}")
for bm, rs in sorted(by_bm.items()):
    frozen_n = sum(1 for r in rs if r["frozen"])
    arc = np.mean([r["action_reward_corr_per_iter_mean"] for r in rs])
    fbv = np.mean([r["rtg_frac_between_traj_var_per_iter_mean"] for r in rs])
    rgc = np.mean([r["rtg_gpbelief_corr_per_iter_mean"] for r in rs])
    gc_e = np.mean([r["grad_coherency_per_iter_early"] for r in rs])
    gc_l = np.mean([r["grad_coherency_per_iter_late"] for r in rs])
    print(f"{bm:<14}{frozen_n}/{len(rs):<7}{arc:<14.4f}{fbv:<14.4f}{rgc:<14.4f}{gc_e:.3f} -> {gc_l:.3f}")

print()
print("=== Hartmann_6D: real-query distance to x* (only benchmark with known_optimal_x) ===")
hartmann_rows = [r for r in rows if r["bm"] == "Hartmann_6D"]
for r in sorted(hartmann_rows, key=lambda r: r["seed"]):
    print(f"  seed{r['seed']}: frozen={r['frozen']}  dist_early={r['xstar_dist_early']:.4f}  "
          f"dist_late={r['xstar_dist_late']:.4f}  dist_min={r['xstar_dist_min']:.4f}  "
          f"final_regret={r['final_regret']:.4f}")
