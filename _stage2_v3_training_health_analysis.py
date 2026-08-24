"""
Analyze fid_mean_per_iter, fid_std_per_iter, L_loc_per_iter, L_fid_per_iter,
neg_rtg_frac_per_iter across the 20 completed MF-DRO Stage 2 v3 runs, split
by frozen vs not-frozen, plus early-vs-late trend within a run (fidelity
head collapse, in particular, would show up as fid_std shrinking toward 0
and fid_mean drifting toward 0 or 1 over the course of a run).
"""
import json
import glob
import numpy as np
from collections import defaultdict

METRICS = ["fid_mean_per_iter", "fid_std_per_iter", "L_loc_per_iter",
           "L_fid_per_iter", "neg_rtg_frac_per_iter"]

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
               final_regret=rc[-1], lf_fraction=d["lf_fraction"])
    for m in METRICS:
        vals = np.array([v for v in d[m] if v is not None and not np.isnan(v)])
        row[f"{m}_mean"] = vals.mean() if len(vals) else float("nan")
        if len(vals) >= 20:
            row[f"{m}_early"] = vals[:20].mean()
            row[f"{m}_late"] = vals[-20:].mean()
        else:
            row[f"{m}_early"] = row[f"{m}_late"] = float("nan")
    rows.append(row)

frozen_rows = [r for r in rows if r["frozen"]]
active_rows = [r for r in rows if not r["frozen"]]
print(f"Frozen: {len(frozen_rows)}/{len(rows)}   Active: {len(active_rows)}/{len(rows)}\n")

print(f"{'metric':<24}{'frozen mean':<15}{'active mean':<15}{'diff'}")
for m in METRICS:
    key = f"{m}_mean"
    fz = np.mean([r[key] for r in frozen_rows if not np.isnan(r[key])])
    ac = np.mean([r[key] for r in active_rows if not np.isnan(r[key])])
    print(f"{m:<24}{fz:<15.4f}{ac:<15.4f}{fz-ac:+.4f}")

print()
print("=== fid_mean / fid_std: early vs late, by frozen status (collapse check) ===")
for m in ["fid_mean_per_iter", "fid_std_per_iter"]:
    print(f"-- {m} --")
    for label, group in [("frozen", frozen_rows), ("active", active_rows)]:
        early = np.mean([r[f"{m}_early"] for r in group if not np.isnan(r[f"{m}_early"])])
        late = np.mean([r[f"{m}_late"] for r in group if not np.isnan(r[f"{m}_late"])])
        print(f"  {label:<8} early={early:.4f}  late={late:.4f}  delta={late-early:+.4f}")

print()
print("=== L_loc / L_fid: early vs late, by frozen status ===")
for m in ["L_loc_per_iter", "L_fid_per_iter"]:
    print(f"-- {m} --")
    for label, group in [("frozen", frozen_rows), ("active", active_rows)]:
        early = np.mean([r[f"{m}_early"] for r in group if not np.isnan(r[f"{m}_early"])])
        late = np.mean([r[f"{m}_late"] for r in group if not np.isnan(r[f"{m}_late"])])
        ratio = early / late if late > 1e-12 else float('inf')
        print(f"  {label:<8} early={early:.5f}  late={late:.5f}  collapse_ratio(early/late)={ratio:.2f}x")

print()
print("=== Per-run detail: fid_mean/fid_std early->late + lf_fraction ===")
for r in sorted(rows, key=lambda r: (r["bm"], r["seed"])):
    print(f"  {r['bm']:<12} seed{r['seed']} frozen={r['frozen']!s:<6} lf_frac={r['lf_fraction']:.3f}  "
          f"fid_mean: {r['fid_mean_per_iter_early']:.3f}->{r['fid_mean_per_iter_late']:.3f}  "
          f"fid_std: {r['fid_std_per_iter_early']:.3f}->{r['fid_std_per_iter_late']:.3f}")

print()
print("=== Per-benchmark breakdown ===")
by_bm = defaultdict(list)
for r in rows:
    by_bm[r["bm"]].append(r)
print(f"{'bench':<14}{'frozen':<9}{'neg_rtg_frac':<14}{'L_loc(e->l)':<20}{'fid_std(e->l)'}")
for bm, rs in sorted(by_bm.items()):
    frozen_n = sum(1 for r in rs if r["frozen"])
    nrf = np.mean([r["neg_rtg_frac_per_iter_mean"] for r in rs])
    lloc_e = np.mean([r["L_loc_per_iter_early"] for r in rs])
    lloc_l = np.mean([r["L_loc_per_iter_late"] for r in rs])
    fs_e = np.mean([r["fid_std_per_iter_early"] for r in rs])
    fs_l = np.mean([r["fid_std_per_iter_late"] for r in rs])
    print(f"{bm:<14}{frozen_n}/{len(rs):<7}{nrf:<14.4f}{lloc_e:.4f}->{lloc_l:<10.4f}{fs_e:.3f}->{fs_l:.3f}")
