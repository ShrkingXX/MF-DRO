# DIAGNOSTIC ONLY -- NOT a quality comparison. Standing instruction (2026-09-03):
# policy quality is compared ONLY by final simple regret (frozen rel% @ cost 200).
"""h198 INTERMEDIATE regret curve, read from in-flight results/ckpt/.

Capped at the LOWEST cost any arm has reached, so no arm is read past what all
arms have actually shown. Uses h83's own sr_curve/grid, imported.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark

OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
G = np.linspace(0, 200, 201)
SEEDS = (42, 43, 44, 45, 46)

RUNS = {
    "h198a lookahead/mes_entropy":
        f"{REPO}/experiments/h198-regret-optimal-teacher/results/ckpt/Borehole_8D__H198A-LOOK-MES__seed{{s}}.json",
    "h198b lookahead/improvement":
        f"{REPO}/experiments/h198-regret-optimal-teacher/results/ckpt/Borehole_8D__H198B-LOOK-IMP__seed{{s}}.json",
    "CTRL-K1 (MES, no window)":
        f"{REPO}/experiments/h194-expert-plan-window/results/Borehole_8D__CTRL-K1__seed{{s}}.json",
    "h199 oracle-lookahead (ceiling)":
        f"{REPO}/experiments/h199-oracle-lookahead-ceiling/results/Borehole_8D__H199-ORACLE-LOOK__seed{{s}}.json",
}

curves, reach = {}, {}
for label, pat in RUNS.items():
    rows, mx = [], []
    for s in SEEDS:
        p = pat.format(s=s)
        if not os.path.exists(p): continue
        c, sr = sr_curve(json.load(open(p)), OPT)
        rows.append(grid(c, sr, G))
        mx.append(c.max() if len(c) else 0.0)
    if rows:
        curves[label] = np.vstack(rows); reach[label] = min(mx)
        print(f"  {label}: {len(rows)}/5 seeds, min reach {min(mx):.0f}")

cap = min(reach.values())
print(f"\n  lowest cost reached by ANY seed of ANY arm: {cap:.0f} -> capped there\n")
pts = [c for c in (25, 50, 75, 100, 125) if c <= cap]
if not pts:
    print("  cap too low for any grid point yet."); sys.exit(0)
hdr = "  " + "arm".ljust(32) + "".join(f"{f'c={c}':>9}" for c in pts)
print(hdr); print("  " + "-"*(len(hdr)-2))
for label, A in curves.items():
    rel = 100.0 * A / abs(OPT)
    m = [np.nanmean(rel[:, np.searchsorted(G, c)]) for c in pts]
    print("  " + label.ljust(32) + "".join(f"{v:9.2f}" for v in m))
