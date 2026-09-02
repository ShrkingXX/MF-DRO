"""h141: cProfile a TRUNCATED Borehole ROI-Q10 run and report cumulative time by
subsystem. Truncation changes absolute times, not relative shares, which is what
P1 and P2 concern. The budget used is recorded with the result.
"""
import cProfile, pstats, io, os, sys, json, importlib.util, time
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
H90 = os.path.join(REPO, "experiments/h90-borehole-confirm/code/worker.py")
_s = importlib.util.spec_from_file_location("h90w", H90)
h90 = importlib.util.module_from_spec(_s); sys.modules["h90w"] = h90; _s.loader.exec_module(h90)

BUDGET = float(os.environ.get("H141_BUDGET", "70"))     # truncated; full is 200
h90.BUDGET = BUDGET
RES = os.path.abspath(os.path.join(H, "..", "results"))
os.makedirs(os.path.join(RES, "ckpt"), exist_ok=True)

pr = cProfile.Profile(); t0 = time.time()
pr.enable()
try:
    h90.run("Borehole_8D", "ROI-Q10", 42, os.path.join(RES, "ckpt", "profile.json"))
finally:
    pr.disable()
wall = time.time() - t0

s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(60); text = s.getvalue()
open(os.path.join(RES, "profile_raw.txt"), "w").write(text)

# Attribute cumulative time to subsystems by module/function, counting each
# frame once at its OUTERMOST occurrence via tottime (self time) sums, which
# cannot double-count nested calls the way cumulative time can.
buckets = {
    "DT training (decisionTransformer)": ["decisionTransformer", "torch/nn", "torch/optim",
                                          "_functional", "autograd", "backward"],
    "GP / posterior (ko_ensemble, gpytorch, cholesky)": ["gp", "posterior", "cholesky",
                                                         "gpytorch", "linalg", "kernel"],
    "ROI construction / candidate pool": ["roi", "candidate", "_draw", "rand"],
    "rollout / simulate": ["simulate", "rollout", "trajectory"],
    "benchmark objective": ["benchmark", "objective", "f_hf", "f_lf"],
}
stats = pstats.Stats(pr)
tot = sum(v[2] for v in stats.stats.values())          # v[2] = tottime
agg = {k: 0.0 for k in buckets}; agg["other"] = 0.0
for (fn, ln, name), v in stats.stats.items():
    key = f"{fn}:{name}".lower()
    for b, pats in buckets.items():
        if any(p.lower() in key for p in pats):
            agg[b] += v[2]; break
    else:
        agg["other"] += v[2]
out = dict(budget=BUDGET, wall_s=wall, total_tottime=tot,
           shares={k: (v, 100.0 * v / tot if tot else 0.0) for k, v in agg.items()})
json.dump(out, open(os.path.join(RES, "profile_summary.json"), "w"), indent=2)

print(f"\n=== h141 profile: Borehole ROI-Q10 seed42, budget {BUDGET} (truncated) ===")
print(f"wall {wall/60:.1f} min | total self-time accounted {tot:.1f}s\n")
print(f"{'subsystem':52s}{'self-time s':>13s}{'share':>9s}")
for k, (v, pct) in sorted(out["shares"].items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:50s}{v:13.1f}{pct:8.1f}%")
print("\nTop 12 by cumulative time:")
for line in text.splitlines():
    if line.strip().startswith(("ncalls", "1 ", "2 ")) or "/" in line[:12]:
        pass
print("\n".join(text.splitlines()[4:20]))
