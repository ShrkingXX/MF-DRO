"""H93 worker -- a SHIM over h83's worker, not a copy of it.

h83's worker already runs every method and benchmark this experiment needs. The
question here is whether Currin's and Ackley's h83 deficits reproduce at fresh
seeds, so the run logic must be the SAME logic that produced the h83 numbers --
not a re-implementation that could quietly diverge. Copying the file would have
forked it; this loads it and overrides exactly one module-level constant, the
output directory, so results land in h93 instead of contaminating h83's.

Nothing else is changed. If h83's worker is edited, this follows it.
"""
import os, sys, importlib.util   # NOTE: no numpy/torch here -- h83's worker sets
                                 # the thread-cap env vars at its own top, before
                                 # it imports them, and that must stay first.
H = os.path.dirname(os.path.abspath(__file__))
H83 = os.path.join(H, "..", "..", "h83-main-comparison", "code", "worker.py")
_spec = importlib.util.spec_from_file_location("h83_worker", H83)
h83 = importlib.util.module_from_spec(_spec)
sys.modules["h83_worker"] = h83
_spec.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results"))
h83.RES = RES          # the one override

if __name__ == "__main__":
    bench, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{bench}__{method}__seed{seed}"
    r = h83.run(bench, method, seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    c = r["hf_regret_curve"]
    print(f"[done] {bench} {method} seed{seed} regret={r['final_regret']:.4f} "
          f"iters={len(c)} queries={len(r['queries'])} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
