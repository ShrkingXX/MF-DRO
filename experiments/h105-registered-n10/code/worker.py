"""H105 worker -- shim over h83's worker, so the new baseline runs are produced by
the same code that produced h83's. Overrides only the output directory."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
H83 = os.path.join(H, "..", "..", "h83-main-comparison", "code", "worker.py")
_spec = importlib.util.spec_from_file_location("h83_worker", H83)
h83 = importlib.util.module_from_spec(_spec); sys.modules["h83_worker"] = h83
_spec.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results"))
h83.RES = RES
if __name__ == "__main__":
    bench, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"{bench}__{method}__seed{seed}"
    r = h83.run(bench, method, seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {bench} {method} seed{seed} regret={r['final_regret']:.4f} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
