"""h184 -- force Borehole to a Hartmann-like fidelity mix and test whether the
benchmark asymmetry moves with it.

usage: worker.py <bench> <seed> <ctrl|head>

Both arms get the IDENTICAL intervention (max_hf_fraction=0.25), so the forcing
cancels in the readout, which is the GAP between them.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
ARM = sys.argv[3] if len(sys.argv) > 3 else "ctrl"
MAX_HF = 0.25                      # Hartmann's control runs at ~0.20 HF
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    cfg = _ORIG_BUILD(*a, **k)
    cfg.max_hf_fraction = MAX_HF          # the intervention, on BOTH arms
    if ARM == "head":
        cfg.rollout_policy = "head_mes"   # the arm under comparison
    return cfg


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__LFF-{ARM.upper()}__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
