"""h181 -- h179 minus standardisation. The matched unstandardised Borehole probe
control, so h179 responsiveness is attributable to standardisation not benchmark."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_ORIG_BUILD = h83._build_mf_dro_config


def _build(*a, **k):
    cfg = _ORIG_BUILD(*a, **k)
    # h181 CONTROL: standardisation deliberately NOT set -- the one-line diff.
    return cfg


# Also switch on h178's embedding-response probe, so the arm MEASURES whether
# standardisation actually made the channel responsive at inference rather than
# assuming it. The smoke could not answer this: it evaluated one benchmark's BTG
# range against another benchmark's running statistics.
from src.policy.mf_dro import DirectMFRegretOptimization as _DMRO
_OI = _DMRO.__init__


def _init(self, *a, **k):
    _OI(self, *a, **k)
    self._h168_probe = [0.0, 0.5, 1.0]
    self._h177_btg_probe = [20.0, 26.0, 28.0, 30.0, 36.0]


_DMRO.__init__ = _init
h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__PROBECTL__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
