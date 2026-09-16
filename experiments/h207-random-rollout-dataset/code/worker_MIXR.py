"""h207 arm MIXR -- control's 20 MES + 20 random per member -- Medium-Expert. usage: worker_MIXR.py <bench> <seed>

Base config = h206N (K=8, no positional embedding, ROI-Q10). Reward label is set
through h83.ROLLOUT_REWARD -- NEVER inside _build, which h83.run() overwrites
(the silent no-op that produced h198's bit-identical arms).
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
from src.policy.mf_dro import DirectMFRegretOptimization as _DMRO
SWEEP = [-1.0, -0.5, -0.2, 0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
BTG_SWEEP = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]   # Borehole 8-step rollout cost-to-go spans [8,16] (c_L=1, c_H=2)   # spans the normalized training range under terminal_improvement (scale = max |rtg0|, so the MES half sits near +0.05 and random near -1)
_OB = h83._build_mf_dro_config

def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True                       # ROI-Q10, matching every control
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10
    c.inference_context_k = 8
    c.max_seq_length = 256
    c.absolute_timesteps = False
    c.real_prefix_training = False
    c.disable_position_embedding = True    # h206
    c.random_p_hf = 0.5                    # h207: both fidelities represented
    c.rollout_mix = [('mes', 20, None), ('random', 20, None)]
    return c

h83._build_mf_dro_config = _build
h83.ROLLOUT_REWARD = "terminal_improvement" # the h207 v3 label, via the knob
_OI = _DMRO.__init__
def _init(self, *a, **k):
    _OI(self, *a, **k)
    self._h168_probe = SWEEP; self._h177_btg_probe = BTG_SWEEP               # RNG-neutral RTG-sensitivity probe
_DMRO.__init__ = _init

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H207MIXR__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h207"] = dict(arm="MIXR", rollout_mix=repr([('mes', 20, None), ('random', 20, None)]), rollout_reward="terminal_improvement",
                      random_p_hf=0.5, inference_context_k=8, disable_position_embedding=True,
                      roi="Q10", h168_sweep=SWEEP, h177_btg_sweep=BTG_SWEEP)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
