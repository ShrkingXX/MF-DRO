"""h197 -- the human's full specification. usage: worker.py <bench> <seed>

  (1) default MES teacher + MES fidelity selection      [h83 default]
  (2) MES reward labelling, rtg = log(b_tau) - log(b_T) [h83 sets mes_entropy]
  (3) L1 location loss                                  <- set here
  (4) sliding window, K=8 => 7 most recent + current    <- set here
  (5) real queries labelled with information gain computed the same way as
      rollouts, history anchored at the dynamic target  [core, h197]
  (6) timestep = relative position within the window    [arange(T)]
  (7) dynamic RTG prompt at the readout position        [unchanged]
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_OB = h83._build_mf_dro_config


def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10      # ROI-Q10, matching every control
    c.inference_context_k = 8       # spec (4): 7 most recent + current
    c.loc_loss = 'l1'               # spec (3)
    return c


h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H197-SPEC__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h197"] = dict(inference_context_k=8, loc_loss="l1", roi="Q10",
                      real_action_feeding=True, real_rtg_labelling=True)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
