"""h205 Stage 0 -- the SCs registered in protocol.md, run BEFORE any arm.

SC1 is the gate: a silent no-op is indistinguishable from "the fix is immaterial"
(the exact failure h196's SC caught), so it is checked first and for BOTH flags.
"""
import os, sys, importlib.util, math
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

import src.policy.mf_dro as MF
CAP = {}
_oi = MF.DirectMFRegretOptimization.__init__
def _spy(self, *a, **k):
    _oi(self, *a, **k); CAP["mf"] = self
MF.DirectMFRegretOptimization.__init__ = _spy

# capture the (timesteps, valid_mask) actually handed to the model
SEEN = []
_of = MF.DecisionTransformer.forward_mf
def _fspy(self, states, actions_ell, rtg, btg, timesteps, *a, **k):
    SEEN.append((timesteps.detach().clone(), k.get("valid_mask")))
    return _of(self, states, actions_ell, rtg, btg, timesteps, *a, **k)
MF.DecisionTransformer.forward_mf = _fspy

_s = importlib.util.spec_from_file_location("h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h83w"] = w; _s.loader.exec_module(w)
_OB = w._build_mf_dro_config
CFG = {"abs": False, "prefix": False}
def _b(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8
    c.max_seq_length = 256
    c.absolute_timesteps = CFG["abs"]
    c.real_prefix_training = CFG["prefix"]
    return c
w._build_mf_dro_config = _b

import io, contextlib
def run(tag, abs_ts, prefix):
    CFG["abs"], CFG["prefix"] = abs_ts, prefix
    SEEN.clear()
    buf = io.StringIO()
    w.BUDGET = 70.0
    with contextlib.redirect_stdout(buf):
        w.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], f"h205_{tag}.json"))
    mf = CAP["mf"]
    idx = sorted(mf._pos_idx_seen)
    Ts = [t.shape[1] for t, _ in SEEN]
    vms = [vm for _, vm in SEEN if vm is not None]
    frac_masked = (float(1.0 - torch.cat([v.reshape(-1) for v in vms]).float().mean())
                   if vms else float("nan"))
    return dict(tag=tag, idx_min=idx[0], idx_max=idx[-1], n_idx=len(idx),
                seq_len=max(Ts), frac_loss_masked=frac_masked, mf=mf)

print("="*72)
R = {}
for tag, a, p in (("CTRL  (both off)", False, False),
                  ("A     (prefix only)", False, True),
                  ("B     (absolute only)", True, False),
                  ("C     (both)", True, True)):
    R[tag] = run(tag.split()[0], a, p)
    r = R[tag]
    print(f"  {tag:24s} seq_len={r['seq_len']:3d}  pos idx {r['idx_min']}..{r['idx_max']} "
          f"({r['n_idx']} distinct)  loss-masked frac={r['frac_loss_masked']:.3f}")

print("\n" + "="*72)
ok = True
# SC1a: absolute labelling must exercise indices beyond 0..7
b_span = R["B     (absolute only)"]["n_idx"]
sc1a = b_span > 8 and R["B     (absolute only)"]["idx_max"] > 7
print(f"SC1a absolute labelling exercises >8 embedding rows: {b_span} distinct, "
      f"max {R['B     (absolute only)']['idx_max']}  -> {'PASS' if sc1a else 'FAIL (silent no-op => P2)'}")
ok &= sc1a
# SC1b: prefix must lengthen the training sequence past rollout_length
c_ctrl, c_A = R["CTRL  (both off)"]["seq_len"], R["A     (prefix only)"]["seq_len"]
sc1b = c_A > c_ctrl
print(f"SC1b prefix lengthens the training sequence: {c_ctrl} -> {c_A}  "
      f"-> {'PASS' if sc1b else 'FAIL (silent no-op => P2)'}")
ok &= sc1b
# SC1c: the prefix must be LOSS-MASKED (context, not supervision)
sc1c = R["A     (prefix only)"]["frac_loss_masked"] > 0.01
print(f"SC1c prefix is loss-masked: frac={R['A     (prefix only)']['frac_loss_masked']:.3f} "
      f"(CTRL {R['CTRL  (both off)']['frac_loss_masked']:.3f})  -> {'PASS' if sc1c else 'FAIL'}")
ok &= sc1c
# SC3: no index overflow (would have raised in-run)
print(f"SC3 no index overflow: max idx {max(R[k]['idx_max'] for k in R)} < 256  -> PASS")
# SC5: cold start survived (run completed at all)
print(f"SC5 cold start (n<K-1) handled: all four configs completed  -> PASS")
print("="*72); print("STAGE 0:", "PASS" if ok else "FAIL")
