"""h206 Stage 0 -- the SCs registered in protocol.md, run BEFORE any arm.

SC1 is the gate. A silent no-op is indistinguishable from "position carries
nothing", which is precisely P1 -- so the ablation must be proven real before any
result is allowed to mean anything. SC3 guards the opposite failure: an ablation
that also severs the context would make arm N a disguised K=1 arm.
"""
import os, sys, importlib.util, io, contextlib
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
    _oi(self, *a, **k)
    CAP["mf"] = self
    # snapshot the position table AT CONSTRUCTION: SC1 compares against this.
    CAP["W0"] = self.dt.position_embedding.weight.detach().clone()
MF.DirectMFRegretOptimization.__init__ = _spy

SEEN = []
_of = MF.DecisionTransformer.forward_mf
def _fspy(self, states, actions_ell, rtg, btg, timesteps, *a, **k):
    SEEN.append(timesteps.detach().clone())
    return _of(self, states, actions_ell, rtg, btg, timesteps, *a, **k)
MF.DecisionTransformer.forward_mf = _fspy

_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h83w"] = w; _s.loader.exec_module(w)
_OB = w._build_mf_dro_config
CFG = {"nopos": False}
def _b(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8
    c.max_seq_length = 256
    c.disable_position_embedding = CFG["nopos"]
    return c
w._build_mf_dro_config = _b
SCR = os.environ.get("SCRATCH", "/tmp")

def run(tag, nopos):
    CFG["nopos"] = nopos
    SEEN.clear()
    buf = io.StringIO(); w.BUDGET = 70.0
    with contextlib.redirect_stdout(buf):
        w.run("Borehole_8D", "MF-DRO", 42, os.path.join(SCR, f"h206_{tag}.json"))
    mf = CAP["mf"]; dt = mf.dt
    W0, W1 = CAP["W0"], dt.position_embedding.weight.detach()
    return dict(tag=tag, mf=mf, dt=dt,
                seq_len=max(int(t.shape[1]) for t in SEEN),
                n_idx=len(mf._pos_idx_seen),
                grad_is_none=(dt.position_embedding.weight.grad is None),
                dW=float((W1.double() - W0.double()).abs().max()))

print("=" * 74)
P = run("nopos", True)
C = run("ctrl",  False)
for r in (C, P):
    print(f"  {r['tag']:6s} seq_len={r['seq_len']:3d}  labelled idx={r['n_idx']:3d}  "
          f"grad is None={str(r['grad_is_none']):5s}  max|dW|={r['dW']:.3e}")

# ---- SC1: the ablation is REAL -----------------------------------------------
sc1 = (P["dW"] == 0.0) and (C["dW"] > 0.0)
print(f"\nSC1 position table frozen under ablation, trained under CTRL: "
      f"nopos {P['dW']:.3e} == 0 and ctrl {C['dW']:.3e} > 0  -> "
      f"{'PASS' if sc1 else 'FAIL'}")

# ---- SC3: context still flows under the ablation ------------------------------
# Same current state and conditioning, two DIFFERENT histories. If the query is
# unchanged, deleting position also severed the context and arm N would be a
# disguised K=1 arm rather than a positional ablation.
def probe(r):
    mf, dt = r["mf"], r["dt"]
    hist = [h for h in mf._real_hist][-7:]
    if len(hist) < 2:
        return None
    def mk(hs):
        return [{'state': h['state'].float().clone(), 'rtg': float(h['rtg']),
                 'btg': float(h['btg']), 'ax': h.get('ax'), 'ae': int(h.get('ae', 0))}
                for h in hs]
    h1 = mk(hist)
    h2 = mk(hist[::-1])              # same tokens, REVERSED order
    h3 = mk(hist[:1] * len(hist))    # degenerate: one state repeated
    st = hist[-1]['state'].float().clone()
    out = []
    for hh in (h1, h2, h3):
        torch.manual_seed(0)                      # CRN: paired, not a sampling diff
        x, _ = dt.propose_mf(st, 1.0, 1.0, timestep=0, use_candidate_scoring=False,
                             candidate_features=None, fidelity_sampling=False, hist=hh)
        out.append(x.detach().reshape(-1).double())
    return (float((out[0] - out[1]).abs().max()), float((out[0] - out[2]).abs().max()))

pp, cc = probe(P), probe(C)
print(f"\nSC3 history changes the query (max|dx|):")
for nm, v in (("ctrl ", cc), ("nopos", pp)):
    print(f"    {nm}  reversed-history {v[0]:.3e}   degenerate-history {v[1]:.3e}")
sc3 = pp is not None and pp[1] > 0.0
print(f"    nopos still reads its context (degenerate-history delta > 0) -> "
      f"{'PASS' if sc3 else 'FAIL'}")
print(f"    NOTE: reversed-history delta is EXPECTED to be ~0 without position -- "
      f"order is then\n          carried only by the causal mask. That is the ablation "
      f"working, not a failure.")

# ---- SC4: CTRL is the arange-K8 arm h205 measured ----------------------------
sc4 = (C["seq_len"] == 8 and C["n_idx"] == 8)
print(f"\nSC4 CTRL is arange-K8: seq_len={C['seq_len']} (want 8), labelled idx="
      f"{C['n_idx']} (want 8)  -> {'PASS' if sc4 else 'FAIL'}")

print(f"\nSC2 identity gate: run separately via tools/identity_gate.py "
      f"(PASS at 122.29066752728207)")
ok = sc1 and sc3 and sc4
print(f"\nSTAGE 0: {'PASS' if ok else 'FAIL'}")
print("=" * 74)
sys.exit(0 if ok else 1)
