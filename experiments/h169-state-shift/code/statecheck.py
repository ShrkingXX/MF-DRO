"""Are the probed states actually DIFFERENT vectors? If real == train, h169's
state axis is a no-op dressed as a manipulation."""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np, torch
import src.policy.mf_dro as MF

CAP = {}
_ORIG = MF.DirectMFRegretOptimization._propose_next_query if hasattr(
    MF.DirectMFRegretOptimization, "_propose_next_query") else None

# Capture the states the probe would see, by wrapping propose_mf.
from src.model.decisionTransformer import DecisionTransformer as _DT
_OP = _DT.propose_mf
def _wrap(self, state, rtg, btg, **kw):
    CAP.setdefault("states", []).append(np.asarray(state.detach().cpu()).reshape(-1).copy())
    return _OP(self, state, rtg, btg, **kw)
_DT.propose_mf = _wrap

sys.argv = ["sc", "Borehole_8D", "42", "random"]
_s = importlib.util.spec_from_file_location("h169w", os.path.join(H, "worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h169w"] = w; _s.loader.exec_module(w)
w.h83.BUDGET = 6.0
r = w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h169_sc.json"))

S = np.array(CAP["states"])
P = r.get("h168_probe", [])
nper = len(P[0]["probes"]) if P else 0
print(f"\n  propose_mf calls captured : {len(S)}   state dim {S.shape[1]}")
print(f"  probes per iteration      : {nper}  (1 real query + {nper} probes per iter)")
# within one iteration's probe block: 9 RTG x 5 states, states in blocks of 9
if len(S) >= 46:
    blk = S[1:46]                      # skip the real query, take one probe block
    reps = blk[::9]                    # first probe of each state group
    print(f"\n  the {len(reps)} probed state vectors, pairwise L2 distances:")
    D = np.linalg.norm(reps[:, None, :] - reps[None, :, :], axis=-1)
    for i in range(len(reps)):
        print("    " + "  ".join(f"{D[i,j]:7.4f}" for j in range(len(reps))))
    off = D[np.triu_indices(len(reps), 1)]
    print(f"\n  max off-diagonal distance : {off.max():.6f}")
    print(f"  STATES DISTINCT           : {'YES' if off.max() > 1e-6 else 'NO -- the state axis is a NO-OP'}")
    print(f"  (real is row 0; train0-3 are rows 1-4)")
