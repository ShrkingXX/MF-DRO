"""H102: add a `loc_loss` option to the DT's location head. HELD OUTSIDE src/.

Run with --apply when compute is clear; --revert to undo; --check to verify.
Not applied while workers run: a worker starting mid-edit imports a half-changed
module. This is the convention a peer session used for h94.

Three edits, each asserted to match exactly once:
  1. mf_dro.py    -- forward loc_loss into dt_cfg
  2. decisionTransformer.py __init__ -- store it on the model
  3. decisionTransformer.py forward_mf -- select L1 vs MSE at both call sites

Default is 'mse' at every step, so an unset config is bit-identical. The
forwarding edit exists because this file already records an H20 bug where a
dt_cfg field was added and never forwarded, leaving an ablation unreachable --
the same mistake would make this arm silently a no-op.
"""
import sys, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MF = os.path.join(REPO, "src", "policy", "mf_dro.py")
DT = os.path.join(REPO, "src", "model", "decisionTransformer.py")

EDITS = [
 (MF,
  """            rtg_conditioning=getattr(config, 'rtg_conditioning', 'token'),""",
  """            rtg_conditioning=getattr(config, 'rtg_conditioning', 'token'),
            # H102: 'mse' (default, unchanged) or 'l1'. An L2 loss fits the
            # conditional MEAN, which is pulled inward from a boundary whenever
            # any target mass lies away from it; an L1 loss fits the MEDIAN,
            # which sits AT the bound once half the mass is there. Forwarded
            # here deliberately -- see the H20 note below for what happens to a
            # dt_cfg field that is not.
            loc_loss=getattr(config, 'loc_loss', 'mse'),"""),
 (DT,
  """        self.input_dim = input_dim
        self.action_dim = action_dim""",
  """        self.input_dim = input_dim
        self.action_dim = action_dim
        # H102: which loss trains the location head. 'mse' reproduces every
        # prior run bit-for-bit.
        self.loc_loss = str(getattr(config, 'loc_loss', 'mse')).lower()"""),
 (DT,
  """                L_loc = (
                    F.mse_loss(x_pred, actions_x, reduction='none')
                    .mean(dim=-1)   # [B, T]
                    * vm
                ).sum() / vm.sum().clamp_min(1)
            else:
                L_loc = F.mse_loss(x_pred, actions_x)""",
  """                _lf = F.l1_loss if getattr(self, 'loc_loss', 'mse') == 'l1' else F.mse_loss
                L_loc = (
                    _lf(x_pred, actions_x, reduction='none')
                    .mean(dim=-1)   # [B, T]
                    * vm
                ).sum() / vm.sum().clamp_min(1)
            else:
                _lf = F.l1_loss if getattr(self, 'loc_loss', 'mse') == 'l1' else F.mse_loss
                L_loc = _lf(x_pred, actions_x)"""),
]

def run(mode):
    for path, old, new in EDITS:
        s = open(path).read()
        a, b = (old, new) if mode == "apply" else (new, old)
        if mode == "check":
            print(f"  {os.path.basename(path):26s} applied={new in s}  original={old in s}")
            continue
        if b in s and a not in s:
            print(f"  {os.path.basename(path):26s} already in target state"); continue
        assert s.count(a) == 1, f"{path}: expected exactly 1 match, found {s.count(a)}"
        open(path, "w").write(s.replace(a, b, 1))
        print(f"  {os.path.basename(path):26s} {mode}ed")

if __name__ == "__main__":
    m = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "check"
    assert m in ("apply", "revert", "check"), "use --apply | --revert | --check"
    run(m)
