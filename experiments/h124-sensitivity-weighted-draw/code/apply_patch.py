"""H124: sensitivity-weighted candidate draw. HELD OUTSIDE src/ until the tree is quiet.

    --apply | --revert | --check

ONE edit, asserted to match exactly once. Default behaviour is bit-identical:
when `sens_draw_gamma` is unset the draw is the original uniform `torch.rand`
call, same shape, same RNG consumption, same tensor.

WHAT IT DOES. `_draw_raw` currently draws uniformly over the box, treating every
dimension alike. The ROI then filters that draw -- and a filter cannot create
probability mass the proposal never had, which is why no amount of ROI tightening
has concentrated the search where the objective actually lives.

This scales per-dimension spread by the GP's own fitted ARD lengthscales: short
lengthscale (the dimension matters) -> draw tightly around the incumbent; long
lengthscale (it does not) -> stay broad. Sensitivity is LEARNED, never oracle.
"""
import sys, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MF = os.path.join(REPO, "src", "policy", "mf_dro.py")

OLD = '''        def _draw_raw():
            return (bounds[0]
                    + (bounds[1] - bounds[0])
                    * torch.rand(roi_raw_pool, ko_model.d,
                                 device=ko_model.device,
                                 dtype=ko_model.dtype))'''

NEW = '''        def _draw_raw():
            # H124: sensitivity-weighted draw. gamma=0 (default) reproduces the
            # original uniform draw EXACTLY -- same shape, same RNG draw, same
            # arithmetic -- so every prior configuration is bit-identical.
            _u = torch.rand(roi_raw_pool, ko_model.d,
                            device=ko_model.device, dtype=ko_model.dtype)
            _g = float(globals().get('_H124_GAMMA', 0.0) or 0.0)
            if _g <= 0.0:
                return bounds[0] + (bounds[1] - bounds[0]) * _u
            # per-dimension relative width from the GP's OWN fitted ARD
            # lengthscales; short lengthscale => sensitive => narrow.
            try:
                _ls = (ko_model.gp_delta.covar_module.base_kernel
                       .lengthscale.detach().reshape(-1)[:ko_model.d])
                _w = (_ls / _ls.max()).clamp(1e-3, 1.0) ** _g
            except Exception:
                return bounds[0] + (bounds[1] - bounds[0]) * _u
            # centre on the incumbent, in NORMALISED coordinates
            try:
                _Y = real_data_hf[1].reshape(-1)
                _c = real_data_hf[0][int(torch.argmax(_Y))].reshape(-1)
                _c = ((_c - bounds[0]) / (bounds[1] - bounds[0])).clamp(0.0, 1.0)
            except Exception:
                _c = torch.full((ko_model.d,), 0.5, device=ko_model.device,
                                dtype=ko_model.dtype)
            _z = (_c.unsqueeze(0) + (_u - 0.5) * _w.unsqueeze(0)).clamp(0.0, 1.0)
            return bounds[0] + (bounds[1] - bounds[0]) * _z'''

def run(mode):
    s = open(MF).read()
    a, b = (OLD, NEW) if mode == "apply" else (NEW, OLD)
    if mode == "check":
        print(f"  applied={NEW in s}   original-present={OLD in s}"); return
    if b in s and a not in s:
        print("  already in target state"); return
    assert s.count(a) == 1, f"expected exactly 1 match, found {s.count(a)}"
    open(MF, "w").write(s.replace(a, b, 1))
    print(f"  {mode}ed")

if __name__ == "__main__":
    m = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "check"
    assert m in ("apply", "revert", "check")
    run(m)
