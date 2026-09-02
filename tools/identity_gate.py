#!/usr/bin/env python3
"""Bit-identity gate for the DEFAULT (use_roi=False) path.

Every change to src/policy/mf_dro.py must leave this untouched. The reference
value 122.2906675273 has gated h155/h171/h177/h178/h179 and is recorded in
findings.md. Run it BEFORE launching any arm that touches the core file.

  python tools/identity_gate.py            # check against the reference
  python tools/identity_gate.py --print    # just print what this build gives
"""
import os, sys, importlib.util, argparse

# Full precision. findings.md records this as 122.2906675273, which is the same
# number displayed to 13 significant figures; comparing against the ROUNDED value
# fails a genuinely identical build by ~1.8e-11. Verified against a HEAD checkout
# on 2026-09-02 before the h184 patch.
REF = 122.29066752728207
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run():
    sys.path.insert(0, REPO)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    spec = importlib.util.spec_from_file_location(
        "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
    h83 = importlib.util.module_from_spec(spec)
    sys.modules["h83w"] = h83
    spec.loader.exec_module(h83)
    h83.BUDGET = 8.0
    out = os.path.join(os.environ.get("SCRATCH", "/tmp"), "identity_gate.json")
    r = h83.run("Borehole_8D", "MF-DRO", 42, out)
    return float(r["final_regret"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="show")
    a = ap.parse_args()
    got = run()
    if a.show:
        print(f"final_regret = {got!r}")
        sys.exit(0)
    ok = got == REF                      # exact: this is a bit-identity gate
    print(f"\n  reference : {REF!r}")
    print(f"  this build: {got!r}")
    print(f"  IDENTITY GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
