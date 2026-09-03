"""h192 readout, written BEFORE any run finished.

Gate: TRANSFER RATIO = (control DT centroid - shifted DT centroid)
                     / (control teacher mean dist - shifted teacher mean dist)
  P1 mechanism HOLDS  : >= 0.50
  P2 partial          : 0.15 <= r < 0.50
  P3 mechanism FALSIFIED: < 0.15

SC (read FIRST): the shifted teacher's tau=0 mean must sit well inside the control's
0.7788, else the shift did not apply and nothing may be read.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid

B = get_benchmark("Borehole_8D_HF"); OPT = float(B["known_optimal_value"])
LO = np.asarray(B["domain_min"], float); HI = np.asarray(B["domain_max"], float)
R = HI - LO; C = np.full(len(LO), 0.5); G = np.linspace(0, 200, 201)


def arm(final_glob, ckpt_glob):
    TM, Q, RL = [], [], []
    for f in sorted(glob.glob(final_glob)):
        j = json.load(open(f)); ts = j.get("teacher_action_stats")
        if not ts:
            continue
        TM.append(np.linalg.norm(np.array(ts[-1]["mean"]) - C))
        hf = [(np.array(e["x"]) - LO) / R
              for e in j["queries"] if e["fid"] and not e.get("is_init")]
        if len(hf) >= 18:
            Q.append(np.linalg.norm(np.array(hf[-20:]).mean(0) - C))
    for f in sorted(glob.glob(ckpt_glob)):
        q = json.load(open(f))["queries"]
        init = max([float(e["cost_cum"]) for e in q if e.get("is_init")] + [0.0])
        c, s, b = [], [], -np.inf
        for e in q:
            if e["fid"]:
                b = max(b, float(e["y"]))
            if not e.get("is_init"):
                c.append(float(e["cost_cum"]) - init); s.append(float(-b - OPT))
        if c:
            RL.append(100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT))
    return np.mean(TM), np.mean(Q), np.mean(RL), len(TM)


if __name__ == "__main__":
    ct, cq, cr, cn = arm(
        f"{REPO}/experiments/h172-rollout-length/results/Borehole_8D__ROLLOUT1__seed4[2-6].json",
        f"{REPO}/experiments/h172-rollout-length/results/ckpt/Borehole_8D__ROLLOUT1__seed4[2-6].json")
    st, sq, sr, sn = arm(
        f"{REPO}/experiments/h192-tau0-shift/results/Borehole_8D__TAU0SHIFT__seed4[2-6].json",
        f"{REPO}/experiments/h192-tau0-shift/results/ckpt/Borehole_8D__TAU0SHIFT__seed4[2-6].json")
    print(f"  SC FIRST -- teacher tau=0 mean dist from centre:")
    print(f"    control (ROLLOUT1, n={cn}) {ct:.4f}")
    print(f"    shifted (TAU0SHIFT, n={sn}) {st:.4f}")
    if not (st < ct - 0.15):
        print("    SC FAIL -- shift did not apply; regret NOT read"); sys.exit(0)
    print("    SC PASS")
    dt_teacher = ct - st
    dt_dt = cq - sq
    ratio = dt_dt / dt_teacher
    print(f"\n  DT query centroid: control {cq:.4f}  shifted {sq:.4f}")
    print(f"  teacher shift imposed : {dt_teacher:+.4f}")
    print(f"  DT shift observed     : {dt_dt:+.4f}")
    print(f"  TRANSFER RATIO        : {ratio:.3f}   (h191 measured ~0.70 for ROI)")
    v = ("P1 mechanism HOLDS" if ratio >= 0.50 else
         "P3 mechanism FALSIFIED" if ratio < 0.15 else "P2 partial")
    print(f"\n  REGISTERED VERDICT: {v}")
    print(f"\n  secondary -- frozen rel%: control {cr:.2f}  shifted {sr:.2f}  ({sr-cr:+.2f})")
