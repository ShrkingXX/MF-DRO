"""h155 BIT-IDENTITY GATE, run through the REAL pipeline.

Loads either the patched src/policy/mf_dro.py or the pre-patch copy INTO
sys.modules under the canonical name before h83's worker imports it, then runs
a short real MF-DRO run. The default path (rollout_policy='mes') must produce a
bit-identical query trace.

  usage: gate.py {new|old} <out.json>
"""
import os, sys, json, importlib.util
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
which, out = sys.argv[1], sys.argv[2]

path = (os.path.join(REPO, "src/policy/mf_dro.py") if which == "new"
        else os.path.join(os.environ["SCRATCH"], "mf_dro_orig.py"))
import src, src.policy                                   # create the packages
sp = importlib.util.spec_from_file_location("src.policy.mf_dro", path)
m = importlib.util.module_from_spec(sp)
sys.modules["src.policy.mf_dro"] = m; sp.loader.exec_module(m)

_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
h83.BUDGET = 8.0
r = h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], f"g_{which}.json"))
json.dump(dict(x=r["x_t_trace"], y=r["y_t_trace"], f=r["fidelity_trace"],
               rtg=r["rtg_target"], regret=r["final_regret"]), open(out, "w"))
print(f"[{which}] regret={r['final_regret']:.6f} n={len(r['x_t_trace'])}", flush=True)
