"""H16 worker: one (seed, reward) job -> per-group Spearman on three axes."""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "h15-fair-reward-gate", "code"))

import numpy as np, torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
import gate as g15   # reuse H15's extract/grouped_spearman -- same instrument


def run(seed, reward):
    g15.SEED = seed
    mf, batch, bounds, hf = g15.build(reward)
    rows = g15.extract(mf, batch, bounds, hf)
    return {"seed": seed, "reward": reward, "n_traj": len(rows),
            "mean_n_hf": float(np.mean([r["n_hf"] for r in rows])),
            **{k: g15.grouped_spearman(rows, k)
               for k in ("f_x0", "best_hf", "best_all")}}


if __name__ == "__main__":
    print(json.dumps(run(int(sys.argv[1]), sys.argv[2]), default=float))
