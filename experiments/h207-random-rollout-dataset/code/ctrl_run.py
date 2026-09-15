"""Short control run from a given repo root. usage:
   ctrl_run.py <repo_root> <out.json> <budget> <rollout_reward>
Config = h206N (K=8, no positional embedding, ROI-Q10, 60 MES rollouts/iter).
Used by Stage 0 SC0 (old-code vs new-code parity) and SC1 (reward fork is real)."""
import os, sys, json, importlib.util, io, contextlib, hashlib
REPO, OUT, BUDGET, REWARD = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import torch; torch.set_num_threads(1)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h83w"] = w; _s.loader.exec_module(w)
_OB = w._build_mf_dro_config
def _b(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8; c.max_seq_length = 256
    c.absolute_timesteps = False; c.real_prefix_training = False
    c.disable_position_embedding = True
    return c
w._build_mf_dro_config = _b
w.ROLLOUT_REWARD = REWARD
w.BUDGET = BUDGET
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = w.run("Borehole_8D", "MF-DRO", 42, OUT + ".ckpt")
q = r["queries"]
trace = [(round(float(e["cost_cum"]), 6), int(e["fid"]), round(float(e["y"]), 9)) for e in q]
md5 = hashlib.md5(json.dumps(trace).encode()).hexdigest()[:12]
json.dump(dict(md5=md5, n_q=len(q), rtg_target=r.get("rtg_target"),
               final_regret=r.get("final_regret")), open(OUT, "w"))
print(f"[ctrl_run] repo={REPO} reward={REWARD} n_q={len(q)} md5={md5}")
