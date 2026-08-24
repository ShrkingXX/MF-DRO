"""
DT action-collapse diagnostic: on a real (not simulated) Hartmann_6D MF-DRO
run, at every real BO iteration, probe the trained DT's predicted action at
SEVERAL RTG-target multipliers of the real target (0.25x, 0.5x, 1x, 2x, 4x),
using the exact real state/btg at that iteration -- then let the real
pipeline proceed completely unmodified (the probes are read-only forward
passes, no side effects on training/data/cost).

Tests: does the DT's predicted location (x_pred) stay essentially fixed
regardless of the RTG target it's conditioned on (collapsed/degenerate
policy -- exactly the same action no matter what outcome it's told to aim
for), or does it vary sensibly with RTG but just fail to help regret? Also
tracks whether the predicted location drifts across iterations (as the real
state/dataset grows) or stays pinned to one region the whole run.

Implementation: _propose_next_query is monkeypatched on the INSTANCE only,
wrapping (not replacing) the original bound method -- probes are additional
read-only self.dt.propose_mf calls using the same state/btg_now the real
call uses, executed alongside it; the real (x_t, ell_t) returned to run()
is byte-for-byte whatever the original method would have returned.
"""
import sys
import os
import json
import types

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 44
# Hard iteration cap -- the previous run already established the pattern
# (frozen regret, LF-lock-in) holds for hundreds of iterations once it
# starts; this rerun only needs enough iterations to capture the
# RTG-sensitivity probe trajectory through the lock-in itself, not to
# re-confirm it persists for hundreds more.
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

BENCHMARK = "Hartmann_6D"
COST_BUDGET = 9999  # not the stopping condition here -- bo_iterations is
INITIAL_HF = 18
INITIAL_LF = 30
NUM_EPOCHS = 10
RTG_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
EXP_NAME = "diag_dt_action_collapse"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__seed{seed}.json")
tag = f"[{BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    with open(out_path) as f:
        _existing = json.load(f)
    if _existing.get("done"):
        print(f"{tag} SKIPPED (already completed)", flush=True)
        sys.exit(0)
    else:
        print(f"{tag} found incomplete checkpoint (killed mid-run) -- restarting fresh", flush=True)

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization, _extract_mf_state

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)

torch.manual_seed(seed)
np.random.seed(seed)
config = _build_mf_dro_config(
    EXP_NAME, BENCHMARK, f"seed{seed}", seed,
    bo_iterations=N_ITERS, num_epochs=NUM_EPOCHS,
    minimum_hf_fraction=0.25, real_hf_warmup=2,
    cost_budget=COST_BUDGET, initial_hf=INITIAL_HF, initial_lf=INITIAL_LF,
    dkl_threshold=9999, bes_delta=0.0, rollout_length=8,
)
config.seed = seed
mf = DirectMFRegretOptimization(config, f_hf, f_lf, bounds)

probe_log = []
_orig_propose_next_query = mf._propose_next_query.__func__  # unbound, class-level


def _instrumented_propose_next_query(self):
    self._last_recent_hf_frac = (
        sum(self.recent_ell_history) / max(len(self.recent_ell_history), 1)
        if self.recent_ell_history else 0.5
    )
    state = _extract_mf_state(
        self.data_hf_x, self.data_hf_y,
        self.data_lf_x, self.data_lf_y,
        self.ko_ensemble,
        len(self.data_hf_y), self.config.bo_iterations,
        self.c_L, self.c_H,
        device='cpu', dtype=torch.float64,
        recent_hf_frac=self._last_recent_hf_frac,
        bounds=self.bounds,
    )
    btg_now = self.btg_target_base
    rtg_tgt = self._last_rtg_target

    probes = []
    for mult in RTG_MULTIPLIERS:
        x_p, ell_p = self.dt.propose_mf(
            state.float(), rtg_tgt * mult, btg_now,
            timestep=0, use_candidate_scoring=self.use_candidate_scoring,
        )
        probes.append(dict(rtg_multiplier=mult, rtg_value=rtg_tgt * mult,
                            x=x_p.tolist() if torch.is_tensor(x_p) else list(x_p),
                            ell=int(ell_p)))

    # Real decision -- unmodified, via the original bound method.
    x_t, ell_t = _orig_propose_next_query(self)

    probe_log.append(dict(
        iteration=len(self.data_hf_y) + len(self.data_lf_y) - INITIAL_HF - INITIAL_LF,
        n_hf=len(self.data_hf_y), n_lf=len(self.data_lf_y),
        rtg_target=rtg_tgt, btg_now=btg_now,
        real_x=x_t.tolist() if torch.is_tensor(x_t) else list(x_t),
        real_ell=int(ell_t),
        probes=probes,
    ))
    return x_t, ell_t


def _write_checkpoint(final_regret=None, n_iters=None, incumbent_improved_count=None, done=False):
    """Incremental checkpoint -- called after every real iteration (and once
    more at the end) so a killed/interrupted run still leaves the probe data
    collected so far on disk, instead of only ever writing at mf.run()'s
    natural completion."""
    probe_spreads = []
    real_locs = []
    for p in probe_log:
        xs = torch.tensor([pr['x'] for pr in p['probes']], dtype=torch.float64)
        spread = (xs - xs.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()
        probe_spreads.append(spread)
        real_locs.append(p['real_x'])
    real_locs_t = torch.tensor(real_locs, dtype=torch.float64) if real_locs else torch.zeros(0, mf.d)
    step_drift = (real_locs_t[1:] - real_locs_t[:-1]).norm(dim=1).tolist() if len(real_locs) > 1 else []

    out = dict(
        benchmark=BENCHMARK, seed=seed, done=done,
        final_regret=final_regret, n_iters=n_iters,
        incumbent_improved_count=incumbent_improved_count,
        probe_log=probe_log,
        mean_probe_spread=float(np.mean(probe_spreads)) if probe_spreads else None,
        mean_step_drift=float(np.mean(step_drift)) if step_drift else None,
        probe_spreads=probe_spreads,
        step_drift=step_drift,
    )
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp_path, out_path)  # atomic, never leaves a half-written file
    return out


_orig_instrumented = _instrumented_propose_next_query


def _instrumented_propose_next_query_checkpointed(self):
    x_t, ell_t = _orig_instrumented(self)
    _write_checkpoint()  # incremental, done=False, no regret/n_iters yet
    return x_t, ell_t


mf._propose_next_query = types.MethodType(_instrumented_propose_next_query_checkpointed, mf)

result = mf.run()

rc = result['hf_regret_curve']
n_improved = sum(1 for i in range(1, len(rc)) if rc[i] < rc[i - 1] - 1e-12)
out = _write_checkpoint(final_regret=rc[-1], n_iters=len(rc),
                         incumbent_improved_count=n_improved, done=True)
print(f"{tag} DONE | final_regret={rc[-1]:.4f} | n_iters={len(rc)} | "
      f"incumbent_improved_count={n_improved} | "
      f"mean_probe_spread={out['mean_probe_spread']:.4f} | "
      f"mean_step_drift={out['mean_step_drift']:.4f}", flush=True)
