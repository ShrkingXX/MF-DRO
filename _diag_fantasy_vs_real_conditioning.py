"""
Fantasy-vs-real conditioning diagnostic (testing the "confidently wrong from
self-sampling" hypothesis): starting from the SAME fitted GP, run two
parallel 8-step chains, using the REAL acquisition mechanism
(compute_joint_mf_mes) to pick each step's query, on a FIXED roi_candidates
pool shared by both chains (matching simulate_mf_trajectory's own
convention -- drawn once before the loop):

  Chain FANTASY: samples y_tau from the GP's OWN posterior (ko.sample_fantasy)
                 and conditions on that -- exactly what simulate_mf_trajectory
                 does for every MF-DRO rollout.
  Chain REAL:    evaluates the TRUE benchmark function at the same kind of
                 query (its own independently-evolving acquisition choice)
                 and conditions on THAT instead -- what a real BO loop does.

Both chains use the identical conditioning machinery (make_fantasy_ko,
frozen-hyperparameter rebuild) -- the ONLY difference is where the
conditioned y-value comes from. If fantasy sampling is what makes the GP
"confidently wrong," chain FANTASY's calibration (measured against the true
function via a shared fixed probe set) should degrade over the 8 steps in a
way chain REAL's doesn't -- and that divergence should be much larger on
Hartmann_6D (where MF-DRO freezes) than on Ackley_10D (where it doesn't).

Run on both benchmarks so the divergence itself can be compared, not just
each chain's absolute calibration.
"""
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BENCHMARK = sys.argv[1]
seed = int(sys.argv[2])

ROLLOUT_LENGTH = 8
INITIAL_HF = 30
INITIAL_LF = 30
N_PROBES = 300
EXP_NAME = "diag_fantasy_vs_real"

OUT_DIR = os.path.join("results", EXP_NAME, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{BENCHMARK}__seed{seed}.json")
tag = f"[{BENCHMARK} seed{seed}]"

if os.path.exists(out_path):
    print(f"{tag} SKIPPED (already exists)", flush=True)
    sys.exit(0)

from benchmarks import get_benchmark
from src.utils.init_design import make_initial_design
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes

torch.set_default_dtype(torch.float64)
hf_spec = get_benchmark(BENCHMARK + "_HF")
lf_spec = get_benchmark(BENCHMARK + "_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()
d = hf_spec["dim"]
bounds = torch.tensor([hf_spec["domain_min"], hf_spec["domain_max"]], dtype=torch.float64)
c_H, c_L = hf_spec["cost"], lf_spec["cost"]


def f_true_h(x):
    return f_hf(x.unsqueeze(0)).reshape(-1)[0].item()


torch.manual_seed(seed)
np.random.seed(seed)

X_hf = make_initial_design(bounds, d, INITIAL_HF, seed, seed_offset=0)
Y_hf = torch.tensor([f_true_h(x) for x in X_hf], dtype=torch.float64)
X_lf = make_initial_design(bounds, d, INITIAL_LF, seed, seed_offset=1)
Y_lf = torch.tensor([f_lf(x.unsqueeze(0)).reshape(-1)[0].item() for x in X_lf], dtype=torch.float64)

ko0 = KennedyOHaganGP(d=d, dkl_threshold=9999)
ko0.fit(X_lf, Y_lf, X_hf, Y_hf, bounds)

# Fixed probe set for calibration measurement (shared by both chains, all steps).
torch.manual_seed(seed + 5000)
probes = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_PROBES, d, dtype=torch.float64)
probe_true = torch.tensor([f_true_h(x) for x in probes], dtype=torch.float64)
true_best_probe_idx = int(probe_true.argmax())
true_best_probe_val = probe_true[true_best_probe_idx].item()

# Fixed roi_candidates for the acquisition, drawn ONCE -- matches
# simulate_mf_trajectory's own convention (not resampled per step).
torch.manual_seed(seed + 6000)
roi_candidates = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(200, d, dtype=torch.float64)


def calibration(ko):
    with torch.no_grad():
        mu, var = ko.hf_posterior(probes)
    mu_np, var_np = mu.numpy(), var.clamp_min(1e-12).numpy()
    sigma_np = np.sqrt(var_np)
    corr = float(np.corrcoef(mu_np, probe_true.numpy())[0, 1])
    within = np.abs(probe_true.numpy() - mu_np) <= 1.96 * sigma_np
    coverage_95 = float(within.mean())
    believed_best_idx = int(mu_np.argmax())
    believed_best_true_val = probe_true[believed_best_idx].item()
    belief_regret = true_best_probe_val - believed_best_true_val
    return dict(corr=corr, coverage_95=coverage_95, belief_regret=belief_regret)


log = {"fantasy": [], "real": []}
chain_fantasy = ko0
chain_real = ko0

cal0 = calibration(ko0)
print(f"{tag} step=-1 (initial) corr={cal0['corr']:.4f} coverage_95={cal0['coverage_95']:.4f} "
      f"belief_regret={cal0['belief_regret']:.4f}", flush=True)

for tau in range(ROLLOUT_LENGTH):
    # --- Chain FANTASY: matches simulate_mf_trajectory exactly ---
    x_f, ell_f, _ = compute_joint_mf_mes(chain_fantasy, roi_candidates, c_H, c_L)
    y_f_sampled = chain_fantasy.sample_fantasy(x_f, 'LH'[ell_f])
    y_f_true_at_x = f_true_h(x_f) if ell_f == 1 else f_lf(x_f.unsqueeze(0)).reshape(-1)[0].item()
    sample_error = float(y_f_sampled) - y_f_true_at_x
    chain_fantasy = chain_fantasy.make_fantasy_ko(
        x_f.unsqueeze(0), torch.tensor([float(y_f_sampled)], dtype=torch.float64), 'LH'[ell_f]
    )
    cal_f = calibration(chain_fantasy)
    log["fantasy"].append(dict(
        tau=tau, ell=ell_f, x=x_f.tolist(), y_sampled=float(y_f_sampled),
        y_true_at_x=y_f_true_at_x, sample_error=sample_error, **cal_f,
    ))

    # --- Chain REAL: same acquisition mechanism, but conditions on the
    # TRUE function value instead of a posterior sample ---
    x_r, ell_r, _ = compute_joint_mf_mes(chain_real, roi_candidates, c_H, c_L)
    y_r_true = f_true_h(x_r) if ell_r == 1 else f_lf(x_r.unsqueeze(0)).reshape(-1)[0].item()
    chain_real = chain_real.make_fantasy_ko(
        x_r.unsqueeze(0), torch.tensor([y_r_true], dtype=torch.float64), 'LH'[ell_r]
    )
    cal_r = calibration(chain_real)
    log["real"].append(dict(
        tau=tau, ell=ell_r, x=x_r.tolist(), y_true=y_r_true, **cal_r,
    ))

    print(f"{tag} step={tau} FANTASY(ell={ell_f} sample_err={sample_error:+.3f} "
          f"corr={cal_f['corr']:.4f} belief_regret={cal_f['belief_regret']:.4f}) | "
          f"REAL(ell={ell_r} corr={cal_r['corr']:.4f} belief_regret={cal_r['belief_regret']:.4f})",
          flush=True)

final_divergence = log["fantasy"][-1]["belief_regret"] - log["real"][-1]["belief_regret"]
out = dict(benchmark=BENCHMARK, seed=seed, initial_calibration=cal0, log=log,
           final_divergence_belief_regret=final_divergence,
           true_best_probe_val=true_best_probe_val)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"{tag} DONE | final_belief_regret: FANTASY={log['fantasy'][-1]['belief_regret']:.4f} "
      f"REAL={log['real'][-1]['belief_regret']:.4f} | divergence={final_divergence:+.4f}", flush=True)
