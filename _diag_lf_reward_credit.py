"""
Diagnostic: does the `improvement` reward discount LF's lower certainty --
or does it PAY A PREMIUM for uncertainty?

Two parts:
 (1) analytic: E[max(0, y-best)] for y ~ N(mu, sigma^2) IS Expected Improvement,
     which is monotone INCREASING in sigma. Shown numerically.
 (2) empirical: across a real rollout batch, does LF-heavy => higher rtg[0]?
     Confound to respect: more HF steps also means more chances at a nonzero
     reward term, which pushes the other way. Net sign is a real measurement.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from scipy import stats

from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

print("=" * 72)
print("(1) ANALYTIC -- expected per-step improvement reward vs posterior sigma")
print("=" * 72)
print("  y ~ N(mu, sigma^2);  r = max(0, y - best);  E[r] = EI(mu,sigma)")
print(f"  {'sigma':>8} {'E[r] (mu=best)':>16} {'E[r] (mu=best-1)':>18}")
rng = np.random.default_rng(0)
for sg in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
    a = np.maximum(0.0, rng.normal(0.0, sg, 400_000)).mean()
    b = np.maximum(0.0, rng.normal(-1.0, sg, 400_000)).mean()
    print(f"  {sg:>8.2f} {a:>16.4f} {b:>18.4f}")
print("\n  E[r] rises MONOTONICALLY with sigma. A rollout that keeps the HF")
print("  posterior wide earns MORE expected reward. LF queries shrink sigma_H")
print("  only weakly (through rho), so LF-heavy rollouts keep the premium.")
print("  => the reward does not discount low certainty; it PAYS for it.\n")

print("=" * 72)
print("(2) EMPIRICAL -- does it bite in a real rollout batch?")
print("=" * 72)
hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg = _build_mf_dro_config("lfrew", "Hartmann_6D", "diag", 44,
                            bo_iterations=1, num_epochs=1, minimum_hf_fraction=0.25,
                            real_hf_warmup=2, cost_budget=1e9, initial_hf=36,
                            initial_lf=60, dkl_threshold=9999, bes_delta=0.0,
                            rollout_length=8)
cfg.seed = 44
mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch = mf._generate_rollout_batch()

lf_frac, rtg0, n_hf, nonzero = [], [], [], []
for t in batch:
    ell = t["actions_ell"].flatten().double()
    m = t.get("valid_mask")
    if m is not None:
        m = m.flatten().bool()
        ell = ell[: m.numel()][m]
    if ell.numel() == 0:
        continue
    lf_frac.append(float((ell == 0).double().mean()))
    n_hf.append(int((ell == 1).sum()))
    r = t["rtg"].flatten()
    rtg0.append(float(r[0]) if r.numel() else 0.0)
    nonzero.append(float((r > 1e-12).double().mean()))

lf_frac = np.array(lf_frac); rtg0 = np.array(rtg0); n_hf = np.array(n_hf)
print(f"  trajectories               : {len(rtg0)}")
print(f"  LF fraction  mean/min/max  : {lf_frac.mean():.3f} / {lf_frac.min():.3f} / {lf_frac.max():.3f}")
print(f"  n_HF per rollout mean      : {n_hf.mean():.2f}")
print(f"  rtg[0] == 0 (dead signal)  : {(rtg0 <= 1e-12).mean():.1%} of trajectories")
print(f"  mean frac of steps w/ r>0  : {np.mean(nonzero):.3f}\n")

if lf_frac.std() > 1e-9:
    s1 = stats.spearmanr(lf_frac, rtg0)
    s2 = stats.spearmanr(n_hf, rtg0)
    print(f"  Spearman(LF fraction, rtg[0]) = {s1.correlation:+.4f}  (p={s1.pvalue:.4f})")
    print(f"  Spearman(n_HF,        rtg[0]) = {s2.correlation:+.4f}  (p={s2.pvalue:.4f})")
    print("\n  If LF-fraction correlates POSITIVELY with rtg[0], the uncertainty")
    print("  premium dominates the 'more HF = more reward terms' confound, and")
    print("  the reward actively prefers less-informative rollouts.")
else:
    print("  LF fraction is CONSTANT across the batch -- the fidelity pattern is")
    print("  pinned (minimum_hf_fraction + teacher), so this correlation is")
    print("  undefined. That is itself the finding: the reward cannot express a")
    print("  preference over fidelity patterns that never vary.")
print("=" * 72)
