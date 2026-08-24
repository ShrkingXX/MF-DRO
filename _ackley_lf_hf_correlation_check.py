"""
Extended LF-HF correlation check for Ackley_10D, building on the existing
_ackley_sanity_checks.py CHECK B (global Pearson r, N=500, seed=0 -- Part 1
below reproduces it at larger N for continuity). Motivated by the
mfdro_stage2_v3 Ackley_10D result showing heavy LF usage (lf_fraction
0.85-0.96) converting to WORSE regret than the near-all-HF ablation
(lf_fraction ~0), despite CHECK B's r=0.75 sitting comfortably inside the
"moderate, by-design" target band [0.70, 0.85] -- global correlation isn't
obviously "bad enough" to explain the regret gap on its own, so this checks
whether the correlation holds up specifically where final regret is actually
decided: among already-promising (top-K% by HF) points and near the shared
optimum (x=[0.5]*10), rather than uniformly across the whole domain.

AckleyFunctionLF reshapes the decay-rate/frequency parameters (b,c) relative
to HF rather than adding a radial bias, specifically because the radial-bias
version stayed correlated at r~=0.98 regardless of scale (concentration of
measure makes 10D Ackley's value nearly a deterministic function of
||x-0.5|| for random x) -- the b/c reshaping instead changes LOCAL
curvature/oscillation structure (see synthetic.py's AckleyFunctionLF
docstring). So it's plausible the global r=0.75 is still substantially
carried by that same shared radial trend, while agreement conditional on
"already being close to x*" -- which is what's left to decide once
optimization has found the right basin -- is weaker. That would explain how
LF can be a legitimately informative global proxy (by design) and still be a
poor tool for final regret refinement specifically.
"""
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from benchmarks import get_benchmark

torch.set_default_dtype(torch.float64)

hf_spec = get_benchmark("Ackley_10D_HF")
lf_spec = get_benchmark("Ackley_10D_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()

# --- Part 1: global correlation (Pearson + Spearman, n=2000, uniform on [0,1]^10) ---
torch.manual_seed(0)
X_global = torch.rand(2000, 10, dtype=torch.float64)
hf_global = f_hf(X_global).numpy().ravel()
lf_global = f_lf(X_global).numpy().ravel()

r_p, _ = pearsonr(hf_global, lf_global)
r_s, _ = spearmanr(hf_global, lf_global)
print(f"PART 1 -- GLOBAL (n=2000, uniform on [0,1]^10):")
print(f"  Pearson r    = {r_p:.4f}")
print(f"  Spearman rho = {r_s:.4f}")
print()

# --- Part 2: correlation restricted to the top-K% by HF value -- the region
# an acquisition function is actually discriminating within once it has
# already found a promising area ---
print("PART 2 -- TOP-K% BY HF VALUE (does LF preserve ranking among the good points?):")
order = np.argsort(-hf_global)  # descending HF value (negate=True -> higher is better)
for frac in [0.20, 0.10, 0.05]:
    k = int(len(hf_global) * frac)
    idx = order[:k]
    r_p_top, _ = pearsonr(hf_global[idx], lf_global[idx])
    r_s_top, _ = spearmanr(hf_global[idx], lf_global[idx])
    print(f"  top {frac:.0%} (n={k}): Pearson r={r_p_top:.4f}  Spearman rho={r_s_top:.4f}")
print()

# --- Part 3: local neighborhood around the shared optimum x=[0.5]*10, at
# shrinking radii -- tests whether agreement holds up in the region that
# actually determines final regret once the right basin has been found ---
print("PART 3 -- LOCAL NEIGHBORHOOD AROUND x*=[0.5]*10 (shrinking radius):")
for radius in [0.30, 0.15, 0.05]:
    torch.manual_seed(1)
    X_local = 0.5 + (torch.rand(2000, 10, dtype=torch.float64) * 2 - 1) * radius
    X_local = X_local.clamp(0.0, 1.0)
    hf_local = f_hf(X_local).numpy().ravel()
    lf_local = f_lf(X_local).numpy().ravel()
    r_p_loc, _ = pearsonr(hf_local, lf_local)
    r_s_loc, _ = spearmanr(hf_local, lf_local)
    print(f"  radius={radius:.2f} (n=2000): Pearson r={r_p_loc:.4f}  Spearman rho={r_s_loc:.4f}")
print()

# --- Part 4: if you trusted LF's single best pick in the global sample, how
# good is that point actually, in HF terms, vs the true best HF point found? ---
print("PART 4 -- COST OF TRUSTING LF'S BEST PICK (global sample, n=2000):")
best_by_lf_idx = int(np.argmax(lf_global))
best_by_hf_idx = int(np.argmax(hf_global))
hf_at_lf_pick = hf_global[best_by_lf_idx]
true_best_hf = hf_global[best_by_hf_idx]
dist = float(np.linalg.norm(X_global[best_by_lf_idx].numpy() - X_global[best_by_hf_idx].numpy()))
domain_diag = float(np.sqrt(10))
print(f"  HF value at argmax_LF's location   = {hf_at_lf_pick:.4f}")
print(f"  true best HF value in sample       = {true_best_hf:.4f}")
print(f"  regret from trusting LF's top pick = {true_best_hf - hf_at_lf_pick:.4f}")
print(f"  ||argmax_LF - argmax_HF||          = {dist:.4f}  (domain diagonal = {domain_diag:.4f}, "
      f"{dist / domain_diag:.1%} of it)")
