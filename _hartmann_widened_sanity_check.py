"""
Pre-BO sanity check for HartmannWidened6D (no BO). Per spec:
  - For each alpha in [1.0, 0.3, 0.1]: does f_HF(x_star) == 3.3224
    (standard Hartmann-6's published optimum), where
    x_star = [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]?
  - Empirical basin half-width: mean L2 distance to optimum over
    {x : f_HF(x) > 0.9 * f_HF(x_star)}.
  - Expectation being tested: basin gets wider as alpha decreases.

NOTE: this also reports the TRUE (empirically found) per-alpha optimum
value/location alongside the fixed-x_star reading, since benchmarks.py's
registration already found these are not the same -- see
HARTMANN_WIDENED_OPTIMA in benchmarks.py and the comment above its
registration block.
"""
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

from src.objectives.synthetic import HartmannWidened6D
import benchmarks  # triggers registration + prints empirical optima

X_STAR_STANDARD = torch.tensor([0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573], dtype=torch.float64)
ALPHAS = [1.0, 0.3, 0.1]

print("\n" + "=" * 78)
print("SANITY CHECK 1: f_HF(x_star_standard) vs claimed-invariant 3.3224")
print("=" * 78)
for alpha in ALPHAS:
    f = HartmannWidened6D(alpha_basin=alpha)
    val = f(X_STAR_STANDARD.unsqueeze(0)).item()
    print(f"  alpha_basin={alpha:<4} f_HF(x_star_standard) = {val:.4f}   "
          f"(claimed-invariant target: 3.3224, delta={val - 3.3224:+.4f})")

print("\n" + "=" * 78)
print("SANITY CHECK 2: empirical basin half-width around x_star_standard")
print("  (mean L2 dist to x_star_standard over {x : f_HF(x) > 0.9*f_HF(x_star_standard)})")
print("  Sampling: N=2,000,000 points from a clipped Gaussian centered at")
print("  x_star_standard (std=0.35/dim, clipped to [0,1]^6) -- pure uniform")
print("  sampling of [0,1]^6 essentially never lands inside a basin this")
print("  narrow for alpha_basin=1.0, so this uses local importance sampling.")
print("=" * 78)
torch.manual_seed(0)
N = 2_000_000
for alpha in ALPHAS:
    f = HartmannWidened6D(alpha_basin=alpha)
    ref_val = f(X_STAR_STANDARD.unsqueeze(0)).item()
    thresh = 0.9 * ref_val
    noise = torch.randn(N, 6) * 0.35
    X = (X_STAR_STANDARD + noise).clamp(0.0, 1.0)
    vals = f(X).reshape(-1)
    mask = vals > thresh
    n_hit = mask.sum().item()
    if n_hit == 0:
        print(f"  alpha_basin={alpha:<4} 0/{N} samples cleared threshold={thresh:.4f} -- basin half-width unmeasurable at this sample size")
        continue
    dists = (X[mask] - X_STAR_STANDARD).norm(dim=-1)
    print(f"  alpha_basin={alpha:<4} n_hit={n_hit:>8}/{N}  mean_dist={dists.mean().item():.4f}  "
          f"max_dist={dists.max().item():.4f}  frac_hit={n_hit / N:.5f}")

print("\n" + "=" * 78)
print("FOR REFERENCE: TRUE per-alpha optimum (from benchmarks.py registration,")
print("random-sample + L-BFGS-B polish) -- this is what benchmarks.py actually")
print("uses as known_optimal_value/known_optimal_x per variant, NOT the fixed")
print("x_star_standard/3.3224 pair used in the two checks above.")
print("=" * 78)
for name, (val, x) in benchmarks.HARTMANN_WIDENED_OPTIMA.items():
    x_arr = np.array(x)
    dist_from_standard = np.linalg.norm(x_arr - X_STAR_STANDARD.numpy())
    print(f"  {name}: true_optimal_value={val:.4f}  true_argmax={np.round(x_arr, 4).tolist()}  "
          f"dist_from_x_star_standard={dist_from_standard:.4f}")
print()
