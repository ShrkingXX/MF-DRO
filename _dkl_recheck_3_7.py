import torch

from src.models.ko_gp import KennedyOHaganGP, DeepKernelFeatureExtractor

torch.set_default_dtype(torch.float64)

print("=" * 100)
print("Parameter count verification (defaults: d_feature=2)")
print("=" * 100)
for d in [2, 6, 8]:
    fe = DeepKernelFeatureExtractor(d=d, d_feature=2)
    n = sum(p.numel() for p in fe.parameters())
    print(f"d={d}, d_feature=2: {n} params  "
          f"(ratio at new threshold=30: {30/n:.2f}x)")

print("\n" + "=" * 100)
print("CHECK 3 (rerun): fit() switches RBF -> DKL at NEW threshold=30")
print("=" * 100)
ko3 = KennedyOHaganGP(d=6)  # defaults: dkl_threshold=30, d_feature=2
print(f"defaults -> dkl_threshold={ko3.dkl_threshold}, d_feature={ko3.d_feature}")
bounds = torch.zeros(2, 6, dtype=torch.float64)
bounds[1] = 1.0
X_lf = torch.rand(40, 6, dtype=torch.float64)
Y_lf = torch.rand(40, dtype=torch.float64)

X_hf_below = torch.rand(25, 6, dtype=torch.float64)
Y_hf_below = torch.rand(25, dtype=torch.float64)
ko3.fit(X_lf, Y_lf, X_hf_below, Y_hf_below, bounds)
print(f"n_hf=25 (below new threshold=30), use_dkl={ko3.use_dkl}")  # EXPECT: False
print(f"gp_lf type: {type(ko3.gp_lf.covar_module)}")  # EXPECT: ScaleKernel

X_hf_above = torch.rand(30, 6, dtype=torch.float64)
Y_hf_above = torch.rand(30, dtype=torch.float64)
ko3.fit(X_lf, Y_lf, X_hf_above, Y_hf_above, bounds)
print(f"n_hf=30 (at new threshold=30), use_dkl={ko3.use_dkl}")  # EXPECT: True
print(f"gp_lf type: {type(ko3.gp_lf.covar_module)}")  # EXPECT: DeepKernel
print(f"gp_lf.covar_module.feature_extractor.d_feature: "
      f"{ko3.gp_lf.covar_module.feature_extractor.d_feature}")  # EXPECT: 2

print("\n" + "=" * 100)
print("CHECK 7 (rerun): DKL gradient flow with gradient clipping active")
print("=" * 100)
ko_test = KennedyOHaganGP(d=6, dkl_threshold=1)  # force DKL immediately
X_lf_t = torch.rand(5, 6, dtype=torch.float64)
Y_lf_t = torch.rand(5, dtype=torch.float64)
X_hf_t = torch.rand(3, 6, dtype=torch.float64)
Y_hf_t = torch.rand(3, dtype=torch.float64)

ko_test.fit(X_lf_t, Y_lf_t, X_hf_t, Y_hf_t, bounds)
w_init = (ko_test.gp_lf.covar_module.feature_extractor.net[0].weight.detach().clone())
print(f"weights finite after 1st fit: {w_init.isfinite().all()}")

ko_test.fit(X_lf_t, Y_lf_t, X_hf_t, Y_hf_t, bounds)
w_after = (ko_test.gp_lf.covar_module.feature_extractor.net[0].weight.detach().clone())
print(f"weights finite after 2nd fit: {w_after.isfinite().all()}")
print(f"DNN weights changed after fit: {not torch.allclose(w_init, w_after)}")  # EXPECT: True

# Stress test: many repeated fits on tiny/noisy data, to see if grad
# clipping keeps things finite where it might not otherwise.
print("\nStress test: 20 repeated fits on tiny (3 LF + 2 HF) noisy data")
ko_stress = KennedyOHaganGP(d=6, dkl_threshold=1)
all_finite = True
for i in range(20):
    Xl = torch.rand(3, 6, dtype=torch.float64)
    Yl = torch.randn(3, dtype=torch.float64) * 100  # noisy/large scale
    Xh = torch.rand(2, 6, dtype=torch.float64)
    Yh = torch.randn(2, dtype=torch.float64) * 100
    ko_stress.fit(Xl, Yl, Xh, Yh, bounds)
    w = ko_stress.gp_lf.covar_module.feature_extractor.net[0].weight
    if not w.isfinite().all():
        all_finite = False
        print(f"  iter {i}: NON-FINITE weights detected")
print(f"All 20 stress-test fits produced finite weights: {all_finite}")

print("\nRecheck complete.")
