import time
import torch

from src.models.ko_gp import KennedyOHaganGP, DeepKernel
from src.policy.mf_dro import _get_mf_state_dim

torch.set_default_dtype(torch.float64)

print("=" * 100)
print("CHECK 1: DeepKernel is a valid GPyTorch kernel")
print("=" * 100)
k = DeepKernel(d=6, d_feature=4)
x1 = torch.rand(5, 6, dtype=torch.float64)
x2 = torch.rand(3, 6, dtype=torch.float64)
K = k(x1, x2).evaluate()
print(f"K shape: {K.shape}")  # EXPECT: [5, 3]
print(f"K positive: {(K > 0).all()}")  # EXPECT: True
print(f"K finite: {K.isfinite().all()}")  # EXPECT: True

x = torch.rand(4, 6, dtype=torch.float64)
K_sq = k(x, x).evaluate()
print(f"K symmetric: {torch.allclose(K_sq, K_sq.T)}")  # EXPECT: True

print("\n" + "=" * 100)
print("CHECK 2: _build_gp works with DKL flag")
print("=" * 100)
ko = KennedyOHaganGP(d=6)
ko.bounds = torch.zeros(2, 6, dtype=torch.float64)
ko.bounds[1] = 1.0
X = torch.rand(20, 6, dtype=torch.float64)
Y = torch.rand(20, dtype=torch.float64)

gp_rbf = ko._build_gp(X, Y, use_dkl=False)
print(f"RBF covar type: {type(gp_rbf.covar_module)}")  # EXPECT: ScaleKernel

gp_dkl = ko._build_gp(X, Y, use_dkl=True)
print(f"DKL covar type: {type(gp_dkl.covar_module)}")  # EXPECT: DeepKernel

x_test = torch.rand(5, 6, dtype=torch.float64)
mu_rbf, var_rbf = (gp_rbf.posterior(x_test).mean, gp_rbf.posterior(x_test).variance)
mu_dkl, var_dkl = (gp_dkl.posterior(x_test).mean, gp_dkl.posterior(x_test).variance)
print(f"RBF posterior finite: {mu_rbf.isfinite().all()}")  # EXPECT: True
print(f"DKL posterior finite: {mu_dkl.isfinite().all()}")  # EXPECT: True

print("\n" + "=" * 100)
print("CHECK 3: fit() switches RBF -> DKL at threshold")
print("=" * 100)
ko3 = KennedyOHaganGP(d=6, dkl_threshold=15)
bounds = torch.zeros(2, 6, dtype=torch.float64)
bounds[1] = 1.0
X_lf = torch.rand(30, 6, dtype=torch.float64)
Y_lf = torch.rand(30, dtype=torch.float64)

X_hf_small = torch.rand(10, 6, dtype=torch.float64)
Y_hf_small = torch.rand(10, dtype=torch.float64)
ko3.fit(X_lf, Y_lf, X_hf_small, Y_hf_small, bounds)
print(f"n_hf=10, use_dkl={ko3.use_dkl}")  # EXPECT: False
print(f"gp_lf type: {type(ko3.gp_lf.covar_module)}")  # EXPECT: ScaleKernel

X_hf_large = torch.rand(20, 6, dtype=torch.float64)
Y_hf_large = torch.rand(20, dtype=torch.float64)
ko3.fit(X_lf, Y_lf, X_hf_large, Y_hf_large, bounds)
print(f"n_hf=20, use_dkl={ko3.use_dkl}")  # EXPECT: True
print(f"gp_lf type: {type(ko3.gp_lf.covar_module)}")  # EXPECT: DeepKernel

print("\n" + "=" * 100)
print("CHECK 4: _rebuild_frozen_gp preserves DKL weights")
print("=" * 100)
ko_dkl = KennedyOHaganGP(d=6, dkl_threshold=15)
ko_dkl.fit(X_lf, Y_lf, X_hf_large, Y_hf_large, bounds)

w_before = (ko_dkl.gp_lf.covar_module.feature_extractor.net[0].weight.detach().clone())

x_new = torch.rand(1, 6, dtype=torch.float64)
y_new = torch.rand(1, dtype=torch.float64)
aug_x = torch.cat([ko_dkl.train_x_lf, x_new])
aug_y = torch.cat([ko_dkl.train_y_lf, y_new])
new_gp = ko_dkl._rebuild_frozen_gp(ko_dkl.gp_lf, aug_x, aug_y)

w_after = (new_gp.covar_module.feature_extractor.net[0].weight.detach().clone())

print(f"DKL weights preserved: {torch.allclose(w_before, w_after)}")  # EXPECT: True
print(f"New GP type: {type(new_gp.covar_module)}")  # EXPECT: DeepKernel

print("\n" + "=" * 100)
print("CHECK 5: make_fantasy_ko works with DKL")
print("=" * 100)
ko_dkl.fit(X_lf, Y_lf, X_hf_large, Y_hf_large, bounds)

x_new2 = torch.rand(1, 6, dtype=torch.float64)
y_new2 = torch.tensor([0.5], dtype=torch.float64)

ko_fantasy_lf = ko_dkl.make_fantasy_ko(x_new2, y_new2, 'L')
print(f"Fantasy LF gp_lf type: {type(ko_fantasy_lf.gp_lf.covar_module)}")  # EXPECT: DeepKernel

ko_fantasy_hf = ko_dkl.make_fantasy_ko(x_new2, y_new2, 'H')
print(f"Fantasy HF gp_delta type: {type(ko_fantasy_hf.gp_delta.covar_module)}")  # EXPECT: DeepKernel

print(f"Original use_dkl: {ko_dkl.use_dkl}")  # EXPECT: True (not mutated)

print("\n" + "=" * 100)
print("CHECK 6: State dimension updated correctly")
print("=" * 100)
dim = _get_mf_state_dim(d=6, M=10)
print(f"State dim d=6, M=10: {dim}")  # EXPECT: 32

dim2 = _get_mf_state_dim(d=2, M=10)
print(f"State dim d=2, M=10: {dim2}")  # EXPECT: 28

print("\n" + "=" * 100)
print("CHECK 7: DKL gradient flow (weights actually update during fit)")
print("=" * 100)
ko_test = KennedyOHaganGP(d=6, dkl_threshold=1)
ko_test.fit(X_lf[:5], Y_lf[:5], X_hf_large[:3], Y_hf_large[:3], bounds)

w_init = (ko_test.gp_lf.covar_module.feature_extractor.net[0].weight.detach().clone())

ko_test.fit(X_lf[:5], Y_lf[:5], X_hf_large[:3], Y_hf_large[:3], bounds)

w_after2 = (ko_test.gp_lf.covar_module.feature_extractor.net[0].weight.detach().clone())

print(f"DNN weights changed after fit: {not torch.allclose(w_init, w_after2)}")  # EXPECT: True

print("\n" + "=" * 100)
print("EXTRA METRICS")
print("=" * 100)

from src.models.ko_gp import DeepKernelFeatureExtractor
fe = DeepKernelFeatureExtractor(d=6, d_feature=3)
n_params = sum(p.numel() for p in fe.parameters())
print(f"DNN parameter count: {n_params}")

# Hartmann_6D-scale timing: 30 LF + 20 HF points, d=6, bounds [0,1]^6
ko_time = KennedyOHaganGP(d=6, dkl_threshold=15)
bounds6 = torch.zeros(2, 6, dtype=torch.float64)
bounds6[1] = 1.0
X_lf_t = torch.rand(30, 6, dtype=torch.float64)
Y_lf_t = torch.rand(30, dtype=torch.float64)
X_hf_t = torch.rand(20, 6, dtype=torch.float64)
Y_hf_t = torch.rand(20, dtype=torch.float64)

t0 = time.time()
ko_time.fit(X_lf_t, Y_lf_t, X_hf_t, Y_hf_t, bounds6)
dkl_time = time.time() - t0
print(f"DKL fit time: {dkl_time:.2f}s")

ko_time2 = KennedyOHaganGP(d=6, dkl_threshold=1000)  # force RBF (never crosses threshold)
t0 = time.time()
ko_time2.fit(X_lf_t, Y_lf_t, X_hf_t, Y_hf_t, bounds6)
rbf_time = time.time() - t0
print(f"RBF fit time: {rbf_time:.2f}s")

print("\nAll 7 checks + extra metrics complete.")
