import torch
from scipy.stats import pearsonr

from benchmarks import get_benchmark

torch.set_default_dtype(torch.float64)

hf_spec = get_benchmark("Ackley_10D_HF")
lf_spec = get_benchmark("Ackley_10D_LF")
f_hf = hf_spec["make_objective"]()
f_lf = lf_spec["make_objective"]()

# CHECK A: shared optimum
x_star = torch.full((1, 10), 0.5, dtype=torch.float64)
hf_at_opt = f_hf(x_star)
lf_at_opt = f_lf(x_star)
print(f"CHECK A: HF at x*: {hf_at_opt.item():.6f}  (expect ~0.0)")
print(f"CHECK A: LF at x*: {lf_at_opt.item():.6f}  (expect ~0.0)")
assert abs(hf_at_opt.item()) < 1e-6
assert abs(lf_at_opt.item()) < 1e-6
print("CHECK A PASSED\n")

# CHECK B: LF-HF correlation
torch.manual_seed(0)
X = torch.rand(500, 10, dtype=torch.float64)
hf_vals = f_hf(X).numpy()
lf_vals = f_lf(X).numpy()
r, _ = pearsonr(hf_vals, lf_vals)
r2 = r ** 2
print(f"CHECK B: Pearson r(HF, LF) = {r:.4f}  (expect 0.70-0.85)")
print(f"CHECK B: R^2 = {r2:.4f}  (expect 0.50-0.72)")
in_range = 0.70 <= r <= 0.85
print(f"CHECK B: {'PASSED' if in_range else 'OUT OF TARGET RANGE'}\n")

# CHECK C: HF landscape spread (many local optima -> wide value range)
print(f"CHECK C: HF value range: [{hf_vals.min():.3f}, {hf_vals.max():.3f}]  (expect range > 5.0)")
val_range = hf_vals.max() - hf_vals.min()
print(f"CHECK C: range = {val_range:.3f}")
assert val_range > 5.0
print("CHECK C PASSED\n")

# CHECK D: cost ratio
cost_ratio = hf_spec["cost"] / lf_spec["cost"]
print(f"CHECK D: cost ratio c_H:c_L = {hf_spec['cost']}:{lf_spec['cost']} = {cost_ratio}  (expect 5.0)")
assert cost_ratio == 5.0
print("CHECK D PASSED\n")

print("Pearson r:", r)
