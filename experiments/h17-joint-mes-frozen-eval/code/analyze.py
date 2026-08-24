"""H17 analysis. Reports per-seed traces alongside mean +/- SE, per PROTOCOL.md."""
import os, json, glob
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
H17 = os.path.join(HERE, "..", "results")
H1 = os.path.join(HERE, "..", "..", "h1-leak-fix-validation", "results")
SEEDS = list(range(42, 52))

# Baselines, reused from h1 per ../protocol.md (identical seeds/init/budget;
# neither baseline contains a DT or a reward, so this intervention cannot
# affect them). Stated in the protocol BEFORE launch.
BASE = {"MF-MI-Greedy": (0.5091, 0.1266), "MF-GP-UCB": (1.7934, 0.1223)}


def final_regret(path):
    d = json.load(open(path))
    c = d.get("hf_regret_curve") or d.get("regret_curve")
    return float(c[-1]) if c else float("nan")


def load(dirpath, method):
    out = {}
    for s in SEEDS:
        p = os.path.join(dirpath, f"{method}__seed{s}.json")
        if os.path.exists(p):
            out[s] = final_regret(p)
    return out


new = load(H17, "MF-DRO")
old = load(H1, "MF-DRO")
common = sorted(set(new) & set(old))
print(f"H17 (mes_entropy) complete: {len(new)}/10   paired with h1: {len(common)}\n")
print(f"{'seed':>5} {'improvement':>13} {'mes_entropy':>13} {'diff':>9}")
for s in common:
    print(f"{s:>5} {old[s]:>13.4f} {new[s]:>13.4f} {new[s]-old[s]:>+9.4f}")

if len(new) < 10:
    print(f"\n{10-len(new)} run(s) still outstanding -- NOT the final analysis.")

a = np.array([new[s] for s in common]); b = np.array([old[s] for s in common])
if len(common) >= 2:
    m, se = a.mean(), a.std(ddof=1)/np.sqrt(len(a))
    mo = b.mean()
    print(f"\nmes_entropy  {m:.4f} +/- {se:.4f}   (n={len(a)})")
    print(f"improvement  {mo:.4f} +/- {b.std(ddof=1)/np.sqrt(len(b)):.4f}")
    d = a - b
    try:
        p = wilcoxon(a, b).pvalue
    except Exception:
        p = float("nan")
    print(f"paired diff  {d.mean():+.4f}   better on {(d<0).sum()}/{len(d)} seeds"
          f"   Wilcoxon p={p:.4f}")
    print("\n" + "=" * 70)
    print(f"PRED 1 (mes_entropy LOWER regret than improvement, p<0.05): "
          f"{'PASS' if (d.mean()<0 and p<0.05) else 'FAIL'}")
    best_lo = min(mu - s_ for mu, s_ in BASE.values())
    print(f"PRED 2 FROZEN SUCCESS TEST: mean+SE {m+se:.4f} < best-baseline "
          f"mean-SE {best_lo:.4f}  ->  {'PASS' if (m+se) < best_lo else 'FAIL'}")
    if not (d.mean() < 0 and p < 0.05):
        print("\nPRED 3 NULL: a demonstrably better conditioning signal (H14/H16)")
        print("  buys no regret. Composes with H5/H8 (score head barely reads h)")
        print("  and H6/H7 (retraining moves 18% of decisions for ~0 regret) into")
        print("  a coherent negative -- and is exactly what Brandfonbrener et al.")
        print("  Fig 1c predicts: in stochastic environments the RCSL bias remains")
        print("  REGARDLESS of the conditioning function.")
    print("=" * 70)
