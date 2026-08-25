"""Large-scale MFBO comparison — protocol derived from the MFBO literature.

Initial design follows Best Practices (Nat Comput Sci 2025): ~10% of total cost
to initial sampling, split 50/50 IN COST between fidelities. This is a 6.4x
reduction from the previous setting (Hartmann: 348 init cost vs a 200 budget,
i.e. 64% of total spent before the optimizer acted), which plausibly compressed
every prior comparison.

BUDGET_MODE:
  "cost"  -- stop at COST_BUDGET (arms are cost-matched; recommended)
  "iters" -- stop at N_ITERS (arms are NOT cost-matched when fidelity mixes
             differ; a 300-iteration Hartmann run spans 300-2400 cost)
"""
BENCHMARKS = {
    #                d  c_H c_L   cost budget   n_HF  n_LF   (init ~= 10% of budget)
    "Currin_2D":   dict(budget=300, n_hf=5,  n_lf=15),
    "Hartmann_6D": dict(budget=900, n_hf=6,  n_lf=45),
    "Borehole_8D": dict(budget=400, n_hf=10, n_lf=20),
}
METHODS = ["MF-DRO", "MF-MES-Greedy", "MF-MI-Greedy", "MF-GP-UCB"]
SEEDS = list(range(42, 52))      # 10; the Best-Practices standard is 20
BUDGET_MODE = "cost"
N_ITERS = 300                    # used only when BUDGET_MODE == "iters"
ITER_CAP = 4000                  # runaway guard for cost mode
ROLLOUT_REWARD = "mes_entropy"
