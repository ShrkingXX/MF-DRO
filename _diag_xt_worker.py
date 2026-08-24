from dro_runner import run_mf_single_seed

result = run_mf_single_seed(
    "mfdro_diag_xt", "Hartmann_6D", "MF-DRO", seed=43,
    bo_iterations=15,
    num_epochs=100,
    minimum_hf_fraction=0.25,
    real_hf_warmup=2,
)
print("DIAG-XT RUN DONE",
      "fidelity_trace=", result["fidelity_trace"],
      "final_regret=", result["hf_regret_curve"][-1])
