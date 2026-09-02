"""
example_asktell.py -- the loop you would actually run in the lab.

Run it as-is and a stand-in "measurement" is computed in software so you can
watch the whole thing work end to end:

    python example_asktell.py

The only line you have to change to make this real is `measure()`. Replace its
body with whatever produces a number in your lab -- a call into the cycler, a
lookup in a spreadsheet you fill in by hand, or a `input()` prompt that waits
for you to type the result in a week's time.
"""
import numpy as np

from sfdro import SFDRO

# --- the search space -------------------------------------------------------
# One row per bound, one column per knob, in whatever physical units you like.
PARAM_NAMES = ["additive_wt_pct", "charge_C_rate", "cutoff_voltage_V"]
BOUNDS = np.array([
    [0.0, 0.2, 4.0],     # lower limits
    [5.0, 3.0, 4.5],     # upper limits
])


# --- the experiment ---------------------------------------------------------
def measure(x):
    """
    STAND-IN. Returns cycles-to-80%-capacity for one cell.
    Replace the body with your real measurement; keep the signature.
    """
    additive, c_rate, v_cut = x
    cycles = (
        900.0
        - 60.0 * (additive - 2.0) ** 2
        - 180.0 * (c_rate - 0.8) ** 2
        - 4000.0 * (v_cut - 4.2) ** 2
        + 40.0 * np.sin(3.0 * additive) * np.cos(2.0 * c_rate)
    )
    return float(cycles + np.random.normal(0.0, 5.0))   # measurement noise


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)          # so the stand-in measurement noise is reproducible

    # --- (1) the initialization dataset -------------------------------------
    # In the lab this is the cells you have already cycled. Here we invent a
    # small space-filling batch. Anything with >= 2 rows works; the paper's
    # single-fidelity runs use 5-10 points before the first proposal.
    n_init = 8
    X_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * rng.random((n_init, 3))
    y_init = np.array([measure(x) for x in X_init])

    # If your history lives in a CSV with columns
    # additive_wt_pct,charge_C_rate,cutoff_voltage_V,cycles instead:
    #     import pandas as pd
    #     df = pd.read_csv("history.csv")
    #     X_init = df[PARAM_NAMES].to_numpy()
    #     y_init = df["cycles"].to_numpy()

    opt = SFDRO(
        bounds=BOUNDS,
        minimize=False,          # more cycles is better
        seed=42,
        # Cheap settings so this example finishes in a couple of minutes.
        # Drop these three lines to get the paper's configuration.
        gp_num_models=5, rollouts_per_iter=20, rollout_length=4,
    )
    opt.initialize(X_init, y_init)
    print(f"seeded with {len(opt)} cells; best so far = {opt.best[1]:.1f} cycles\n")

    # --- (2) propose, measure, update ---------------------------------------
    for i in range(5):
        x_next = opt.ask()                       # SF-DRO decides what to try
        print(f"round {i+1}: build a cell at "
              + ", ".join(f"{n}={v:.3f}" for n, v in zip(PARAM_NAMES, x_next)))

        y_next = measure(x_next)                 # <-- the real experiment

        opt.tell(x_next, y_next)                 # feed the result back
        print(f"          measured {y_next:.1f} cycles; "
              f"best now {opt.best[1]:.1f}\n")

    x_best, y_best = opt.best
    print("best recipe found:")
    for n, v in zip(PARAM_NAMES, x_best):
        print(f"  {n:>18} = {v:.4f}")
    print(f"  {'cycles':>18} = {y_best:.1f}")

    X, y = opt.history                           # everything, for your records
    np.savetxt("campaign.csv", np.column_stack([X, y]), delimiter=",",
               header=",".join(PARAM_NAMES + ["cycles"]), comments="")
    print("\nfull campaign written to campaign.csv")


if __name__ == "__main__":
    main()
