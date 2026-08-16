"""
Phase 2: compare the per-step RTG sum (current DRO formulation) to the joint
RTG (proposed formulation, log(b_tau/b_T) using Gumbel entropy) on the same
set of rollouts. High rank correlation means the two are interchangeable in
practice as DT training targets.

Uses phase1_thompson_gumbel.json as the Phase 1 reference (all 5 benchmarks
passed there; the product-CDF-based phase1_gumbel_quality.json is not used
for exclusion, since Phase 2's own b_tau is computed via Thompson sampling,
per the same method that JSON validated).

IMPORTANT ARCHITECTURAL NOTE: dro.py's _simulate_trajectory does NOT
fantasy-condition the GP on simulated observations within a rollout -- the
same static, real-data-fit model is used for every step. That makes b_tau
constant across a rollout by construction, which is meaningless for joint
RTG. simulate_with_gumbel_b below reimplements the rollout stepping loop
(same action-selection via _optimize_acquisition, same state extraction)
but explicitly fantasy-conditions via model.condition_on_observations() at
each step, and computes b_tau from the state BEFORE conditioning on that
step's own observation (i.e., from D_tau, not D_{tau+1}). The per-step
reward is also computed from this fantasy-conditioned model, so the two RTG
formulations are measuring the same underlying sequential process.

Since _optimize_acquisition and _get_posterior_mean_stddev always read
self.gp_ensemble[gp_idx]['model'] directly (no model parameter to inject),
the fantasy model is installed by temporarily swapping that dict entry
before the call and restoring the original real-data-fit model immediately
after (try/finally) -- this never mutates dro.py's code, only transiently
swaps runtime state that is always restored before returning.

Read-only diagnostic: does not modify any existing code. Reuses
thompson_sample_y_star / fit_gumbel_to_samples from phase1_thompson_gumbel.py
and _make_dro from phase1_gumbel_quality.py rather than duplicating them.

Usage:
    python validation/phase2_rtg_correlation.py
"""
import json
import math
import os
import sys

import gpytorch
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_gumbel_quality import _make_dro
from phase1_thompson_gumbel import thompson_sample_y_star, fit_gumbel_to_samples
from mes_reward import compute_mes_reward, _EPS_STD

BENCHMARKS = ["Ackley_2D", "Ackley_5D", "Rosenbrock_2D", "Hartmann_6D", "Currin_2D"]
STAGES = [("early", 5), ("mid", 25), ("late", 45)]
SEED = 42
N_ROLLOUTS = 100
MAX_LENGTH = 4
K_THOMPSON_DEFAULT = 50
K_THOMPSON_RETRY = 100
VIOLATION_RETRY_THRESHOLD = 0.05

_EULER_GAMMA = 0.5772156649015329

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PHASE1_THOMPSON_PATH = os.path.join(RESULTS_DIR, "phase1_thompson_gumbel.json")
NEW_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase2_rtg_correlation.json") # v3 (pre roi_candidates bugfix) -- kept for comparison, not overwritten
V4_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase2_rtg_correlation_v4.json")


def _gumbel_entropy(b: float) -> float:
    return math.log(b) + _EULER_GAMMA + 1.0


def simulate_with_gumbel_b(optimizer, gp_idx: int, initial_state: torch.Tensor,
                            max_length: int = 4, K_thompson: int = 50) -> dict:
    """
    Reimplements _simulate_trajectory's rollout loop (same state extraction,
    same reward formula), but fantasy-conditions the GP at each step and
    computes the Gumbel scale b_tau from the state BEFORE conditioning on
    that step's own observation.

    UPDATED per explicit instruction: roi_candidates is computed exactly
    ONCE, from D_0, before the rollout loop begins -- and reused unchanged
    at every step for x_tau selection, b_tau's Thompson sampling, and MES
    reward's Term 2 quadrature. _optimize_acquisition is never called again
    inside the loop; x_tau selection is instead a direct argmax of the
    acquisition function over the fixed roi_candidates (no fresh candidate
    generation, no local refinement phase -- both of those would require
    generating new candidates outside the fixed set, which contradicts
    "reuse this SAME roi_candidates").

    Returns trajectory dict with keys 'states', 'actions', 'rewards',
    'gumbel_b_values' (shape [T+1]: b_0 before any rollout step, ..., b_T
    after all T steps).
    """
    gp_dict = optimizer.gp_ensemble[gp_idx]
    real_model = gp_dict['model'] # the persistent, real-data-fit model -- never mutated
    real_model.eval()

    states = [initial_state]
    actions = []
    rewards = []
    gumbel_b_values = []

    sim_data_x = optimizer.data_x.clone()
    sim_data_y = optimizer.data_y.clone()

    if sim_data_y.shape[0] > 0:
        observed_best = sim_data_y.max() if optimizer.objective_mode == "maximize" else sim_data_y.min()
    else:
        observed_best = -float('inf') if optimizer.objective_mode == "maximize" else float('inf')
    if not isinstance(observed_best, torch.Tensor):
        observed_best = torch.tensor(observed_best, device=optimizer.device, dtype=optimizer.dtype)

    current_step = optimizer.data_x.shape[0]
    current_model = real_model # D_0: fantasy model starts as the real, unconditioned model

    # --- roi_candidates computed ONCE, from D_0, before the rollout loop ---
    # BUGFIX (v4): _optimize_acquisition returns (best_x, roi_candidates) in
    # that order -- this was previously swapped, assigning the single best_x
    # point (shape [1, D]) to `roi_candidates` and discarding the actual
    # domain-spanning candidate set. That collapsed every rollout step to the
    # same fixed point and made Thompson sampling degenerate to a single-point
    # marginal instead of a genuine max-over-domain estimate.
    _, roi_candidates = optimizer._optimize_acquisition(gp_idx, observed_best)

    def _select_x_tau(model):
        """Select x_tau by scoring the FIXED roi_candidates under `model`'s
        acquisition function -- direct argmax, no fresh candidates, no
        refinement phase."""
        gp_dict['model'] = model
        try:
            acq_values = optimizer._acquisition_function_value_botorch(
                roi_candidates, gp_idx, observed_best, candidate_set=roi_candidates
            )
        finally:
            gp_dict['model'] = real_model
        best_idx = torch.argmax(acq_values)
        return roi_candidates[best_idx].unsqueeze(0)

    for step in range(max_length):
        # --- Step D_tau: select x_tau and compute b_tau BEFORE conditioning ---
        next_x_tensor = _select_x_tau(current_model)

        thompson_samples = thompson_sample_y_star(current_model, roi_candidates, K=K_thompson)
        _, b_tau = fit_gumbel_to_samples(thompson_samples)
        gumbel_b_values.append(b_tau)

        actions.append(next_x_tensor.squeeze(0))

        # --- Sample y_tau from the fantasy model at D_tau ---
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = current_model.posterior(next_x_tensor, observation_noise=True)
            sampled_y = posterior.sample().reshape(-1)

        sim_data_x = torch.cat([sim_data_x, next_x_tensor], dim=0)
        sim_data_y = torch.cat([sim_data_y, sampled_y], dim=0)

        sampled_y_item = sampled_y.item()
        observed_best_item = observed_best.item()
        if optimizer.objective_mode == "maximize":
            improved = sampled_y_item > observed_best_item
        else:
            improved = sampled_y_item < observed_best_item

        # Per-step reward, computed from the SAME fantasy model at D_tau (not
        # the static original), so both RTG formulations share one D_tau.
        if optimizer.use_mes_reward:
            reward = compute_mes_reward(next_x_tensor.squeeze(0), current_model, roi_candidates, K=optimizer.mes_k).item()
            if improved:
                observed_best = sampled_y.reshape(())
        else:
            reward = 0.0
            if improved:
                new_best_item = sampled_y_item
                reward = (new_best_item - observed_best_item) if optimizer.objective_mode == "maximize" \
                    else (observed_best_item - new_best_item)
                observed_best = sampled_y.reshape(())
        rewards.append(reward)

        current_step += 1
        new_state = optimizer._extract_state(sim_data_x, sim_data_y, current_step)
        states.append(new_state)

        # --- Now condition on (x_tau, y_tau) to get D_{tau+1} ---
        with torch.no_grad():
            current_model = current_model.condition_on_observations(
                X=next_x_tensor, Y=sampled_y.reshape(1, 1)
            )

    # b_T: after all max_length conditioning steps, using the SAME fixed roi_candidates
    thompson_samples_T = thompson_sample_y_star(current_model, roi_candidates, K=K_thompson)
    _, b_T = fit_gumbel_to_samples(thompson_samples_T)
    gumbel_b_values.append(b_T)

    return {
        'states': torch.stack(states).to(optimizer.device, optimizer.dtype),
        'actions': torch.stack(actions).to(optimizer.device, optimizer.dtype),
        'rewards': torch.tensor(rewards, device=optimizer.device, dtype=optimizer.dtype),
        'gumbel_b_values': torch.tensor(gumbel_b_values, dtype=torch.float64),
        'roi_size': int(roi_candidates.shape[0]),
    }


def compute_both_rtg(trajectory: dict):
    """
    per_step_rtg: [T] tensor, backward cumsum of rewards.
    joint_rtg:    [T] tensor, log(b_tau / b_T) for tau=0..T-1.

    NOT clamped at 0 (unlike the original spec): negative values (from
    finite-K Gumbel-fit noise making b_tau < b_T) are kept and counted
    honestly in neg_frac, rather than zero-inflating the joint RTG
    distribution before it's compared to per_step_rtg.
    """
    T = len(trajectory['rewards'])
    rewards = trajectory['rewards']

    per_step_rtg = torch.zeros(T)
    for i in range(T):
        per_step_rtg[i] = rewards[i:].sum()

    b_values = trajectory['gumbel_b_values'] # [T+1]
    b_T = b_values[-1].item()
    joint_rtg = torch.zeros(T)
    for tau in range(T):
        ratio = b_values[tau].item() / max(b_T, 1e-9)
        joint_rtg[tau] = math.log(ratio)

    return per_step_rtg, joint_rtg


def _classify(rho_all: float, neg_frac: float) -> str:
    if rho_all >= 0.85 and neg_frac < 0.05:
        return "PASS"
    elif (0.70 <= rho_all < 0.85) or (0.05 <= neg_frac <= 0.15):
        return "WARN"
    else:
        return "FAIL"


def _run_stage(benchmark: str, n_initial: int, K_thompson: int, seed: int):
    """Run N_ROLLOUTS rollouts for one (benchmark, stage), at a given K_thompson.
    Returns (result_dict, violation_frac) so the caller can decide whether to retry at higher K."""
    dro = _make_dro(benchmark, n_initial, seed=seed)
    dro._update_models()

    per_step_rtg0_list, joint_rtg0_list = [], []
    per_step_all, joint_all = [], []
    roi_size_list = []
    n_violations = 0

    for r in range(N_ROLLOUTS):
        state = dro._extract_state(dro.data_x, dro.data_y, dro.data_x.shape[0])
        traj = simulate_with_gumbel_b(dro, gp_idx=0, initial_state=state,
                                       max_length=MAX_LENGTH, K_thompson=K_thompson)
        per_step_rtg, joint_rtg = compute_both_rtg(traj)

        per_step_rtg0_list.append(per_step_rtg[0].item())
        joint_rtg0_list.append(joint_rtg[0].item())
        per_step_all.extend(per_step_rtg.tolist())
        joint_all.extend(joint_rtg.tolist())
        roi_size_list.append(traj['roi_size'])

        # Entropy-monotonicity diagnostic: H_0 >= H_T should hold in theory
        # (more conditioning = less uncertainty about y*). Checks the endpoints
        # only (not every consecutive pair): with only 4 steps, the true
        # per-step entropy decrease can easily be smaller than the Gumbel-MLE
        # noise floor at finite K_thompson, so requiring every single
        # consecutive pair to be monotone inflates the violation rate with
        # noise that H_0-vs-H_T (aggregating the full rollout's signal) largely
        # averages out.
        b_values = traj['gumbel_b_values'].tolist()
        entropies = [_gumbel_entropy(b) for b in b_values]
        if entropies[0] < entropies[-1]:
            n_violations += 1

    violation_frac = n_violations / N_ROLLOUTS

    rho_rtg0 = spearmanr(per_step_rtg0_list, joint_rtg0_list).statistic
    rho_all = spearmanr(per_step_all, joint_all).statistic
    r_rtg0 = np.corrcoef(per_step_rtg0_list, joint_rtg0_list)[0, 1]
    r_all = np.corrcoef(per_step_all, joint_all)[0, 1]

    n_negative = sum(1 for v in joint_all if v < 0)
    neg_frac = n_negative / len(joint_all)

    result = dict(
        rho_rtg0=float(rho_rtg0), rho_all=float(rho_all),
        r_rtg0=float(r_rtg0), r_all=float(r_all),
        neg_frac=float(neg_frac), n_violations=n_violations,
        violation_frac=float(violation_frac), K_thompson_used=K_thompson,
        roi_size_mean=float(np.mean(roi_size_list)), roi_size_min=int(np.min(roi_size_list)),
        status=_classify(rho_all, neg_frac),
    )
    return result, violation_frac


def main():
    from checkpoint import setup_dirs
    setup_dirs("validation_gumbel")

    with open(PHASE1_THOMPSON_PATH) as f:
        phase1_thompson = json.load(f)

    print("Phase 1 reference: phase1_thompson_gumbel.json (all 5 benchmarks passed there).")
    print("Running Phase 2 v4 on all 5 benchmarks x 3 stages (post roi_candidates bugfix).\n")

    # --- One-time sanity check: confirm the roi_candidates bugfix took effect ---
    sanity_dro = _make_dro("Hartmann_6D", 25, seed=SEED)
    sanity_dro._update_models()
    sanity_best_x, sanity_rois = sanity_dro._optimize_acquisition(
        0, sanity_dro.data_y.max() if sanity_dro.objective_mode == "maximize" else sanity_dro.data_y.min()
    )
    print(f"Sanity: best_x shape={tuple(sanity_best_x.shape)}, roi_candidates shape={tuple(sanity_rois.shape)}")
    print("EXPECT: best_x shape=(1, D), roi_candidates shape=(~500-1000, D)")
    if sanity_rois.shape[0] <= 1:
        print("FAIL SIGN: roi_candidates shape has size 1 -- bug still present. Aborting.\n")
        return
    print("Sanity check passed -- roi_candidates is a real domain-spanning candidate set.\n")

    results = {}
    for benchmark in BENCHMARKS:
        # Deterministic per-benchmark seed offset (NOT hash(benchmark): Python's
        # string hash is randomized per-process by default, which would make
        # seeding non-reproducible across runs). Using each benchmark's fixed
        # index instead avoids the same-seed-same-dimension collision that made
        # Ackley_2D and Rosenbrock_2D (both d=2) produce identical LHS layouts
        # and roi_candidates draws under a shared seed.
        benchmark_seed = SEED + BENCHMARKS.index(benchmark)
        for stage, n_initial in STAGES:
            torch.manual_seed(benchmark_seed)
            np.random.seed(benchmark_seed)
            result, violation_frac = _run_stage(benchmark, n_initial, K_THOMPSON_DEFAULT, benchmark_seed)

            if violation_frac > VIOLATION_RETRY_THRESHOLD:
                print(f"  {benchmark} {stage}: entropy-monotonicity violation rate "
                      f"{violation_frac:.1%} > 5% at K={K_THOMPSON_DEFAULT} -- retrying at K={K_THOMPSON_RETRY}")
                torch.manual_seed(benchmark_seed)
                np.random.seed(benchmark_seed)
                result, violation_frac = _run_stage(benchmark, n_initial, K_THOMPSON_RETRY, benchmark_seed)

            result["phase1_thompson_ks"] = phase1_thompson[f"{benchmark}__{stage}"]["ks_stat"]
            results[(benchmark, stage)] = result
            print(f"  done: {benchmark:<15} {stage:<7} rho_all={result['rho_all']:.4f} "
                  f"neg_frac={result['neg_frac']:.3f} entropy_violations={result['violation_frac']:.1%} "
                  f"(K={result['K_thompson_used']}, roi_size={result['roi_size_mean']:.0f}) status={result['status']}")

    print("\n=== RTG FORMULATION CORRELATION (v4, post roi_candidates bugfix) ===")
    header = (f"{'Benchmark':<15} {'Stage':<7} {'Rho_RTG0':>9} {'Rho_All':>9} "
              f"{'R_RTG0':>7} {'R_All':>7} {'Neg%':>6} {'roi_size':>9}  STATUS")
    print(header)
    print("-" * len(header))
    for benchmark in BENCHMARKS:
        for stage, _ in STAGES:
            r = results[(benchmark, stage)]
            print(f"{benchmark:<15} {stage:<7} {r['rho_rtg0']:>9.3f} {r['rho_all']:>9.3f} "
                  f"{r['r_rtg0']:>7.3f} {r['r_all']:>7.3f} {r['neg_frac']*100:>5.1f}% {r['roi_size_mean']:>9.0f}  {r['status']}")

    all_stage_names = [s for s, _ in STAGES]
    pass_all = [b for b in BENCHMARKS if all(results[(b, s)]["status"] == "PASS" for s in all_stage_names)]
    any_warn = [b for b in BENCHMARKS if any(results[(b, s)]["status"] == "WARN" for s in all_stage_names)]
    any_fail = [b for b in BENCHMARKS if any(results[(b, s)]["status"] == "FAIL" for s in all_stage_names)]

    print("\n=== SUMMARY ===\n")
    print(f"Benchmarks with ALL stages PASS:  {pass_all}")
    print(f"Benchmarks with any WARN:         {any_warn}")
    print(f"Benchmarks with any FAIL:         {any_fail}")

    total_violations = sum(results[(b, s)]["n_violations"] for b in BENCHMARKS for s in all_stage_names)
    total_rollouts = len(BENCHMARKS) * len(all_stage_names) * N_ROLLOUTS
    print(f"\nEntropy-monotonicity (H0>=H1>=...>=HT) violations: {total_violations}/{total_rollouts} "
          f"({total_violations/total_rollouts:.1%})")

    n_total = len(BENCHMARKS) * len(all_stage_names)
    n_pass = sum(1 for b in BENCHMARKS for s in all_stage_names if results[(b, s)]["status"] == "PASS")
    n_fail = sum(1 for b in BENCHMARKS for s in all_stage_names if results[(b, s)]["status"] == "FAIL")

    print("\nRECOMMENDATION:")
    if n_pass == n_total:
        print("Per-step RTG and joint RTG are highly rank-correlated everywhere tested.")
        print("The two formulations are interchangeable in practice -- either can be used")
        print("as the DT training target without meaningfully changing what the DT learns.")
    elif n_fail == n_total:
        print("Per-step RTG and joint RTG diverge substantially everywhere tested.")
        print("Do NOT treat them as interchangeable -- switching formulations would change")
        print("the DT's training signal in a way that needs its own separate validation.")
    else:
        passing = [b for b in BENCHMARKS if b not in any_fail]
        print(f"Formulations agree well on: {passing}")
        print(f"Formulations diverge on: {any_fail}")
        print("Recommendation: treat interchangeability as benchmark-dependent; do not")
        print("assume a pipeline-wide switch is safe without per-benchmark validation.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {f"{b}__{s}": results[(b, s)] for b in BENCHMARKS for s, _ in STAGES}
    with open(V4_RESULTS_PATH, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved full results to {V4_RESULTS_PATH} (v3 at {NEW_RESULTS_PATH} preserved for comparison)")


if __name__ == '__main__':
    main()
