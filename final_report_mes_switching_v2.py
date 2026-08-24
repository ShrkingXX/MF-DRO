"""
Final analysis for the mes_switching_v2 cluster experiment (or any experiment
sharing its result-dict shape, e.g. entropy_joint_ackley5d -- used here as a
test harness since real cluster data doesn't exist yet). Reads COMPLETED
checkpoints only (checkpoint.load_all_results, gated on the .done flag) --
this is the authoritative final analysis, distinct from
build_partial_regret_plot.py's live-log-based in-progress view.

Produces, per benchmark:
  1. Final regret-vs-iteration plot, mean +/- SE across seeds, all variants
     overlaid (the deliverable requested at the start of this planning: "the
     final plot to be the simple regret change proceeding with the iteration
     number for all methods each benchmark").
  2. Best-observed-value plot, same aggregation, on the raw objective scale
     (flips best_observed's internal negated-for-maximize sign back) with the
     known optimum drawn as a reference line -- complements regret_curve for
     anyone who wants the actual objective trajectory, not just the gap.
  3. Cost audit: per-phase wall-clock breakdown (gp_refit/rollout_sim/
     dt_train/real_query) per variant, both as a table and a stacked bar
     chart -- "audit the computation cost and the wallclock cost for all
     variants and compare them in the report".
  4. Regret-vs-cost plot: regret against cumulative wall-clock cost (log-x)
     instead of iteration count -- the fair comparison given DRO variants
     cost ~100-500x more per iteration than NaiveBO.
  5. Mechanism diagnostics: rollout_action_diversity trend (rotate vs fixed
     acquisition, over the FULL run -- not just the early iterations the
     local sanity check covered), neg_rtg_frac trend (jointMES variants),
     reward sparsity (zero_frac -- Impr vs MES reward), corner_proximity/
     mu_proposed trend (checking for any Hartmann-style GP-confidence
     collapse).

Every field is read via .get(...) with a None fallback: older checkpoints
(e.g. entropy_joint_ackley5d's original DRO-PerStep/DRO-EntropyJoint runs,
from before the per-phase timing / rollout_action_diversity diagnostics were
added) simply show up as "N/A" in the relevant table rather than crashing --
verified against exactly that mixed-vintage data before relying on this for
the real cluster results.

Usage:
    python final_report_mes_switching_v2.py --exp-name mes_switching_v2_cluster \\
        --benchmarks Ackley_2D Ackley_5D Ackley_10D \\
        --variants DRO-rotate-Impr DRO-rotate-MES DRO-MES-Impr DRO-MES-MES \\
                   NaiveBO-EI NaiveBO-MES DRO-rotate-jointMES DRO-MES-jointMES \\
        --seeds 42 43 44 45 46

    # Or, to use exactly EXP9's own definition from run_experiment.py:
    python final_report_mes_switching_v2.py --use-exp9

    # Test harness (already-completed data, different experiment):
    python final_report_mes_switching_v2.py --exp-name entropy_joint_ackley5d \\
        --benchmarks Ackley_5D \\
        --variants DRO-PerStep DRO-EntropyJoint DRO-EntropyJoint-L4 DRO-MES-jointMES \\
        --seeds 42 43 44
"""
import argparse
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import BENCHMARKS
from checkpoint import load_all_results

PHASE_FIELDS = ["gp_refit_time", "rollout_sim_time", "dt_train_time", "real_query_time"]
PHASE_LABELS = {"gp_refit_time": "GP refit", "rollout_sim_time": "Rollout sim",
                "dt_train_time": "DT train", "real_query_time": "Real query"}


def _mean_se(values):
    # NaN, not just None, marks "not applicable" here: naive_bo.py deliberately
    # fills DRO-only rollout diagnostics (mean_reward, zero_frac, ...) with
    # float('nan') per iteration for NaiveBO runs, since those concepts don't
    # apply to a rollout-free baseline -- distinct from None, which means an
    # older checkpoint predates the field entirely.
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    mean = st.mean(vals)
    se = (st.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return mean, se


def _fmt(x, suffix="", nd=3):
    return "N/A" if x is None else f"{x:.{nd}f}{suffix}"


def _jitter_offsets(n, spread=0.12):
    """n deterministic, evenly-spaced x-offsets centered on 0 -- avoids
    overplotting individual seed points at the same box-plot x position
    without pulling in a random-jitter dependency (or non-determinism)."""
    if n <= 1:
        return [0.0] * n
    step = 2 * spread / (n - 1)
    return [-spread + i * step for i in range(n)]


def load_data(exp_name, benchmarks, variants, seeds):
    results = load_all_results(exp_name, benchmarks, variants, seeds)
    missing = [(b, v, s) for b in benchmarks for v in variants for s in seeds
               if (b, v, s) not in results]
    if missing:
        print(f"WARNING: {len(missing)}/{len(benchmarks)*len(variants)*len(seeds)} "
              f"(benchmark, variant, seed) combos are not completed yet -- "
              f"analysis below uses only what's available. "
              f"First few missing: {missing[:5]}")
    return results


_LOG_FLOOR = 1e-6  # visual-only clip so exact/near-zero converged regret doesn't vanish or warn on a log axis


def plot_regret_curves(results, benchmarks, variants, seeds, out_dir, log_scale=False, filename_suffix=""):
    """log_scale=True produces a semilogy version -- the near-zero tail where
    all DRO variants otherwise visually collapse into one indistinguishable
    blob on a linear axis is exactly where a 100x difference (e.g. 0.3 vs
    0.003 final regret) actually lives; log-y is the standard way BO papers
    show that instead of letting it get compressed to invisibility.
    filename_suffix distinguishes output files when called again with a
    different variant subset (e.g. DRO-only, excluding NaiveBO) so it
    doesn't overwrite the full-variant version.
    """
    os.makedirs(out_dir, exist_ok=True)
    suffix = ("_log" if log_scale else "") + filename_suffix
    for benchmark in benchmarks:
        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False
        for variant in variants:
            curves = [results[(benchmark, variant, s)]["regret_curve"]
                      for s in seeds if (benchmark, variant, s) in results]
            if not curves:
                continue
            any_data = True
            n_iters = min(len(c) for c in curves)  # in case of ragged lengths (partial completion)
            means, los, his = [], [], []
            for i in range(n_iters):
                vals = [c[i] for c in curves]
                m = st.mean(vals)
                se = (st.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
                means.append(max(m, _LOG_FLOOR) if log_scale else m)
                los.append(max(m - se, _LOG_FLOOR) if log_scale else m - se)
                his.append(m + se)
            xs = list(range(1, n_iters + 1))
            ax.plot(xs, means, label=f"{variant} (n={len(curves)})", linewidth=1.8)
            ax.fill_between(xs, los, his, alpha=0.15)

        if not any_data:
            plt.close(fig)
            print(f"  {benchmark}: no completed data, skipping regret plot")
            continue

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Real BO iteration")
        ax.set_ylabel("Simple regret")
        ax.set_title(f"{benchmark} -- simple regret vs iteration (final){' [log scale]' if log_scale else ''}")
        ax.legend(fontsize=8, loc="upper right" if not log_scale else "lower left")
        ax.grid(alpha=0.3, which="both" if log_scale else "major")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_regret_final{suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def plot_best_observed_curves(results, benchmarks, variants, seeds, out_dir):
    """Raw-scale best-observed-value plot, complementing plot_regret_curves.

    Checkpoints store best_observed on the internal "maximize" scale (negated
    Ackley, per benchmarks.py's negate=True construction), so it's flipped
    back here (raw = -best_observed) to plot in the same units as
    known_optimal_value -- more directly interpretable than regret for anyone
    who wants to see the actual objective trajectory rather than the gap to
    optimum, or where the optimum value itself is worth showing as a line.
    """
    os.makedirs(out_dir, exist_ok=True)
    for benchmark in benchmarks:
        known_optimal = BENCHMARKS[benchmark]["known_optimal_value"]
        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False
        for variant in variants:
            curves = [[-v for v in results[(benchmark, variant, s)]["best_observed"]]
                      for s in seeds if (benchmark, variant, s) in results]
            if not curves:
                continue
            any_data = True
            n_iters = min(len(c) for c in curves)  # in case of ragged lengths (partial completion)
            means, los, his = [], [], []
            for i in range(n_iters):
                vals = [c[i] for c in curves]
                m = st.mean(vals)
                se = (st.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
                means.append(m)
                los.append(m - se)
                his.append(m + se)
            xs = list(range(1, n_iters + 1))
            ax.plot(xs, means, label=f"{variant} (n={len(curves)})", linewidth=1.8)
            ax.fill_between(xs, los, his, alpha=0.15)

        if not any_data:
            plt.close(fig)
            print(f"  {benchmark}: no completed data, skipping best-observed plot")
            continue

        ax.axhline(known_optimal, color="black", linestyle="--", linewidth=1.2,
                   label=f"known optimum ({known_optimal:g})")
        ax.set_xlabel("Real BO iteration")
        ax.set_ylabel("Best observed value (raw scale)")
        ax.set_title(f"{benchmark} -- best observed value vs iteration (final)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_best_observed_final.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def cost_audit(results, benchmarks, variants, seeds, out_dir):
    print("\n" + "=" * 100)
    print("COST AUDIT: mean per-iteration wall-clock, by phase (aggregated across all seeds/benchmarks)")
    print("=" * 100)
    header = f"{'variant':22s}" + "".join(f"{PHASE_LABELS[f]:>14s}" for f in PHASE_FIELDS) + f"{'total/iter':>14s}{'total/run':>14s}"
    print(header)

    variant_phase_means = {}
    for variant in variants:
        all_iter_times = []
        phase_means = {}
        for f in PHASE_FIELDS:
            vals = [t for b in benchmarks for s in seeds if (b, variant, s) in results
                    for t in results[(b, variant, s)].get(f, []) if t is not None]
            phase_means[f] = st.mean(vals) if vals else None
        for b in benchmarks:
            for s in seeds:
                if (b, variant, s) in results:
                    all_iter_times.extend(results[(b, variant, s)].get("iter_times", []))
        mean_total_iter, _ = _mean_se(all_iter_times)
        n_iters_per_run = st.mean([len(results[(b, variant, s)]["iter_times"])
                                    for b in benchmarks for s in seeds if (b, variant, s) in results]) \
            if any((b, variant, s) in results for b in benchmarks for s in seeds) else None
        total_per_run = mean_total_iter * n_iters_per_run if (mean_total_iter and n_iters_per_run) else None

        variant_phase_means[variant] = phase_means
        row = f"{variant:22s}" + "".join(f"{_fmt(phase_means[f], 's'):>14s}" for f in PHASE_FIELDS)
        row += f"{_fmt(mean_total_iter, 's'):>14s}{_fmt(total_per_run, 's', nd=0):>14s}"
        print(row)

    # Stacked bar chart, only for variants with at least one phase field present.
    plot_variants = [v for v in variants if any(m is not None for m in variant_phase_means[v].values())]
    if plot_variants:
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(8, len(plot_variants) * 1.2), 6))
        bottoms = [0.0] * len(plot_variants)
        for f in PHASE_FIELDS:
            vals = [variant_phase_means[v][f] or 0.0 for v in plot_variants]
            ax.bar(plot_variants, vals, bottom=bottoms, label=PHASE_LABELS[f])
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_ylabel("Mean wall-clock per iteration (s)")
        ax.set_title("Per-phase cost breakdown by variant")
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        out_path = os.path.join(out_dir, "cost_audit_breakdown.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n  saved {out_path}")
    else:
        print("\n  (no variants have per-phase timing data -- skipping cost breakdown chart;"
              " this is expected for checkpoints from before the per-phase timing diagnostics were added)")


def plot_regret_vs_cost(results, benchmarks, variants, seeds, out_dir):
    """Direct cost-efficiency scatter: one point per variant at (mean
    per-iteration wall-clock cost, final regret) -- replaces an earlier
    cumulative-cost-vs-iteration curve, which required mentally integrating
    cost-per-iteration against elapsed iterations to read off an efficiency
    ranking. Per-iteration cost spans ~3 orders of magnitude (NaiveBO's
    ~0.1-0.7s/iter vs DRO's ~150-320s/iter) and final regret spans a similar
    range (~0.005 to ~19 across benchmarks/variants), so both axes are log --
    a variant strictly down-and-left of another is strictly more efficient,
    readable at a glance instead of requiring curve-tracing.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Cost-clustered variants (e.g. all 6 DRO ones sit within a ~2x band on
    # x) would otherwise print labels on top of each other -- fan them across
    # a few vertical tiers instead of a single fixed offset.
    label_tiers = [(9, 8), (9, -6), (9, 20), (9, -18), (9, 32), (9, -30), (9, 44), (9, -42)]
    for benchmark in benchmarks:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        any_data = False
        for idx, variant in enumerate(variants):
            iter_times = [t for s in seeds if (benchmark, variant, s) in results
                          for t in results[(benchmark, variant, s)].get("iter_times", [])]
            finals = [results[(benchmark, variant, s)]["regret_curve"][-1]
                      for s in seeds if (benchmark, variant, s) in results
                      if results[(benchmark, variant, s)].get("regret_curve")]
            if not iter_times or not finals:
                continue
            any_data = True
            mean_cost, _ = _mean_se(iter_times)
            mean_regret, se_regret = _mean_se(finals)
            y = max(mean_regret, _LOG_FLOOR)
            yerr_lo = y - max(mean_regret - se_regret, _LOG_FLOOR)
            points = ax.errorbar([mean_cost], [y], yerr=[[yerr_lo], [se_regret]],
                                  fmt="o", markersize=9, capsize=4)
            point_color = points.lines[0].get_color()
            ax.annotate(variant, (mean_cost, y), textcoords="offset points",
                        xytext=label_tiers[idx % len(label_tiers)], fontsize=8,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=point_color, alpha=0.7))

        if not any_data:
            plt.close(fig)
            print(f"  {benchmark}: no completed data, skipping regret-vs-cost plot")
            continue

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean wall-clock cost per iteration (s)")
        ax.set_ylabel("Final simple regret (last iteration)")
        ax.set_title(f"{benchmark} -- final regret vs per-iteration cost")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_regret_vs_cost_final.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def plot_cost_weighted_regret_curves(results, benchmarks, variants, seeds, out_dir):
    """Combined performance-efficiency score plotted over iterations:
    regret(t) * cumulative_cost(t), lower is better throughout, penalizing a
    variant for being slow to converge AND for costing a lot to get there.
    NaiveBO is deliberately excluded by the caller: its near-zero cost would
    make the product tiny regardless of its (much worse) regret, which would
    make the metric misleadingly favor it rather than reflect the DRO-only
    efficiency question this was built to answer.

    Caveat worth stating plainly: "regret times seconds" has no natural
    physical unit -- it's one specific, arbitrary way to combine two
    differently-scaled quantities into one number, and whichever of the two
    happens to dominate numerically will dominate the product. Treat this as
    a supplementary combined-score view, not a replacement for the more
    standard cost_to_target / paired-difference comparisons elsewhere in
    this report.
    """
    os.makedirs(out_dir, exist_ok=True)
    for benchmark in benchmarks:
        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False
        for variant in variants:
            runs = [(results[(benchmark, variant, s)]["regret_curve"],
                     results[(benchmark, variant, s)]["iter_times"])
                    for s in seeds if (benchmark, variant, s) in results]
            runs = [(rc, it) for rc, it in runs if rc and it]
            if not runs:
                continue
            any_data = True
            n_iters = min(min(len(rc), len(it)) for rc, it in runs)
            scores = []
            for i in range(n_iters):
                mean_regret = st.mean(rc[i] for rc, _ in runs)
                mean_cum_cost = st.mean(sum(it[:i + 1]) for _, it in runs)
                scores.append(max(mean_regret * mean_cum_cost, _LOG_FLOOR))
            xs = list(range(1, n_iters + 1))
            ax.plot(xs, scores, label=f"{variant} (n={len(runs)})", linewidth=1.8)

        if not any_data:
            plt.close(fig)
            print(f"  {benchmark}: no completed data, skipping cost-weighted regret plot")
            continue

        ax.set_yscale("log")
        ax.set_xlabel("Real BO iteration")
        ax.set_ylabel("Regret x cumulative cost (regret . seconds)")
        ax.set_title(f"{benchmark} -- cost-weighted regret (regret x cumulative wall-clock cost)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_cost_weighted_regret.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def plot_paired_seed_differences(results, benchmarks, variants, seeds, out_dir,
                                  reference_variant="DRO-rotate-jointMES"):
    """Seed-matched final-regret differences (variant - reference), per
    benchmark. All variants share the same 5 seeds, so pairing controls for
    seed-specific difficulty and is a much more sensitive test than comparing
    independent means -- directly answers whether the apparent ranking from
    the mean+/-SE plots is real or just seed noise. Points above the zero
    line mean the reference variant had LOWER (better) regret for that seed;
    below the line means the other variant won that seed.
    """
    if reference_variant not in variants:
        print(f"  reference variant {reference_variant!r} not in variants list, skipping paired-diff plot")
        return
    os.makedirs(out_dir, exist_ok=True)
    other_variants = [v for v in variants if v != reference_variant]
    for benchmark in benchmarks:
        ref_finals = {s: results[(benchmark, reference_variant, s)]["regret_curve"][-1]
                      for s in seeds if (benchmark, reference_variant, s) in results
                      if results[(benchmark, reference_variant, s)].get("regret_curve")}
        if not ref_finals:
            print(f"  {benchmark}: no {reference_variant} data, skipping paired-diff plot")
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False
        for i, variant in enumerate(other_variants, start=1):
            diffs = [results[(benchmark, variant, s)]["regret_curve"][-1] - ref_finals[s]
                     for s in seeds if s in ref_finals and (benchmark, variant, s) in results
                     and results[(benchmark, variant, s)].get("regret_curve")]
            if not diffs:
                continue
            any_data = True
            offsets = _jitter_offsets(len(diffs))
            ax.scatter([i + o for o in offsets], diffs, color="#4c72b0", s=30, zorder=3, alpha=0.8)
            mean_d, se_d = _mean_se(diffs)
            ax.errorbar([i], [mean_d], yerr=[se_d], fmt="D", color="black",
                        markersize=7, capsize=4, zorder=4)

        if not any_data:
            plt.close(fig)
            print(f"  {benchmark}: no completed data, skipping paired-diff plot")
            continue

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(range(1, len(other_variants) + 1))
        ax.set_xticklabels(other_variants, rotation=30, ha="right")
        ax.set_ylabel(f"Final regret difference\n(variant - {reference_variant})")
        ax.set_title(f"{benchmark} -- seed-paired regret difference vs {reference_variant}\n"
                     f"(above 0 = {reference_variant} won that seed)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_paired_diff_vs_ref.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def plot_iters_to_target(results, benchmarks, variants, seeds, out_dir, reduction_frac=0.90):
    """Iterations needed to cut regret to (1 - reduction_frac) of its own
    iteration-1 value, per seed -- a convergence-speed metric relative to
    each run's own starting regret, so it's comparable across benchmarks
    whose absolute regret scales differ by orders of magnitude (2D's ~0.01
    final regret vs 10D's ~0.2, for instance), unlike a fixed absolute
    threshold. More decision-relevant than "final regret at iteration 200"
    alone: shows which variant gets to a good answer FASTEST, not just which
    wins at a fixed, arbitrary budget. Seeds that never reach the target
    within the run are excluded from the box (censored) and counted in the
    x-tick label instead of silently dropped.
    """
    os.makedirs(out_dir, exist_ok=True)
    for benchmark in benchmarks:
        data, labels = [], []
        for variant in variants:
            hits, n_censored = [], 0
            for s in seeds:
                rc = results.get((benchmark, variant, s), {}).get("regret_curve")
                if not rc:
                    continue
                target = rc[0] * (1 - reduction_frac)
                hit = next((i + 1 for i, v in enumerate(rc) if v <= target), None)
                if hit is None:
                    n_censored += 1
                else:
                    hits.append(hit)
            if hits or n_censored:
                data.append(hits)
                label = variant if n_censored == 0 else f"{variant}\n({n_censored} never reached)"
                labels.append(label)

        plot_pairs = [(l, d) for l, d in zip(labels, data) if d]
        if not plot_pairs:
            print(f"  {benchmark}: no completed data, skipping iters-to-target plot")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_labels = [l for l, _ in plot_pairs]
        plot_data = [d for _, d in plot_pairs]
        bp = ax.boxplot(plot_data, labels=plot_labels, showmeans=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#a8c6e8")
            patch.set_alpha(0.6)
        for i, vals in enumerate(plot_data, start=1):
            offsets = _jitter_offsets(len(vals))
            ax.scatter([i + o for o in offsets], vals, color="black", s=20, zorder=3, alpha=0.75)

        ax.set_ylabel(f"Iterations to reach {int(reduction_frac * 100)}% regret reduction")
        ax.set_title(f"{benchmark} -- convergence speed ({int(reduction_frac * 100)}% regret reduction)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_iters_to_target.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def plot_cost_to_target(results, benchmarks, variants, seeds, out_dir, reduction_frac=0.90,
                         highlight_variants=("DRO-rotate-jointMES", "DRO-MES-jointMES")):
    """Cumulative wall-clock cost needed to cut regret to (1 - reduction_frac)
    of its own iteration-1 value, per seed -- the direct answer to "does
    jointMES reach the same performance level for less compute". Unlike
    plot_regret_vs_cost (final regret at whatever cost was actually spent
    over the full run) or plot_iters_to_target (iteration count, blind to
    per-iteration cost), this measures cost spent to reach a FIXED
    performance bar -- the metric that actually demonstrates a cost
    advantage rather than just a cheaper per-iteration price tag that could
    in principle be offset by needing more iterations. jointMES variants are
    colored green so the advantage reads immediately without label
    cross-referencing.
    """
    os.makedirs(out_dir, exist_ok=True)
    for benchmark in benchmarks:
        rows = []  # (variant, label, cost_values)
        for variant in variants:
            hits, n_censored = [], 0
            for s in seeds:
                r = results.get((benchmark, variant, s))
                if not r:
                    continue
                rc, it = r.get("regret_curve"), r.get("iter_times")
                if not rc or not it:
                    continue
                target = rc[0] * (1 - reduction_frac)
                hit_idx = next((i for i, v in enumerate(rc) if v <= target), None)
                if hit_idx is None:
                    n_censored += 1
                else:
                    hits.append(sum(it[:hit_idx + 1]))
            if hits or n_censored:
                label = variant if n_censored == 0 else f"{variant}\n({n_censored} never reached)"
                rows.append((variant, label, hits))

        plot_rows = [(v, l, d) for v, l, d in rows if d]
        if not plot_rows:
            print(f"  {benchmark}: no completed data, skipping cost-to-target plot")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_labels = [l for _, l, _ in plot_rows]
        plot_data = [d for _, _, d in plot_rows]
        bp = ax.boxplot(plot_data, labels=plot_labels, showmeans=True, patch_artist=True)
        for (variant, _, _), patch in zip(plot_rows, bp["boxes"]):
            is_jointmes = variant in highlight_variants
            patch.set_facecolor("#2ca02c" if is_jointmes else "#a8c6e8")
            patch.set_alpha(0.65 if is_jointmes else 0.5)
        for i, (_, _, vals) in enumerate(plot_rows, start=1):
            offsets = _jitter_offsets(len(vals))
            ax.scatter([i + o for o in offsets], vals, color="black", s=20, zorder=3, alpha=0.75)

        ax.set_yscale("log")
        ax.set_ylabel(f"Wall-clock cost to reach {int(reduction_frac * 100)}% regret reduction (s)")
        ax.set_title(f"{benchmark} -- cost to reach target performance (green = jointMES)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.3, which="both", axis="y")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{benchmark}_cost_to_target.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {benchmark}: saved {out_path}")


def mechanism_diagnostics(results, benchmarks, variants, seeds):
    print("\n" + "=" * 100)
    print("MECHANISM DIAGNOSTICS")
    print("=" * 100)

    # Rollout action diversity: mean over the FULL run (not just early iterations).
    print("\n-- rollout_action_diversity (mean over full run, aggregated across benchmarks/seeds) --")
    for variant in variants:
        vals = [d for b in benchmarks for s in seeds if (b, variant, s) in results
                for d in results[(b, variant, s)].get("rollout_action_diversity", []) if d is not None]
        mean, se = _mean_se(vals)
        print(f"  {variant:22s} mean={_fmt(mean)}  se={_fmt(se)}  (n_iters={len(vals)})")

    # neg_rtg_frac: only meaningful for jointMES / entropy_joint variants.
    print("\n-- neg_rtg_frac (mean over full run -- jointMES/entropy_joint variants only, others show N/A) --")
    for variant in variants:
        vals = [d for b in benchmarks for s in seeds if (b, variant, s) in results
                for d in results[(b, variant, s)].get("neg_rtg_frac", []) if d is not None]
        mean, se = _mean_se(vals)
        print(f"  {variant:22s} mean={_fmt(mean)}  se={_fmt(se)}  (n_iters={len(vals)})")

    # Reward sparsity: zero_frac, Impr vs MES reward comparison.
    print("\n-- zero_frac (reward sparsity -- Impr reward should be higher than MES reward's) --")
    for variant in variants:
        vals = [d for b in benchmarks for s in seeds if (b, variant, s) in results
                for d in results[(b, variant, s)].get("zero_frac", []) if d is not None]
        mean, se = _mean_se(vals)
        print(f"  {variant:22s} mean={_fmt(mean)}  se={_fmt(se)}  (n_iters={len(vals)})")

    # corner_proximity / mu_proposed: check for any Hartmann-style GP-confidence collapse.
    print("\n-- corner_proximity, mu_proposed (mean over full run) --")
    for variant in variants:
        cp_vals = [d for b in benchmarks for s in seeds if (b, variant, s) in results
                   for d in results[(b, variant, s)].get("corner_proximity", []) if d is not None]
        mu_vals = [d for b in benchmarks for s in seeds if (b, variant, s) in results
                   for d in results[(b, variant, s)].get("mu_proposed", []) if d is not None]
        cp_mean, _ = _mean_se(cp_vals)
        mu_mean, _ = _mean_se(mu_vals)
        print(f"  {variant:22s} corner_proximity={_fmt(cp_mean)}  mu_proposed={_fmt(mu_mean)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--use-exp9", action="store_true",
                         help="Use run_experiment.py's EXP9 (mes_switching_v2_cluster) definition directly.")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.use_exp9:
        from run_experiment import EXP9_NAME, EXP9_BENCHMARKS, EXP9_VARIANTS, EXP9_SEEDS
        exp_name, benchmarks, variants, seeds = EXP9_NAME, EXP9_BENCHMARKS, EXP9_VARIANTS, EXP9_SEEDS
    else:
        if not (args.exp_name and args.benchmarks and args.variants and args.seeds):
            parser.error("Either --use-exp9, or all of --exp-name/--benchmarks/--variants/--seeds")
        exp_name, benchmarks, variants, seeds = args.exp_name, args.benchmarks, args.variants, args.seeds

    out_dir = args.out_dir or os.path.join("results", exp_name, "plots")

    print(f"Loading {exp_name}: benchmarks={benchmarks}  variants={variants}  seeds={seeds}")
    results = load_data(exp_name, benchmarks, variants, seeds)
    print(f"Loaded {len(results)}/{len(benchmarks)*len(variants)*len(seeds)} completed (benchmark, variant, seed) runs.\n")

    dro_variants = [v for v in variants if not v.startswith("NaiveBO")]

    print("=" * 100)
    print("REGRET CURVES")
    print("=" * 100)
    plot_regret_curves(results, benchmarks, variants, seeds, out_dir)
    plot_regret_curves(results, benchmarks, variants, seeds, out_dir, log_scale=True)

    print("\n" + "=" * 100)
    print("REGRET CURVES (DRO variants only, no NaiveBO baseline)")
    print("=" * 100)
    plot_regret_curves(results, benchmarks, dro_variants, seeds, out_dir, filename_suffix="_dro_only")
    plot_regret_curves(results, benchmarks, dro_variants, seeds, out_dir, log_scale=True, filename_suffix="_dro_only")

    print("\n" + "=" * 100)
    print("BEST OBSERVED VALUE CURVES (raw scale)")
    print("=" * 100)
    plot_best_observed_curves(results, benchmarks, variants, seeds, out_dir)

    cost_audit(results, benchmarks, variants, seeds, out_dir)

    print("\n" + "=" * 100)
    print("REGRET VS WALL-CLOCK COST (DRO variants only -- NaiveBO's ~1000x cheaper/worse points")
    print("otherwise force a log-log range that squashes the DRO cluster into one corner)")
    print("=" * 100)
    plot_regret_vs_cost(results, benchmarks, dro_variants, seeds, out_dir)

    print("\n" + "=" * 100)
    print("COST-WEIGHTED REGRET CURVES (regret x cumulative cost, DRO variants only)")
    print("=" * 100)
    plot_cost_weighted_regret_curves(results, benchmarks, dro_variants, seeds, out_dir)

    print("\n" + "=" * 100)
    print("SEED-PAIRED REGRET DIFFERENCES (DRO variants only)")
    print("=" * 100)
    plot_paired_seed_differences(results, benchmarks, dro_variants, seeds, out_dir)

    print("\n" + "=" * 100)
    print("CONVERGENCE SPEED: ITERATIONS TO TARGET (DRO variants only)")
    print("=" * 100)
    plot_iters_to_target(results, benchmarks, dro_variants, seeds, out_dir)

    print("\n" + "=" * 100)
    print("CONVERGENCE SPEED: COST TO TARGET (DRO variants only, jointMES highlighted)")
    print("=" * 100)
    plot_cost_to_target(results, benchmarks, dro_variants, seeds, out_dir)

    mechanism_diagnostics(results, benchmarks, variants, seeds)


if __name__ == "__main__":
    main()
