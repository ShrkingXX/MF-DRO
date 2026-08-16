# mes_switching_v2 cluster experiment

200-iteration, 3-benchmark (Ackley_2D/5D/10D), 8-variant, 5-seed (42-46)
comparison of acquisition-function source (rotate vs MES) x reward type
(improvement vs MES) x DT-vs-no-DT (NaiveBO baselines), plus 2 jointMES
variants (entropy_joint RTG) -- DRO-rotate-jointMES is untested at cluster
scale (included by explicit choice despite thin evidence); DRO-MES-jointMES
passed a small targeted validation. See run_experiment.py's EXP9 comment
block for the full rationale. 120 total runs (8 variants x 3 benchmarks x 5
seeds).

Variant matrix, checkpoint format, and cost/behavioral rationale are all
documented in `run_experiment.py`'s EXP9 block (`EXP9_NAME =
"mes_switching_v2_cluster"`) -- read that first.

## Why the "m" series, not a GPU node

`dro_runner.py` hardcodes `"device": "cpu"` -- nothing in the GP (BoTorch/
GPyTorch) or Decision Transformer pipeline touches CUDA. A GPU node would sit
idle for this workload. The "m" series (3 nodes, 128 CPU threads, 1.5TB RAM,
no GPU) gives the most CPU parallelism with zero GPU contention, and matches
`run_experiment_parallel.py`'s existing process-pool-plus-thread-cap design.

## Before running on the cluster

1. **Sanity check locally first** (already done once on this machine, but
   re-run after any code change): `python sanity_check_mes_switching_v2.py`
   -- crash smoke test, regret monotonicity, rollout-action-diversity
   direction check (rotate vs a fixed-EI reference), MES acquisition
   non-degeneracy, and a NaiveBO-EI regression check against Experiment 1's
   existing `results/mes_reward` data (same seeds 42-46).
2. Run the same script's `check3_timing_calibration()` (or just run the
   whole script) to get real per-phase wall-clock numbers -- these came back
   from a local Mac, not the actual cluster hardware, so treat them as a
   rough guide, not gospel; re-run the calibration on an actual "m" node once
   you have access, to size `WORKERS`/`THREADS_PER_WORKER` and confirm the
   `CHAIN_LENGTH`/walltime-per-job assumptions in `submit_chain.sh`.
3. Follow `submit_chain.sh`'s header for the one-time setup: rsync the repo,
   build a Linux venv from `requirements.txt` (the local `.venv` is
   macOS-built and won't run on the cluster), find your real partition name
   and the 3 "m"-series node hostnames via `cs-sinfo`.

## Running it

```bash
cd cluster
./submit_chain.sh
```

Submits 3 independent chains (one per benchmark, one per node), each
`CHAIN_LENGTH` sequential ~4-hour jobs linked by `--dependency=afterany`.
Each job in a chain automatically continues from where the previous one left
off or was killed -- no manual resume step, no self-resubmission logic
needed (see `run_shard.sbatch`'s header for why: every real BO iteration is
checkpointed atomically before the next one starts, and `run_single_seed`
transparently resumes from the latest checkpoint on its own).

Check queue status: `cs-squeue`

## Checking progress / partial results (e.g. before a meeting)

Works at ANY point, complete or not -- reads directly from the live
per-iteration log files, not the final `.json` checkpoints:

```bash
python build_partial_regret_plot.py --exp-name mes_switching_v2_cluster
```

Produces one PNG per benchmark in `results/mes_switching_v2_cluster/plots/`,
with mean +/- SE bands computed from however many seeds have reached each
iteration so far (different seeds may be at different points if some got
interrupted/resumed more than others).

For the raw numbers instead of a plot:
`results/mes_switching_v2_cluster/logs/{benchmark}__{variant}__seed{seed}.log`
-- one human-readable `key=value` line per completed real iteration,
including the per-phase wall-clock fields (`gp_refit_time`,
`rollout_sim_time`, `dt_train_time`, `real_query_time`) and
`rollout_action_diversity` for the cost/mechanism audit.

## Final analysis (once every run is complete)

Use `checkpoint.load_all_results("mes_switching_v2_cluster", benchmarks,
variants, seeds)` against the completed `.json` checkpoints (only present
once a run's `.done` flag is written) -- these are the authoritative final
per-iteration records; the partial-log-based plot above is for
in-progress visibility only.
