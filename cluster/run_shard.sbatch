#!/bin/bash
# One "shard" job: runs run_experiment_parallel.py for the
# mes_switching_v2_cluster experiment, scoped to ONE benchmark (so 3 of
# these -- one per benchmark -- can run concurrently on 3 separate "m"-series
# nodes, each owning its own 40-run sub-grid: 8 variants x 5 seeds).
#
# Does NOT need any resume logic of its own: run_single_seed
# (dro_runner.py) already checks for and continues from an in-progress
# checkpoint automatically (checkpoint.py's .resume.pt files), and skips
# anything already completed (.done flag) -- so this exact same script,
# re-submitted as the next link in a --dependency=afterany chain, just picks
# up wherever the previous job in the chain left off. No self-resubmission,
# no SIGTERM trap needed: every real iteration is checkpointed atomically
# (temp file + os.replace) before this job even reaches its time limit, so a
# hard SIGKILL at the walltime cap loses at most one in-flight iteration.
#
# Never submitted directly -- use submit_chain.sh, which passes
# PROJECT_DIR/BENCHMARK/WORKERS/THREADS_PER_WORKER via `sbatch --export=...`
# and PARTITION/NODE via `sbatch -p/-w` (see that script's header for the
# one-time setup this requires: cs-sinfo lookup, venv, rsync target).
#
#SBATCH --job-name=mes_switch_v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64                       # was 120 -- scontrol show job showed backfill
                                                  # projecting up to ~9.7 DAYS for a near-full-node
                                                  # (120/128) request on these heavily-shared nodes.
                                                  # 64 matches the size of jobs we've repeatedly
                                                  # observed actually getting scheduled promptly here
                                                  # (e.g. two 64-cpu jobs coexisting on one node).
#SBATCH --time=03:50:00                          # 10 min under the 4h cap, safety margin
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

: "${PROJECT_DIR:?Set via sbatch --export=PROJECT_DIR=...}"
: "${BENCHMARK:?Set via sbatch --export=BENCHMARK=... (Ackley_2D/Ackley_5D/Ackley_10D)}"
WORKERS="${WORKERS:-32}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-2}"    # 32*2=64, matches --cpus-per-task above

cd "$PROJECT_DIR"

# set +u around activation: see run_sanity_check.sbatch's comment -- older
# venv activate scripts reference $PS1 with no default, fine interactively
# but unset here under set -u.
set +u
source .venv/bin/activate
set -u

# Self-healing check: fe01 (login node) and the "m" compute nodes turned out
# to resolve `python3` to genuinely different versions despite sharing the
# same NFS-mounted .venv path (confirmed: ModuleNotFoundError for a package
# that WAS installed, just under a different lib/pythonX.Y directory). If
# THIS node's python can't see the installed packages, rebuild the venv here
# rather than fail outright -- costs ~5-10 min once, only when actually
# needed, instead of paying it on every job in the chain unconditionally.
if ! python -c "import torch" 2>/dev/null; then
    echo "[$(date)] WARNING: venv on $(hostname) can't import torch (python: $(python --version 2>&1))"
    echo "[$(date)] Rebuilding venv for this node's Python..."
    deactivate 2>/dev/null || true
    rm -rf .venv
    python3 -m venv .venv
    set +u
    source .venv/bin/activate
    set -u
    pip install --quiet -r requirements.txt
    python -c "import torch" # let this crash loudly if it still fails -- something deeper is wrong
    echo "[$(date)] Rebuild OK, venv python: $(python --version 2>&1)"
fi

echo "[$(date)] Starting shard for benchmark=$BENCHMARK on $(hostname)  workers=$WORKERS threads=$THREADS_PER_WORKER"

python run_experiment_parallel.py \
    --experiment mes_switching_v2_cluster \
    --benchmarks "$BENCHMARK" \
    --workers "$WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER"

echo "[$(date)] Shard finished (or hit walltime -- check results/mes_switching_v2_cluster/checkpoints/ for .done flags to see what completed)"
