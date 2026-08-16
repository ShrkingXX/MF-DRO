#!/bin/bash
# Runs sanity_check_mes_switching_v2.py (crash/behavioral checks + timing
# calibration) on an actual "m"-series compute node -- NOT meant to be run
# on the login node (fe01/fe02), which is for light tasks only. Submitted as
# a proper background job (not an interactive srun) so it survives your SSH
# connection dropping -- the calibration piece alone took the better part of
# an hour on a contended local Mac; treat this the same way.
#
#SBATCH --job-name=sanity_check
#SBATCH --partition=yuxinchen-contrib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00                  # venv rebuild+install (~5-10 min) + checks
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# NOTE: no --nodelist here (unlike an earlier version of this script, which
# pinned m001/m002/m003 specifically). scontrol show job/node confirmed
# m001/m003 had 64/52 CPUs genuinely free, but backfill still projected a
# ~20 hour wait -- SLURM was holding that free capacity for other, higher-
# priority pending jobs (this cluster is a CPU-only workload, but that
# capacity isn't ours to jump the queue for). Dropping the node pin lets
# SLURM place this on ANY node in the partition with free room -- likely a
# GPU-equipped node (b/d/e/r/etc. series) sitting on idle CPUs, since this
# job never touches CUDA anyway. Tradeoff, by explicit request: a GPU sits
# unused wherever this lands, in exchange for actually starting soon instead
# of tomorrow afternoon. submit_chain.sh's real experiment shards still pin
# to m001/m002/m003 specifically -- that's a deliberate 3-way benchmark/node
# partitioning, not something to relax the same way for a one-off check.

set -euo pipefail

PROJECT_DIR="/net/scratch/$USER/DRO-aistats-submission"
cd "$PROJECT_DIR"

# Rebuild the venv HERE, inside the job, rather than relying on one built on
# the login node -- fe01/fe02 and the "m" compute nodes turned out to resolve
# `python3` to genuinely different versions (3.8 vs 3.10) despite sharing the
# same NFS-mounted path, so a venv built on one silently doesn't work on the
# other (packages land in .venv/lib/python3.X/site-packages for whichever X
# built it; a different-X interpreter just doesn't see them -- confirmed via
# ModuleNotFoundError: No module named 'torch' despite the venv having torch
# installed under a different lib/pythonX.Y directory). Building fresh here
# guarantees build-time and run-time are the same interpreter.
echo "[$(date)] hostname=$(hostname)"
echo "[$(date)] system python3: $(python3 --version 2>&1)"
rm -rf .venv
python3 -m venv .venv

# set +u around activation: older venv activate scripts (Python 3.8's included)
# often reference $PS1 with no default, which is fine in an interactive shell
# but unset in a non-interactive batch script -- under set -u that silently
# breaks activation before PATH is fully updated. Standard, safe workaround.
set +u
source .venv/bin/activate
set -u

echo "[$(date)] venv python: $(python --version 2>&1)"
echo "[$(date)] VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"

pip install --quiet -r requirements.txt
python -c "import torch; print('[import check] torch', torch.__version__, 'OK')"

echo "[$(date)] Running sanity check on $(hostname)"
python sanity_check_mes_switching_v2.py
echo "[$(date)] Done"
