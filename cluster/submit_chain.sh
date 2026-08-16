#!/bin/bash
# Submits the full mes_switching_v2_cluster experiment as 3 independent
# job CHAINS (one per benchmark, one per "m"-series node), each chain being
# CHAIN_LENGTH sequential 4-hour jobs linked with --dependency=afterany so
# job N+1 only starts once job N has finished/died, and automatically
# resumes/skips whatever job N already completed (see run_shard.sbatch's
# header for why no explicit resume logic is needed).
#
# Over-provisioning CHAIN_LENGTH is cheap: once a benchmark's full 40-run
# sub-grid (8 variants x 5 seeds) is done, every later job in its chain
# finds nothing to do (run_experiment_parallel.py sees "0 runs remaining")
# and exits in seconds -- it just occupies a queue slot briefly, no wasted
# compute. Better to over-provision than under-provision and have to notice
# a chain stalled out and manually submit more.
#
# ============================================================================
# ONE-TIME SETUP (do this before running this script):
# ============================================================================
#   1. rsync this whole project directory to the cluster, e.g.:
#        rsync -avz --exclude .venv --exclude results/ \
#            /path/to/DRO-aistats-submission/ \
#            $USER@fe.ai.cs.uchicago.edu:/net/scratch/$USER/DRO-aistats-submission/
#      (/net/scratch chosen for speed/NFS-shared access across all "m"
#      nodes during the run -- copy final results/ to /home/$USER or
#      /net/projects afterward, since /net/scratch may be auto-deleted
#      after ~90 days.)
#
#   2. SSH in (ssh $USER@fe.ai.cs.uchicago.edu), cd into that directory, and
#      build a fresh venv there (the local .venv on your Mac is macOS-built,
#      won't run on the cluster's Linux nodes):
#        python3 -m venv .venv
#        source .venv/bin/activate
#        pip install -r requirements.txt
#
#   3. DONE -- confirmed via cs-sinfo: PARTITION=yuxinchen-contrib,
#      NODES=(m001 m002 m003), already filled in below. Re-run cs-sinfo
#      yourself if you suspect node availability/partition access has
#      changed since.
#
#   4. Run the sanity check (fast) and, ideally, the timing calibration
#      (slower -- gives real per-run cost numbers on THIS cluster's
#      hardware, which may differ from local timing) BEFORE submitting the
#      real chain, from the login node:
#        python sanity_check_mes_switching_v2.py
#
# ============================================================================

set -euo pipefail

PARTITION="yuxinchen-contrib"             # from cs-sinfo -- your group's dedicated partition
                                           # (m001/m002/m003 are also reachable via the public
                                           # "general" partition, but yuxinchen-contrib should have
                                           # less contention/better priority for your group)
# m002 (Ackley_5D): still strictly pinned -- ~30hr projected start, identical
# across multiple resubmissions at different times/sizes, an externally-fixed
# backfill block our own request can't move, but survivable.
#
# m001 (Ackley_2D) and m003 (Ackley_10D): both use candidate lists now, NOT
# pinned. m001 initially responded well to the cpu=120->64 resize (~31hr ->
# ~19hr projected start) but a later recheck showed the estimate had gotten
# WORSE over time (~19hr -> ~31hr, priority climbing from 5->146 but still
# losing ground to other jobs) rather than resolving -- genuine ongoing
# contention, not a one-time fixable number. m003's original ~9.7 DAY
# estimate was the same "fixed, identical across resubmissions" signature
# m002 shows. For both, a "g"/"q" (L40S, Ada generation) fallback measured
# ~5x slower per-iteration than local in a direct test (a dual-socket
# effect, not CPU generation), but Ackley_5D and Ackley_10D both fully
# completed all 200 iterations x 40 runs under that same slowdown on q002/
# g001/g002/q001 already, so it's a proven, working path, not a gamble at
# this point. Not free: Ackley_2D's cost-audit numbers won't be comparable
# to Ackley_5D's (which is itself already on non-"m" hardware) or to
# whatever Ackley_10D actually landed on. Worth revisiting if m001/m003 ever
# free up on their own -- neither has shown m002's dead-fixed pattern, so
# there's a real chance they do.
NODES=("m001,g001,g002,q001,q002" m002 "m003,g001,g002,q001,q002")
BENCHMARKS=(Ackley_2D Ackley_5D Ackley_10D)                # one benchmark per node, matched by index
PROJECT_DIR="/net/scratch/$USER/DRO-aistats-submission"
WORKERS=32                                # was 40 (cpu=120) -- scontrol show job showed backfill
THREADS_PER_WORKER=2                      # projecting ~9.7 DAYS for a 120/128-cpu request on m003
                                           # specifically. 32*2=64 cpus matches the size of jobs
                                           # we've repeatedly seen actually get scheduled promptly
                                           # on these nodes. Re-verify with scontrol show job on the
                                           # new submission before trusting this is fixed.
CHAIN_LENGTH=8                            # 8 x 3h50m ~= 30.7 hours of budget per benchmark

if [ "${#NODES[@]}" -ne "${#BENCHMARKS[@]}" ]; then
    echo "NODES and BENCHMARKS must be the same length (one node per benchmark)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    node="${NODES[$i]}"
    echo "=== Submitting chain for $benchmark on $node ==="

    prev_jobid=""
    for ((j = 1; j <= CHAIN_LENGTH; j++)); do
        if [ -z "$prev_jobid" ]; then
            dep_args=()
        else
            dep_args=(--dependency=afterany:"$prev_jobid")
        fi

        jobid=$(sbatch --parsable \
            -p "$PARTITION" \
            -w "$node" \
            "${dep_args[@]}" \
            --export="ALL,PROJECT_DIR=$PROJECT_DIR,BENCHMARK=$benchmark,WORKERS=$WORKERS,THREADS_PER_WORKER=$THREADS_PER_WORKER" \
            "$SCRIPT_DIR/run_shard.sbatch")

        echo "  job $j/$CHAIN_LENGTH: $jobid"
        prev_jobid="$jobid"
    done
done

echo ""
echo "All chains submitted. Check status with: cs-squeue"
echo "Partial progress at any point: python build_partial_regret_plot.py --exp-name mes_switching_v2_cluster"
