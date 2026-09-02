import os, json, time
from datetime import datetime

RESULTS_ROOT = "results"

def _flag_path(exp_name, benchmark, variant, seed):
    return os.path.join(RESULTS_ROOT, exp_name, "checkpoints",
                        f"{benchmark}__{variant}__seed{seed}.done")

def _json_path(exp_name, benchmark, variant, seed):
    return os.path.join(RESULTS_ROOT, exp_name, "checkpoints",
                        f"{benchmark}__{variant}__seed{seed}.json")

def _progress_log_path(exp_name):
    return os.path.join(RESULTS_ROOT, exp_name, "logs",
                        "experiment_progress.log")

def _seed_log_path(exp_name, benchmark, variant, seed):
    return os.path.join(RESULTS_ROOT, exp_name, "logs",
                        f"{benchmark}__{variant}__seed{seed}.log")

def resume_checkpoint_path(exp_name, benchmark, variant, seed):
    """
    Path for a resumable in-progress run checkpoint (DirectRegretOptimization.
    save_checkpoint/load_checkpoint) -- distinct from _json_path/_flag_path,
    which only hold the FINAL summary once a run completes. Separate '.pt'
    extension since it's a torch.save blob (model/optimizer state), not JSON;
    kept out of this module's own (torch-free) dependencies -- the actual
    torch.save/torch.load calls live in dro.py, this just builds the path.
    """
    return os.path.join(RESULTS_ROOT, exp_name, "checkpoints",
                        f"{benchmark}__{variant}__seed{seed}.resume.pt")

def has_resume_checkpoint(exp_name, benchmark, variant, seed):
    return os.path.exists(resume_checkpoint_path(exp_name, benchmark, variant, seed))

def delete_resume_checkpoint(exp_name, benchmark, variant, seed):
    """Called once a run's FINAL result is safely saved (after save_result's
    JSON+.done flag) -- the in-progress checkpoint is no longer needed."""
    path = resume_checkpoint_path(exp_name, benchmark, variant, seed)
    if os.path.exists(path):
        os.remove(path)

def setup_dirs(exp_name):
    """Create all required directories for an experiment."""
    for sub in ["checkpoints", "logs", "plots"]:
        os.makedirs(os.path.join(RESULTS_ROOT, exp_name, sub),
                    exist_ok=True)

def is_completed(exp_name, benchmark, variant, seed):
    """Return True if this (benchmark, variant, seed) run is already done."""
    return os.path.exists(_flag_path(exp_name, benchmark, variant, seed))

def save_result(exp_name, benchmark, variant, seed, result_dict):
    """
    Save result JSON then write .done flag.
    ALWAYS write JSON before flag so a crash between the two
    leaves a recoverable JSON with no flag (run will be retried).
    """
    json_path = _json_path(exp_name, benchmark, variant, seed)
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    # Only write flag after JSON is fully flushed
    open(_flag_path(exp_name, benchmark, variant, seed), 'w').close()

def load_result(exp_name, benchmark, variant, seed):
    with open(_json_path(exp_name, benchmark, variant, seed)) as f:
        return json.load(f)

def load_all_results(exp_name, benchmarks, variants, seeds):
    """
    Load all completed results. Returns dict keyed by
    (benchmark, variant, seed). Silently skips missing entries.
    """
    results = {}
    for b in benchmarks:
        for v in variants:
            for s in seeds:
                if is_completed(exp_name, b, v, s):
                    results[(b, v, s)] = load_result(exp_name, b, v, s)
    return results

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_global(exp_name, message):
    """Append a timestamped line to the global progress log."""
    path = _progress_log_path(exp_name)
    with open(path, 'a') as f:
        f.write(f"[{_ts()}] {message}\n")

def log_iter(exp_name, benchmark, variant, seed, iter_dict):
    """
    Append one line per BO iteration to the per-seed log.
    iter_dict keys: iter, regret, best, mean_reward, zero_frac,
                    rtg_target, batch_max_rtg, running_max_rtg, iter_time
    """
    path = _seed_log_path(exp_name, benchmark, variant, seed)
    parts = "  ".join(f"{k}={v}" for k, v in iter_dict.items())
    with open(path, 'a') as f:
        f.write(parts + "\n")

def print_progress(completed, skipped, failed, total, global_start):
    done = completed + skipped
    elapsed = time.perf_counter() - global_start
    rate = done / elapsed if elapsed > 0 else 1e-9
    eta_min = (total - done) / rate / 60 if rate > 0 else float('inf')
    print(f"  [{done}/{total}] completed={completed} skipped={skipped} "
          f"failed={failed} | ETA {eta_min:.1f} min")

def missing_runs(exp_name, benchmarks, variants, seeds):
    """Return list of (benchmark, variant, seed) not yet completed."""
    missing = []
    for b in benchmarks:
        for v in variants:
            for s in seeds:
                if not is_completed(exp_name, b, v, s):
                    missing.append((b, v, s))
    return missing


if __name__ == '__main__':
    exp = "test_exp"
    setup_dirs(exp)
    assert not is_completed(exp, "bench", "var", 0)
    save_result(exp, "bench", "var", 0, {"regret": [1.0, 0.5, 0.2]})
    assert is_completed(exp, "bench", "var", 0)
    r = load_result(exp, "bench", "var", 0)
    assert r["regret"] == [1.0, 0.5, 0.2]
    log_global(exp, "TEST PASSED")
    print("checkpoint.py self-test: PASSED")
    # Clean up
    import shutil
    shutil.rmtree(os.path.join(RESULTS_ROOT, exp))
