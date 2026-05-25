#!/usr/bin/env python
"""
quick_train.py - Sequential training launcher for the NEW network topologies.

Edit the TRAINING_JOBS list below to choose which training scripts to run.
Each job runs for EPISODES_PER_JOB episodes (default: 15), then this launcher
moves on to the next job. To skip a job without deleting it, change its
enabled flag from True to False.

Usage:
    python quick_train.py

The launcher uses subprocess to call each existing train_multi_dc_*.py file,
so the behaviour of each underlying script is exactly the same as running it
manually — only the episode count and seed are reduced.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG -- Edit this section
# =============================================================================

# Number of training episodes per job. 1 episode = EPISODE_LENGTH simulated days.
# num_env_steps is computed as: EPISODES_PER_JOB * EPISODE_LENGTH * N_ROLLOUT_THREADS
# so the underlying CRunner will loop exactly EPISODES_PER_JOB times.
EPISODES_PER_JOB    = 15
EPISODE_LENGTH      = 365   # must match the value baked into each train script
N_ROLLOUT_THREADS   = 4     # must match the value baked into each train script

# Single seed to use per job (instead of the default [0..9] list which runs
# all 10 seeds back-to-back inside each script). Set to None to keep the
# script's default list — usually you don't want this here.
SEED = 0

# If a job fails (non-zero exit), should we abort the whole batch?
STOP_ON_ERROR = False


# -- TOGGLE WHICH JOBS TO RUN --
# Each tuple: (enabled, script_filename, experiment_name)
# Change True to False to skip a job. Reorder freely to change run order.

TRAINING_JOBS = [
    # ----- 1 DC x 3 Retailers -----
    (True,  "train_multi_dc_gnn_1x3.py",    "gnn_happo_1x3"),
    (True,  "train_multi_dc_happo_1x3.py",  "happo_1x3"),
    (True,  "train_multi_dc_mappo_1x3.py",  "mappo_1x3"),

    # ----- 1 DC x 10 Retailers -----
    (True,  "train_multi_dc_gnn_1x10.py",   "gnn_happo_1x10"),
    (True,  "train_multi_dc_happo_1x10.py", "happo_1x10"),
    (True,  "train_multi_dc_mappo_1x10.py", "mappo_1x10"),

    # ----- 2 DCs x 20 Retailers -----
    (True,  "train_multi_dc_gnn_2x20.py",   "gnn_happo_2x20"),
    (True,  "train_multi_dc_happo_2x20.py", "happo_2x20"),
    (True,  "train_multi_dc_mappo_2x20.py", "mappo_2x20"),

    # ----- 2 DCs x 40 Retailers -----
    (True,  "train_multi_dc_gnn_2x40.py",   "gnn_happo_2x40"),
    (True,  "train_multi_dc_happo_2x40.py", "happo_2x40"),
    (True,  "train_multi_dc_mappo_2x40.py", "mappo_2x40"),

    # ----- 4 DCs x 15 Retailers -----
    (False,  "train_multi_dc_gnn_4x15.py",   "gnn_happo_4x15"),
    (False,  "train_multi_dc_happo_4x15.py", "happo_4x15"),
    (False,  "train_multi_dc_mappo_4x15.py", "mappo_4x15"),

    # ----- 4 DCs x 30 Retailers -----
    (False,  "train_multi_dc_gnn_4x30.py",   "gnn_happo_4x30"),
    (False,  "train_multi_dc_happo_4x30.py", "happo_4x30"),
    (False,  "train_multi_dc_mappo_4x30.py", "mappo_4x30"),
]

# =============================================================================
# Runner -- no edits needed below
# =============================================================================

def main():
    here = Path(__file__).resolve().parent
    num_env_steps = EPISODES_PER_JOB * EPISODE_LENGTH * N_ROLLOUT_THREADS

    jobs_to_run = [(s, n) for (en, s, n) in TRAINING_JOBS if en]
    skipped     = [s for (en, s, _) in TRAINING_JOBS if not en]

    print("=" * 72)
    print("Quick Training Launcher")
    print("=" * 72)
    print(f"  Episodes per job   : {EPISODES_PER_JOB}")
    print(f"  Episode length     : {EPISODE_LENGTH}")
    print(f"  Rollout threads    : {N_ROLLOUT_THREADS}")
    print(f"  -> num_env_steps   : {num_env_steps:,}")
    print(f"  Seed (single run)  : {SEED}")
    print(f"  Jobs queued        : {len(jobs_to_run)}")
    if skipped:
        print(f"  Jobs skipped       : {len(skipped)}")
        for s in skipped:
            print(f"      - {s}")
    print("=" * 72)
    print()

    failures = []
    start_all = datetime.now()

    for idx, (script, exp_name) in enumerate(jobs_to_run, 1):
        script_path = here / script
        if not script_path.exists():
            print(f"\n[SKIP] {script} not found at {script_path}")
            failures.append((script, "missing"))
            if STOP_ON_ERROR:
                break
            continue

        job_start = datetime.now()
        print()
        print("#" * 72)
        print(f"#  [{idx}/{len(jobs_to_run)}]  {script}")
        print(f"#  experiment_name  : {exp_name}")
        print(f"#  started          : {job_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("#" * 72)

        cmd = [
            sys.executable,
            str(script_path),
            "--experiment_name",  exp_name,
            "--num_env_steps",    str(num_env_steps),
        ]
        if SEED is not None:
            cmd += ["--seed", str(SEED)]

        try:
            result = subprocess.run(cmd, cwd=str(here))
            elapsed = datetime.now() - job_start
            if result.returncode != 0:
                failures.append((script, f"exit {result.returncode}"))
                print(f"\n[FAIL] {script} -> exit code {result.returncode}  "
                      f"(elapsed {elapsed})")
                if STOP_ON_ERROR:
                    print("STOP_ON_ERROR=True -- aborting remaining jobs.")
                    break
            else:
                print(f"\n[OK]   {script}  (elapsed {elapsed})")
        except KeyboardInterrupt:
            print("\n[ABORT] Interrupted by user. Stopping batch.")
            break
        except Exception as e:
            failures.append((script, str(e)))
            print(f"\n[FAIL] {script} crashed: {e}")
            if STOP_ON_ERROR:
                break

    total_elapsed = datetime.now() - start_all

    print()
    print("=" * 72)
    print(f"BATCH COMPLETE    |    Total wall time: {total_elapsed}")
    print(f"  Succeeded : {len(jobs_to_run) - len(failures)} / {len(jobs_to_run)}")
    if failures:
        print(f"  Failed    : {len(failures)}")
        for s, reason in failures:
            print(f"      - {s}  ({reason})")
    print("=" * 72)


if __name__ == "__main__":
    main()
