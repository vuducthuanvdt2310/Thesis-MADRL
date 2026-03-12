#!/usr/bin/env python
"""
FAST DEBUG Training Script for Multi-DC 2-Echelon Inventory Environment

Purpose: Quickly validate if rewards trend upward after env/config fixes.
- 90-day episodes (vs 365) → 4x faster feedback per episode
- Fewer total steps → finishes in ~30–60 min instead of days
- Everything else identical to train_multi_dc_baseline.py

Run:
    python train_multi_dc_debug.py --experiment_name "debug_run"
"""

import sys
import os
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnvMultiDC, DummyVecEnvMultiDC
from runners.separated.runner import CRunner as Runner


def make_train_env(all_args):
    return SubprocVecEnvMultiDC(all_args)

def make_eval_env(all_args):
    return DummyVecEnvMultiDC(all_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":
    BASE_SAVE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

    parser = get_config()
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=17,         # 2 DCs + 15 Retailers
        episode_length=90,     # ← SHORT for fast feedback (vs 365 in full training)
        num_env_steps=900000,  # 90 days × 10 threads × ~1000 episodes
        n_rollout_threads=10,
        n_training_threads=1,
        algorithm_name="happo",
        experiment_name="debug_run",
        use_eval=True,
        n_eval_rollout_threads=1,
        eval_interval=5,       # Eval every 5 episodes (shorter episodes = faster evals)
        eval_episodes=3,
        log_interval=1,
        n_warmup_evaluations=3,
        n_no_improvement_thres=500,
    )

    all_args = parse_args(sys.argv[1:], parser)

    RESUME_MODEL_DIR = None
    if RESUME_MODEL_DIR:
        all_args.model_dir = RESUME_MODEL_DIR

    seeds = [all_args.seed[0]]  # Only run 1 seed for debug speed

    print("=" * 60)
    print("FAST DEBUG Training — 90-day episodes")
    print(f"Episodes: {int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads}")
    print(f"Expected wall-clock time: ~30-60 minutes")
    print("=" * 60)

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        run_dir = BASE_SAVE_DIR / "results" / all_args.experiment_name
        os.makedirs(str(run_dir), exist_ok=True)

        curr_run = f"run_seed_{seed + 1}"
        run_dir = run_dir / curr_run
        os.makedirs(str(run_dir), exist_ok=True)
        os.makedirs(str(run_dir / "models"), exist_ok=True)

        print(f"\nResults: {run_dir}\n")

        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args)

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": all_args.num_agents,
            "device": device,
            "run_dir": run_dir,
        }

        try:
            print("Starting debug training...\n")
            runner = Runner(config)
            reward, bw = runner.run()
            print(f"\nDebug training done. Best reward: {reward:.2f}")
        except KeyboardInterrupt:
            print("\nStopped manually.")
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            envs.close()
            if eval_envs is not envs:
                eval_envs.close()
