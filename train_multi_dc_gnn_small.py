#!/usr/bin/env python
"""
FAST GNN-HAPPO Training Script for Multi-DC 2-Echelon Inventory Environment
(Small Demo Version - 30 days per episode)

This script is configured to train very quickly to demonstrate the process
within ~5 minutes, outputting models that are fully compatible with your
test_trained_model_gnn.py script.
"""

import sys
import os
import numpy as np
from pathlib import Path
import shutil
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnvMultiDC, DummyVecEnvMultiDC
from runners.separated.gnn_base_runner import GNNRunner as Runner

def is_running_in_colab():
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False

def make_train_env(all_args):
    return SubprocVecEnvMultiDC(all_args)

def make_eval_env(all_args):
    return DummyVecEnvMultiDC(all_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":
    BASE_SAVE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f"Models will be saved locally: {BASE_SAVE_DIR / 'results'}")

    parser = get_config()

    # GNN-HAPPO SPECIFIC ARGUMENTS
    parser.add_argument('--gnn_type', type=str, default='GCN',
                       choices=['GAT', 'GCN'])
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--use_residual', type=lambda x: (str(x).lower() == 'true'),
                       default=True)
    parser.add_argument('--critic_pooling', type=str, default='mean',
                       choices=['mean', 'max', 'concat'])
    parser.add_argument('--single_agent_obs_dim', type=int, default=30)
    
    # We add env_config_path just in case, but it uses default multi_dc_config.yaml
    parser.add_argument('--env_config_path', type=str, default='configs/multi_dc_config.yaml')

    # ================================================================
    # FAST DEMO CONFIGURATION
    # ================================================================
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=17,        # 2 DCs + 15 Retailers
        episode_length=15,    # ** 30 Days per episode **
        
        # Reduced steps for ~5 minute training (e.g. 100 episodes total)
        num_env_steps=1200,  # 4 threads * 30 steps * 100 updates = 12000
        
        n_rollout_threads=4,  # Parallel environments
        n_training_threads=1,
        algorithm_name="gnn_happo",
        experiment_name="gnn_happo_small",
        use_eval=True,
        n_eval_rollout_threads=1,
        eval_interval=1,      # Evaluate every 1 episode
        eval_episodes=2,
        log_interval=1,
        n_warmup_evaluations=3,
        n_no_improvement_thres=1000,
        
        entropy_coef=0.08,
        std_x_coef=2.0,
        std_y_coef=1.5,
    )

    all_args = parse_args(sys.argv[1:], parser)

    # CRITICAL: single_agent_obs_dim must equal the LARGEST obs dim across all agents.
    all_args.single_agent_obs_dim = 28

    seeds = all_args.seed
    if isinstance(seeds, int):
        seeds = [seeds]

    print("="*70)
    print("FAST GNN-HAPPO Training (30 Days Demo)")
    print("="*70)
    print(f"Environment: {all_args.env_name}")
    print(f"Parallel envs: {all_args.n_rollout_threads}")
    print(f"Episode length: {all_args.episode_length} days")
    print(f"Total steps: {all_args.num_env_steps:,}")
    print("="*70)

    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU for training...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU for training...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Training starts for seed: {seed}")
        print(f"{'='*70}\n")

        run_dir = BASE_SAVE_DIR / "results" / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        curr_run = 'run_seed_%i' % (seed + 1)
        seed_res_record_file = run_dir / "seed_results.txt"
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        models_dir = run_dir / "models"
        if not models_dir.exists():
            os.makedirs(str(models_dir))

        print(f"Results will be saved to: {run_dir}")
        print(f"Models will be saved to: {models_dir}\n")

        if not os.path.exists(seed_res_record_file):
            open(seed_res_record_file, 'a+')

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        print("Creating training environments...")
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = all_args.num_agents

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        try:
            print("Starting FAST GNN-HAPPO training...\n")
            runner = Runner(config)
            reward, bw = runner.run()

            with open(seed_res_record_file, 'a+') as f:
                f.write(str(seed) + ' ' + str(reward) + ' ')
                if bw is not None:
                    for fluc in bw:
                        f.write(str(fluc) + ' ')
                f.write('\n')

            print(f"\n{'='*70}")
            print(f"Training completed for seed {seed}")
            print(f"Final reward: {reward}")
            print(f"{'='*70}\n")

        except KeyboardInterrupt:
            print(f"\n{'='*70}")
            print(f"Training interrupted manually (KeyboardInterrupt)")
            print(f"{'='*70}\n")
            break
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")
        finally:
            envs.close()
            if all_args.use_eval and eval_envs is not envs:
                eval_envs.close()

    print("\n" + "="*70)
    print("All training runs completed!")
    print(f"To test this model, you can run:")
    print(f"python test_trained_model_gnn.py --model_dir results/gnn_happo_small/run_seed_1/models --episode_length 30 --num_episodes 5 --experiment_name eval_gnn_small")
    print("="*70)
