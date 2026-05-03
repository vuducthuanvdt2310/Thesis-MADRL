#!/usr/bin/env python
"""
GNN-HAPPO Training Script for SCALED Multi-DC Inventory Environment
Topology: 1 Supplier → 2 DCs → 30 Retailers (32 agents total)

Replicates train_multi_dc_gnn.py but uses configs/multi_dc_2x30_config.yaml.
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

# ================================================================
# GOOGLE COLAB CONFIGURATION
# ================================================================
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_google_drive():
    try:
        if os.path.exists('/content/drive/MyDrive'):
            print("[OK] Google Drive already mounted!")
            return True
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive mounted successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
        return False

USE_GOOGLE_DRIVE = False
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/thesis_models"

# ================================================================
# SCALED ENVIRONMENT CONFIG
# ================================================================
N_DCS = 2
N_RETAILERS = 30
N_AGENTS = N_DCS + N_RETAILERS  # 32
CONFIG_PATH = 'configs/multi_dc_2x30_config.yaml'

def make_train_env(all_args):
    return SubprocVecEnvMultiDC(all_args)

def make_eval_env(all_args):
    return DummyVecEnvMultiDC(all_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":
    in_colab = is_running_in_colab()
    use_gdrive = USE_GOOGLE_DRIVE or in_colab

    if use_gdrive:
        print("=" * 70)
        print("Google Colab Environment Detected")
        print("=" * 70)
        if in_colab:
            mount_success = mount_google_drive()
            if not mount_success:
                use_gdrive = False
        print()

    if use_gdrive:
        BASE_SAVE_DIR = Path(GOOGLE_DRIVE_PATH)
    else:
        BASE_SAVE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

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
    parser.add_argument('--single_agent_obs_dim', type=int, default=28)

    # DEFAULT CONFIGURATION for 2x30 scale
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon_2x30",
        num_agents=N_AGENTS,
        episode_length=365,
        num_env_steps=36500000,
        n_rollout_threads=4,
        n_training_threads=1,
        algorithm_name="gnn_happo",
        experiment_name="gnn_happo_2x30",
        use_eval=True,
        n_eval_rollout_threads=1,
        eval_interval=1,
        eval_episodes=5,
        log_interval=1,
        n_warmup_evaluations=3,
        n_no_improvement_thres=1000,
        entropy_coef=0.08,
        std_x_coef=2.0,
        std_y_coef=1.5,
    )

    all_args = parse_args(sys.argv[1:], parser)

    # Point to the 2x30 config
    all_args.env_config_path = CONFIG_PATH

    # DC obs = 28D, Retailer obs = 22D → max = 28
    all_args.single_agent_obs_dim = 28

    # --- Resume Training (Optional) ---
    RESUME_MODEL_DIR = None
    if RESUME_MODEL_DIR:
        all_args.model_dir = RESUME_MODEL_DIR

    seeds = all_args.seed
    if isinstance(seeds, int):
        seeds = [seeds]

    print("=" * 70)
    print("GNN-HAPPO Training — SCALED 2×30 (Proposed Method)")
    print("=" * 70)
    print(f"Config          : {CONFIG_PATH}")
    print(f"Agents          : {N_AGENTS} ({N_DCS} DCs + {N_RETAILERS} Retailers)")
    print(f"Parallel envs   : {all_args.n_rollout_threads}")
    print(f"Episode length  : {all_args.episode_length}")
    print(f"Total steps     : {all_args.num_env_steps:,}")
    print(f"GNN type        : {all_args.gnn_type}")
    print(f"GNN hidden dim  : {all_args.gnn_hidden_dim}")
    print(f"GNN layers      : {all_args.gnn_num_layers}")
    print("=" * 70)

    # CUDA
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

        print(f"Results: {run_dir}")
        print(f"Models:  {models_dir}\n")

        if not os.path.exists(seed_res_record_file):
            open(seed_res_record_file, 'a+')

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        print("Creating training environments...")
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = all_args.num_agents

        print(f"Environments created: {envs.num_envs} parallel envs")
        print(f"Agents per env: {num_agents}")
        print(f"Obs spaces: DC=28D, Retailer=22D (zero-padded to 28D for GNN)")
        print(f"Action spaces: all agents 3D continuous\n")

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        try:
            print("Starting GNN-HAPPO training...\n")
            runner = Runner(config)
            reward, bw = runner.run()

            with open(seed_res_record_file, 'a+') as f:
                f.write(str(seed) + ' ' + str(reward) + ' ')
                for fluc in bw:
                    f.write(str(fluc) + ' ')
                f.write('\n')

            print(f"\n{'='*70}")
            print(f"Training completed for seed {seed}")
            print(f"Final reward: {reward}")
            print(f"{'='*70}\n")

        except KeyboardInterrupt:
            print(f"\nTraining interrupted. Saving artifacts...")
            break

        except Exception as e:
            print(f"\nTraining failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            envs.close()
            if all_args.use_eval and eval_envs is not envs:
                eval_envs.close()

        # ZIP ARTIFACTS
        try:
            zip_filename = f"{all_args.experiment_name}"
            output_path = os.path.join(os.getcwd(), zip_filename)
            shutil.make_archive(output_path, 'zip', run_dir)
            print(f"✓ Zip: {output_path}.zip")
        except Exception as e:
            print(f"[FAIL] Zip failed: {e}")

    print(f"\n{'='*70}")
    print("All training runs completed!")
    print("=" * 70)
