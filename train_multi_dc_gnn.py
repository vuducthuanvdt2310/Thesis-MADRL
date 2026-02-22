#!/usr/bin/env python
"""
GNN-HAPPO Training Script for Multi-DC 2-Echelon Inventory Environment

This is the PROPOSED method using Graph Neural Networks to capture supply chain topology.

Key differences from baseline:
1. Uses GNN-HAPPO policy instead of standard MLP-based policy
2. Constructs and passes adjacency matrix to represent supply chain graph
3. Learns to coordinate agents through explicit graph structure

The training setup (parallel envs, episode length, total steps, eval interval, etc.)
is IDENTICAL to the baseline for a fair comparison.
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
    """Check if the script is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_google_drive():
    """Mount Google Drive in Colab."""
    try:
        if os.path.exists('/content/drive/MyDrive'):
            print("[OK] Google Drive already mounted!")
            return True

        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive mounted successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to mount Google Drive automatically.")
        print(f"Error: {e}")
        print("\n" + "="*70)
        print("MANUAL SETUP REQUIRED:")
        print("Please run this command in a Colab cell BEFORE running training:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        print("="*70 + "\n")
        return False

# Set this to True to force using Google Drive even if not in Colab
USE_GOOGLE_DRIVE = False

# Configure your Google Drive path here
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/thesis_models"

def make_train_env(all_args):
    """Create parallel training environments for multi-DC."""
    return SubprocVecEnvMultiDC(all_args)

def make_eval_env(all_args):
    """Create single evaluation environment for multi-DC."""
    return DummyVecEnvMultiDC(all_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":
    # ================================================================
    # GOOGLE DRIVE SETUP (for Google Colab)
    # ================================================================
    in_colab = is_running_in_colab()
    use_gdrive = USE_GOOGLE_DRIVE or in_colab

    if use_gdrive:
        print("="*70)
        print("Google Colab Environment Detected")
        print("="*70)
        if in_colab:
            mount_success = mount_google_drive()
            if not mount_success:
                print("WARNING: Continuing without Google Drive. Models will be saved locally.")
                use_gdrive = False
        print()

    # Set base directory for saving models
    if use_gdrive:
        BASE_SAVE_DIR = Path(GOOGLE_DRIVE_PATH)
        print(f"Models will be saved to Google Drive: {BASE_SAVE_DIR}")
    else:
        BASE_SAVE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
        print(f"Models will be saved locally: {BASE_SAVE_DIR / 'results'}")

    print()
    # ================================================================

    parser = get_config()

    # ================================================================
    # GNN-HAPPO SPECIFIC ARGUMENTS
    # ================================================================
    parser.add_argument('--gnn_type', type=str, default='GAT',
                       choices=['GAT', 'GCN'],
                       help='Type of GNN to use')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                       help='Hidden dimension for GNN layers')
    parser.add_argument('--gnn_num_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                       help='Number of attention heads for GAT')
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                       help='Dropout rate for GNN')
    parser.add_argument('--use_residual', type=lambda x: (str(x).lower() == 'true'),
                       default=True,
                       help='Use residual connections in GNN')
    parser.add_argument('--critic_pooling', type=str, default='mean',
                       choices=['mean', 'max', 'concat'],
                       help='Pooling method for critic')
    parser.add_argument('--single_agent_obs_dim', type=int, default=30,
                       help='Single agent observation dimension (both DCs and Retailers: 30D)')

    # ================================================================
    # DEFAULT CONFIGURATION — IDENTICAL to baseline for fair comparison
    # Only algorithm_name and experiment_name differ
    # ================================================================
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=17,        # 2 DCs + 15 Retailers
        episode_length=365,   # Days per episode
        num_env_steps=36500000,  # Total training steps (same as baseline)
        n_rollout_threads=10,    # Parallel environments (same as baseline)
        n_training_threads=1,
        algorithm_name="gnn_happo",
        experiment_name="gnn_happo_full",
        use_eval=True,
        n_eval_rollout_threads=1,
        eval_interval=1,      # Evaluate every 1 episode (same as baseline)
        eval_episodes=5,
        log_interval=1,
        n_warmup_evaluations=3,
        n_no_improvement_thres=1000
    )

    all_args = parse_args(sys.argv[1:], parser)

    # CRITICAL: Force single_agent_obs_dim to 36 (max obs dim for retailers)
    all_args.single_agent_obs_dim = 30

    # --- Resume Training (Optional) ---
    RESUME_MODEL_DIR = None
    # RESUME_MODEL_DIR = r"d:\thuan\thesis\...\results\gnn_happo_full\run_seed_1"
    if RESUME_MODEL_DIR:
        all_args.model_dir = RESUME_MODEL_DIR
        print(f"Resuming from: {all_args.model_dir}")
    # --------------------------------------------------

    seeds = all_args.seed

    print("="*70)
    print("GNN-HAPPO Training (Proposed Method)")
    print("="*70)
    print(f"Environment: {all_args.env_name}")
    print(f"Scenario: {all_args.scenario_name}")
    print(f"Algorithm: {all_args.algorithm_name}")
    print(f"Agents: {all_args.num_agents} (2 DCs + {all_args.num_agents - 2} Retailers)")
    print(f"Parallel envs: {all_args.n_rollout_threads}")
    print(f"Episode length: {all_args.episode_length}")
    print(f"Total steps: {all_args.num_env_steps:,}")
    print(f"GNN type: {all_args.gnn_type}")
    print(f"GNN hidden dim: {all_args.gnn_hidden_dim}")
    print(f"GNN layers: {all_args.gnn_num_layers}")
    print("="*70)

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

        # Create run directory
        run_dir = BASE_SAVE_DIR / "results" / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        curr_run = 'run_seed_%i' % (seed + 1)

        seed_res_record_file = run_dir / "seed_results.txt"

        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        # Create models directory for saving
        models_dir = run_dir / "models"
        if not models_dir.exists():
            os.makedirs(str(models_dir))

        print(f"Results will be saved to: {run_dir}")
        print(f"Models will be saved to: {models_dir}\n")

        if not os.path.exists(seed_res_record_file):
            open(seed_res_record_file, 'a+')

        # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Create environments
        print("Creating training environments...")
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = all_args.num_agents

        print(f"Environments created: {envs.num_envs} parallel envs")
        print(f"Agents per env: {num_agents}")
        print(f"Observation spaces: DCs=30D, Retailers=30D (uniform for GNN)")
        print(f"Action spaces: DCs=3D active (6D buffer), Retailers=3D active (6D buffer)\n")

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        # Run training with GNN-HAPPO
        try:
            print("Starting GNN-HAPPO training...\n")
            runner = Runner(config)
            reward, bw = runner.run()

            # Save final results
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
            print(f"\n{'='*70}")
            print(f"Training interrupted manually (KeyboardInterrupt)")
            print(f"Saving current artifacts before exit...")
            print(f"{'='*70}\n")
            break

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")

        finally:
            # Close environments
            envs.close()
            if all_args.use_eval and eval_envs is not envs:
                eval_envs.close()

        # ================================================================
        # ZIP ARTIFACTS FOR KAGGLE / EASY DOWNLOAD
        # ================================================================
        print("\n" + "="*70)
        print("Zipping Training Artifacts...")
        print("="*70)

        try:
            zip_filename = f"{all_args.experiment_name}"
            output_path = os.path.join(os.getcwd(), zip_filename)
            shutil.make_archive(output_path, 'zip', run_dir)

            print(f"✓ Zip archive created successfully!")
            print(f"  Location: {output_path}.zip")
            print(f"  Content:  {run_dir}")
            print("="*70)

        except Exception as e:
            print(f"[FAIL] Failed to create zip archive: {e}")
            print("="*70)

    print("\n" + "="*70)
    print("All training runs completed!")
    print("="*70)
