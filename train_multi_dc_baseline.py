#!/usr/bin/env python
"""
Training script for Multi-DC 2-Echelon Inventory Environment with HAPPO

This script trains a Multi-Agent RL model on the multi-DC environment using HAPPO algorithm.
Models are saved automatically during training.
"""

import sys
import os
import socket
import numpy as np
from pathlib import Path
import shutil
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnvMultiDC, DummyVecEnvMultiDC
from runners.separated.runner import CRunner as Runner

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
        # Check if already mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("✓ Google Drive already mounted!")
            return True
        
        # Try to mount
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
# Default: /content/drive/MyDrive/thesis_models
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
    # DEFAULT CONFIGURATION FOR FULL TRAINING
    # These values are set as defaults but CAN be overridden by command line args
    # e.g., python train_multi_dc.py --experiment_name test_run
    # ================================================================
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=17,        # 2 DCs + 15 Retailers
        episode_length=365,  # Days per episode
        num_env_steps=36500000, # Total training steps
        n_rollout_threads=10, # Parallel environments
        n_training_threads=1, # Training threads
        algorithm_name="happo",
        experiment_name="full_training",
        use_eval=True,
        n_eval_rollout_threads=1,
        eval_interval=1,     # Evaluate every 1 episodes (was 500 - too large!)
        eval_episodes=5,
        log_interval=1,
        n_warmup_evaluations=3,  # Minimum evaluations before early stopping kicks in
        n_no_improvement_thres=1000  # Allow 20 evaluations without improvement before stopping
    )
    
    all_args = parse_args(sys.argv[1:], parser)
    
    # ================================================================
    # PARAMETER EXPLANATIONS:
    # ================================================================
    # --algorithm_name happo
    #   → HAPPO = Heterogeneous-Agent Proximal Policy Optimization
    #   → Best for environments with different agent types (DCs vs Retailers)
    #
    # --experiment_name full_training
    #   → Name to identify this training run
    #   → Results saved to: results/MultiDC/inventory_2echelon/happo/full_training/
    #
    # --num_env_steps 3650000
    #   → Total training steps = 10,000 episodes × 365 days
    #   → Expected training time: ~3-4 hours
    #
    # --episode_length 365
    #   → Each episode simulates 1 year (365 days)
    #   → Matches your data file length
    #
    # --n_rollout_threads 5
    #   → Runs 5 parallel environments simultaneously
    #   → 5x faster than single environment
    #   → Effective throughput: ~9,000 episodes/hour
    #
    # --save_interval (Unused)
    #   → This parameter is NOT used.
    #   → Models are saved automatically when evaluation reward improves.
    #   → Checkpoints: "A better model is saved!" will appear in logs.
    #   → Locations: results/.../models/actor_agentN.pt
    #
    # --use_eval True
    #   → Enables periodic evaluation during training
    #   → Helps track performance without noise from exploration
    #
    # --eval_interval 250
    #   → Runs evaluation every 250 episodes
    #   → 40 evaluation runs total over 10,000 episodes
    #
    # --log_interval 10
    #   → Prints progress to console every 10 episodes
    #   → Helps monitor training in real-time
    # ================================================================
    
    # --- Resume Training (Optional) ---
    # Set this path to resume from a saved checkpoint
    RESUME_MODEL_DIR = None
    # RESUME_MODEL_DIR = r"D:\thuan\thesis\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management\results\09Feb_test1_kaggle\run_seed_1"
    if RESUME_MODEL_DIR:
        all_args.model_dir = RESUME_MODEL_DIR
        print(f"Resuming from: {all_args.model_dir}")
    # --------------------------------------------------

    seeds = all_args.seed

    print("="*70)
    print("Multi-DC 2-Echelon Inventory Training")
    print("="*70)
    print(f"Environment: {all_args.env_name}")
    print(f"Scenario: {all_args.scenario_name}")
    print(f"Algorithm: {all_args.algorithm_name}")
    print(f"Agents: {all_args.num_agents} (2 DCs + {all_args.num_agents - 2} Retailers)")
    print(f"Parallel envs: {all_args.n_rollout_threads}")
    print(f"Episode length: {all_args.episode_length}")
    print(f"Total steps: {all_args.num_env_steps:,}")
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
        print(f"Observation spaces: DCs=30D, Retailers=36D")
        print(f"Action spaces: DCs=3D continuous, Retailers=6D continuous\n")

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        # Run training
        # Run training
        try:
            print("Starting training...\n")
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
            # Break the loop to stop training for other seeds too
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
            # We want to zip the specific run directory: run_dir
            # Format: results/experiment_name/run_seed_X
            
            # Output zip filename
            zip_filename = f"{all_args.experiment_name}"
            
            # For Kaggle, it is best to save the zip in the working directory (./)
            # or strictly in /kaggle/working if we want to be safe, but usually ./ works.
            output_path = os.path.join(os.getcwd(), zip_filename)
            
            # Create zip archive
            # root_dir is the directory we want to compress
            shutil.make_archive(output_path, 'zip', run_dir)
            
            print("[OK] Zip archive created successfully!")
            print(f"  Location: {output_path}.zip")
            print(f"  Content:  {run_dir}")
            print("="*70)
            
        except Exception as e:
            print(f"[FAIL] Failed to create zip archive: {e}")
            print("="*70)

    print("\n" + "="*70)
    print("All training runs completed!")
    print("="*70)
