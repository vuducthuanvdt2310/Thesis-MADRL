import sys
import os
import time
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Sensitivity Analysis for Actor Learning Rate')
    
    # Episode settings
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of training episodes per configuration (default: 100)')
    parser.add_argument('--episode_length', type=int, default=365,
                        help='Length of each episode in days (default: 365)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')
    
    # Training settings
    parser.add_argument('--n_rollout_threads', type=int, default=4,
                        help='Number of parallel rollout threads (default: 4)')
                        
    return parser.parse_args()

def main(args):
    # 3 values for actor learning rate
    actor_lrs = [0.001, 0.0005, 0.0001]
    
    # Calculate the required num_env_steps to achieve roughly the requested number of episodes
    num_env_steps = args.num_episodes * args.episode_length 
    
    print("=" * 80)
    print(f"Starting Sensitivity Analysis for actor_lr")
    print(f"Values to test: {actor_lrs}")
    print(f"Training for target ~{args.num_episodes} episodes per configuration")
    print(f"Episode length: {args.episode_length} days")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    sensitivity_results = []
    start_total_time = time.time()
    
    for lr in actor_lrs:
        print(f"\n{'#' * 80}")
        print(f"Starting Run: actor_lr = {lr} (seed = {args.seed})")
        print(f"{'#' * 80}\n")
        
        # We specify a unique experiment_name so logs and models are saved separately 
        # in the 'results/gnn_happo_sensitivity_actor_lr_X' directory
        experiment_name = f"gnn_happo_sensitivity_actor_lr_{lr}"
        
        # Build the command to call the main training script as a subprocess
        cmd = [
            sys.executable, "train_multi_dc_gnn.py",
            "--lr", str(lr),
            "--num_env_steps", str(num_env_steps),
            "--episode_length", str(args.episode_length),
            "--experiment_name", experiment_name,
            "--n_rollout_threads", str(args.n_rollout_threads),
            "--seed", str(args.seed)
        ]
        
        print(f"Executing: {' '.join(cmd)}\n")
        
        start_time = time.time()
        
        try:
            # Run the training script, this will stream output to the console automatically
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during training with actor_lr={lr}: {e}")
            print("Continuing to the next value...")
            continue
            
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        
        # Convert seconds to hours, minutes, seconds string
        hrs = int(elapsed_seconds // 3600)
        mins = int((elapsed_seconds % 3600) // 60)
        secs = int(elapsed_seconds % 60)
        time_str = f"{hrs}h {mins}m {secs}s"
        
        sensitivity_results.append({
            'actor_lr': lr,
            'time_taken_sec': elapsed_seconds,
            'time_str': time_str
        })
        
        print(f"\n--> Training for actor_lr={lr} completed in {time_str}.")

    # ================================================================
    # SENSITIVITY SUMMARY REPORT
    # ================================================================
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY REPORT")
    print("="*80)
    print(f"Total target episodes per configuration: {args.num_episodes}")
    print(f"Seed used: {args.seed}")
    print(f"{'Actor LR':<15} | {'Time Taken (Seconds)':<22} | {'Time Formatted':<15}")
    print("-" * 80)
    for res in sensitivity_results:
        print(f"{res['actor_lr']:<15.6f} | {res['time_taken_sec']:<22.2f} | {res['time_str']:<15}")
    print("="*80)
    
    total_elapsed = time.time() - start_total_time
    hrs = int(total_elapsed // 3600)
    mins = int((total_elapsed % 3600) // 60)
    secs = int(total_elapsed % 60)
    print(f"Total time for all runs: {hrs}h {mins}m {secs}s")
    print("\nYou can now view the results by running:")
    print("tensorboard --logdir=results")
    print("="*80)

if __name__ == "__main__":
    args = parse_args()
    main(args)
