import os
import subprocess
import shutil
import re

RUNNER_PATH = 'runners/separated/gnn_base_runner.py'

def set_decay_rate(decay_rate):
    """Updates the SHAPING_DECAY_RATE in gnn_base_runner.py."""
    with open(RUNNER_PATH, 'r') as f:
        content = f.read()
    
    # Replace SHAPING_DECAY_RATE = <value> with the new value
    new_content = re.sub(r'SHAPING_DECAY_RATE = [\d\.]+', f'SHAPING_DECAY_RATE = {decay_rate}', content)
    
    with open(RUNNER_PATH, 'w') as f:
        f.write(new_content)

def run_training(experiment_name, num_steps=292000):
    """
    Runs the training script.
    num_steps=292000 corresponds to 200 episodes (200 episodes * 365 days * 4 threads).
    """
    cmd = [
        "python", "train_multi_dc_gnn.py",
        "--experiment_name", experiment_name,
        "--num_env_steps", str(num_steps),
        "--seed", "1"
    ]
    print(f"\n[{experiment_name}] Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Backup original runner script
    print("Backing up original gnn_base_runner.py...")
    with open(RUNNER_PATH, 'r') as f:
        original_runner_str = f.read()
    
    try:
        # --- 1. Run No Decay (Baseline Ceiling) ---
        print("\n" + "="*60)
        print(" PHASE 2 - Run 1: No Decay (The Baseline Ceiling) | decay=1.0")
        print("="*60)
        set_decay_rate(1.0)
        run_training("gnn_decay_1.0", num_steps=292000)
        
        # --- 2. Aggressive Decay (The Handoff Shock) ---
        print("\n" + "="*60)
        print(" PHASE 2 - Run 2: Aggressive Decay (The Handoff Shock) | decay=0.95")
        print("="*60)
        set_decay_rate(0.95)
        run_training("gnn_decay_0.95", num_steps=292000)
        
        # --- 3. Smooth Baseline (The Goldilocks Handoff) ---
        print("\n" + "="*60)
        print(" PHASE 2 - Run 3: Smooth Baseline (The Goldilocks Handoff) | decay=0.998")
        print("="*60)
        set_decay_rate(0.998)
        run_training("gnn_decay_0.998", num_steps=292000)
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        
    finally:
        # Restore original runner script
        print("\nRestoring original gnn_base_runner.py...")
        with open(RUNNER_PATH, 'w') as f:
            f.write(original_runner_str)
        
        print("\n" + "="*60)
        print(" SENSITIVITY ANALYSIS COMPLETE ")
        print("="*60)
        print("The output models are saved in:")
        print(" - results/gnn_decay_1.0/")
        print(" - results/gnn_decay_0.95/")
        print(" - results/gnn_decay_0.998/")
