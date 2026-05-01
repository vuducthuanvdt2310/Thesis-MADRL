import os
import subprocess
import shutil

CONFIG_PATH = 'configs/multi_dc_config.yaml'

def set_heuristic_config(enabled_val, k_val=10.0):
    """Safely updates the enabled flag and k value for heuristic shaping in the config file."""
    with open(CONFIG_PATH, 'r') as f:
        lines = f.readlines()
    
    in_heuristic = False
    for i, line in enumerate(lines):
        if 'heuristic_shaping:' in line:
            in_heuristic = True
        if in_heuristic and 'enabled:' in line:
            prefix = line.split('enabled:')[0]
            lines[i] = f"{prefix}enabled: {'true' if enabled_val else 'false'}\n"
        if in_heuristic and 'k:' in line:
            prefix = line.split('k:')[0]
            lines[i] = f"{prefix}k: {k_val}\n"
            
    with open(CONFIG_PATH, 'w') as f:
        f.writelines(lines)

def run_training(experiment_name, num_steps=36500):
    """
    Runs the training script.
    num_steps=36500 corresponds to 100 episodes (since episode_length=365).
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
    # Backup original config
    print("Backing up original configuration...")
    with open(CONFIG_PATH, 'r') as f:
        original_config_str = f.read()
    
    try:
        # --- 1. Run WITH heuristic teacher ---
        print("\n" + "="*60)
        print(" PHASE 1: Training WITH Heuristic Teacher (k=10.0)")
        print("="*60)
        set_heuristic_config(True, k_val=10.0)
        # Using 36500 steps (100 episodes * 365 steps/episode)
        run_training("gnn_with_heuristic_k10", num_steps=36500)
        
        # --- 2. Run WITHOUT heuristic teacher ---
        print("\n" + "="*60)
        print(" PHASE 2: Training WITHOUT Heuristic Teacher")
        print("="*60)
        set_heuristic_config(False, k_val=10.0)
        run_training("gnn_without_heuristic_k10", num_steps=36500)
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        
    finally:
        # Restore original config
        print("\nRestoring original configuration...")
        with open(CONFIG_PATH, 'w') as f:
            f.write(original_config_str)
        
        print("\n" + "="*60)
        print(" TEST COMPLETE ")
        print("="*60)
        print("The output models are saved in:")
        print(" - results/gnn_with_heuristic_k10/")
        print(" - results/gnn_without_heuristic_k10/")
