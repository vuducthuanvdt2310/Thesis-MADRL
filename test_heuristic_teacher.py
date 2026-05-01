import os
import subprocess
import shutil

CONFIG_PATH = 'configs/multi_dc_config.yaml'

def set_heuristic_enabled(enabled_val):
    """Safely updates the enabled flag for heuristic shaping in the config file."""
    with open(CONFIG_PATH, 'r') as f:
        lines = f.readlines()
    
    in_heuristic = False
    for i, line in enumerate(lines):
        if 'heuristic_shaping:' in line:
            in_heuristic = True
        if in_heuristic and 'enabled:' in line:
            prefix = line.split('enabled:')[0]
            lines[i] = f"{prefix}enabled: {'true' if enabled_val else 'false'}\n"
            break
            
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
        print(" PHASE 1: Training WITH Heuristic Teacher")
        print("="*60)
        set_heuristic_enabled(True)
        # Using 36500 steps (100 episodes * 365 steps/episode)
        run_training("gnn_with_heuristic", num_steps=36500)
        
        # --- 2. Run WITHOUT heuristic teacher ---
        print("\n" + "="*60)
        print(" PHASE 2: Training WITHOUT Heuristic Teacher")
        print("="*60)
        set_heuristic_enabled(False)
        run_training("gnn_without_heuristic", num_steps=36500)
        
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
        print(" - results/gnn_with_heuristic/")
        print(" - results/gnn_without_heuristic/")
