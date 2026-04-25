import sys
import os

# ==============================================================================
# WRAPPER SCRIPT: test_handover.py
# Analyzes the reward shaping "handover point" using the existing GNN training loop
# ==============================================================================

# 1. Modify sys.argv to configure the short run
# 200 episodes = 292,000 steps (200 episodes * 365 steps/episode * 4 threads)
sys.argv = [
    "train_multi_dc_gnn.py",
    "--experiment_name", "handover_test",
    "--num_env_steps", "292000",   
    "--n_rollout_threads", "4",
    "--episode_length", "365",
    "--save_interval", "10000",     # Extremely high to prevent normal model saving
]

# 2. Monkey-patch GNNRunner to apply modifications
from runners.separated.gnn_base_runner import GNNRunner

# Disable heavy I/O (.pt file saving)
GNNRunner.save = lambda self, *args, **kwargs: None

original_run = GNNRunner.run

def mocked_run(self):
    print("\n[INFO] Injecting aggressive shaping weight decay (<0.01 by Episode 50)")
    print("[INFO] Creating targeted log: handover_log.csv\n")

    # Initialize targeted log
    with open("handover_log.csv", "w") as f:
        f.write("Episode,Episode_Reward,w_shape_value\n")

    # Save original bound method
    original_decay = self.envs.decay_shaping_weight

    def aggressive_decay_and_log(decay_rate=0.998):
        # Apply aggressive decay: 0.91^50 ≈ 0.0089
        val = original_decay(decay_rate=0.91)
        
        # GNNRunner logs to progress.csv right before decaying the weight. 
        # We can extract the latest episode and reward from it.
        csv_path = os.path.join(str(self.run_dir), "progress.csv")
        try:
            with open(csv_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    episode = last_line[0]
                    reward = last_line[2]
                    
                    # Append to our targeted log
                    with open("handover_log.csv", "a") as log_f:
                        log_f.write(f"{episode},{reward},{val:.6f}\n")
        except Exception as e:
            pass
            
        return val

    # Override the environment's decay method for this runner
    self.envs.decay_shaping_weight = aggressive_decay_and_log
    
    # Run the original training loop
    return original_run(self)

GNNRunner.run = mocked_run

if __name__ == "__main__":
    print("Starting targeted handover test...")
    import runpy
    # Execute the existing train_multi_dc_gnn.py script safely
    runpy.run_path("train_multi_dc_gnn.py", run_name="__main__")
