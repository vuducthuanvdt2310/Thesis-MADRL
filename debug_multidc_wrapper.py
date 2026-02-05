"""Quick debug script to test environment initialization"""
import numpy as np
from envs.env_wrappers import SubprocVecEnvMultiDC, DummyVecEnvMultiDC
from config import get_config
import sys

# Get config
parser = get_config()
all_args = parser.parse_known_args(sys.argv[1:])[0]

# Set required args
all_args.n_rollout_threads = 1
all_args.num_agents = 5

print("Creating environment...")
try:
    env = SubprocVecEnvMultiDC(all_args)
    print(f"✓ Environment created successfully")
    print(f"  Agents: {env.num_agent}")
    print(f"  Action spaces: {env.action_space}")
    print(f"  Observation spaces: {env.observation_space}")
    print(f"  Share obs spaces: {env.share_observation_space}")
    
    print("\nTesting reset...")
    obs, _ = env.reset()
    print(f"✓ Reset successful")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Expected: (n_envs=1, n_agents=5, obs_dim=varies)")
    
    # Check individual obs shapes
    for i in range(5):
        print(f"  Agent {i} obs shape: {obs[0][i].shape}")
    
    print("\nTesting step...")
    actions = [{
        0: np.array([10.0, 10.0, 10.0]),  # DC 0
        1: np.array([10.0, 10.0, 10.0]),  # DC 1
        2: np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]),  # Retailer 0
        3: np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]),  # Retailer 1
        4: np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]),  # Retailer 2
    }]
    
    obs, rews, dones, infos = env.step(actions)
    print(f"✓ Step successful")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Rewards shape: {rews.shape}")
    print(f"  Dones shape: {dones.shape}")
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
