"""
Quick test training script for Multi-DC Environment - Simple version
"""

import sys
import os
import time
import numpy as np
from envs.multi_dc_env import MultiDCInventoryEnv

def test_train_multi_dc_simple():
    print("=" * 70)
    print("Multi-DC Environment Training Test")
    print("=" * 70)
    
    # Initialize environment
    print("\n[1] Initializing environment...")
    env = MultiDCInventoryEnv(config_path='configs/multi_dc_config.yaml')
    print(f"Environment initialized: {env.n_agents} agents, {env.n_skus} SKUs")
    
    # Test single episode
    print("\n[2] Running single episode...")
    obs = env.reset()
    
    episode_rewards = {i: 0 for i in range(env.n_agents)}
    steps = 0
    done = False
    
    start_time = time.time()
    
    while not done and steps < env.max_days:
        # Random continuous actions
        actions = {}
        
        # DC actions (agents 0, 1): 3D continuous
        for dc_id in [0, 1]:
            actions[dc_id] = np.random.uniform(0, 20, 3)
        
        # Retailer actions (agents 2, 3, 4): 6D continuous
        for retailer_id in [2, 3, 4]:
            actions[retailer_id] = np.random.uniform(0, 10, 6)
        
        # Step
        obs, rewards, dones, infos = env.step(actions)
        
        # Accumulate rewards
        for i in range(env.n_agents):
            episode_rewards[i] += rewards[i]
        
        done = dones[0]
        steps += 1
        
        if steps % 50 == 0:
            avg_reward = np.mean(list(rewards.values()))
            print(f"  Step {steps}/{env.max_days}: Avg step reward = {avg_reward:.1f}")
    
    elapsed = time.time() - start_time
    steps_per_sec = steps / elapsed
    
    print(f"\nEpisode completed: {steps} steps in {elapsed:.2f}s")
    print(f"Performance: {steps_per_sec:.1f} steps/second")
    
    # Print episode rewards
    print(f"\nEpisode Rewards ({steps} steps):")
    print(f"  DC_0 (Agent 0): {episode_rewards[0]:,.0f}")
    print(f"  DC_1 (Agent 1): {episode_rewards[1]:,.0f}")
    print(f"  Retailer_0 (Agent 2): {episode_rewards[2]:,.0f}")
    print(f"  Retailer_1 (Agent 3): {episode_rewards[3]:,.0f}")
    print(f"  Retailer_2 (Agent 4): {episode_rewards[4]:,.0f}")
    print(f"  Total: {sum(episode_rewards.values()):,.0f}")
    
    # Multi-episode benchmark
    print("\n[3] Multi-episode benchmark (10 episodes)...")
    total_steps = 0
    start_time = time.time()
    
    for ep in range(10):
        obs = env.reset()
        done = False
        ep_steps = 0
        
        while not done and ep_steps < env.max_days:
            # Random actions
            actions = {
                0: np.random.uniform(0, 20, 3),
                1: np.random.uniform(0, 20, 3),
                2: np.random.uniform(0, 10, 6),
                3: np.random.uniform(0, 10, 6),
                4: np.random.uniform(0, 10, 6),
            }
            
            obs, rewards, dones, infos = env.step(actions)
            done = dones[0]
            ep_steps += 1
        
        total_steps += ep_steps
        if (ep + 1) % 5 == 0:
            print(f"  Completed {ep + 1}/10 episodes ({total_steps} total steps)")
    
    elapsed = time.time() - start_time
    avg_steps_per_sec = total_steps / elapsed
    
    print(f"\nBenchmark: {total_steps} steps in {elapsed:.2f}s")
    print(f"Average: {avg_steps_per_sec:.1f} steps/second")
    
    # Training time estimates
    print("\n" + "=" * 70)
    print("Training Time Estimates")
    print("=" * 70)
    
    episode_length = env.max_days
    
    scenarios = [
        ("Quick test", 100, 1),
        ("Medium run", 1_000, 5),
        ("Full training", 10_000, 5),
        ("Extended training", 30_000, 10),
    ]
    
    print(f"\nAssumptions:")
    print(f"  - Episode length: {episode_length} steps")
    print(f"  - Performance: {avg_steps_per_sec:.1f} steps/sec (random policy)")
    print(f"  - With learning: ~{avg_steps_per_sec * 0.5:.1f} steps/sec (50% slower)")
    print(f"\nEstimates:\n")
    
    for name, episodes, parallel in scenarios:
        total_steps = episodes * episode_length
        
        # Raw time (no learning)
        raw_time_sec = total_steps / avg_steps_per_sec / parallel
        
        # With learning overhead (2x slower)
        with_learning_sec = raw_time_sec * 2
        
        hours = with_learning_sec / 3600
        
        print(f"{name}:")
        print(f"  Episodes: {episodes:,} x {parallel} parallel = {episodes * parallel:,} total")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Estimated time: {hours:.1f} hours ({hours/24:.1f} days)")
        print()
    
    print("=" * 70)
    print("Test Training Complete!")
    print("=" * 70)
    print("\nConclusions:")
    print(f"  - Environment is functional")
    print(f"  - Random policy achieves ~{avg_steps_per_sec:.0f} steps/sec")
    print(f"  - Ready for HAPPO/MAPPO training")
    print(f"\nNext steps:")
    print(f"  1. Integrate with existing HAPPO training framework")
    print(f"  2. Configure 2 policy groups (DCs + Retailers)")
    print(f"  3. Start with 1,000 episodes to verify learning")

if __name__ == '__main__':
    test_train_multi_dc_simple()
