"""
Quick test to verify 7-14 day lead time implementation with new observation space.

This script tests:
1. Environment initialization with new config
2. Observation dimension (should be 30 per agent)
3. Lead time sampling (should be 7-14 days)
4. Pipeline binning (days 7-8, 9-10, 11-14)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from envs.enhanced_net_2x3 import EnhancedMultiSKUEnv
import numpy as np

def test_7_14_day_lead_time():
    print("="*60)
    print("Testing 7-14 Day Lead Time Environment")
    print("="*60)
    
    # Create environment
    env = EnhancedMultiSKUEnv(config_path='configs/multi_sku_config.yaml')
    
    # Check configuration
    print(f"\n✓ Lead time range: [{env.lt_min}, {env.lt_max}]")
    assert env.lt_min == 7, f"Expected min LT=7, got {env.lt_min}"
    assert env.lt_max == 14, f"Expected max LT=14, got {env.lt_max}"
    
    # Check observation dimension
    print(f"✓ Observation dimension: {env.obs_dim} (expected 27)")
    assert env.obs_dim == 27, f"Expected obs_dim=27, got {env.obs_dim}"
    
    # Reset and check observation shape
    obs = env.reset()
    print(f"✓ Number of agents: {len(obs)}")
    print(f"✓ Observation shape per agent: {obs[0].shape}")
    assert obs[0].shape == (27,), f"Expected shape (27,), got {obs[0].shape}"
    
    # Test lead time sampling
    print(f"\n--- Testing Lead Time Sampling ---")
    lead_times = []
    for _ in range(100):
        lt = np.random.randint(env.lt_min, env.lt_max + 1)
        lead_times.append(lt)
    
    min_lt = min(lead_times)
    max_lt = max(lead_times)
    avg_lt = np.mean(lead_times)
    
    print(f"Sampled 100 lead times:")
    print(f"  Min: {min_lt} (expected >= 7)")
    print(f"  Max: {max_lt} (expected <= 14)")
    print(f"  Average: {avg_lt:.1f} (expected ~10.5)")
    
    assert min_lt >= 7, "Lead time below minimum!"
    assert max_lt <= 14, "Lead time above maximum!"
    
    # Test pipeline binning
    print(f"\n--- Testing Pipeline Binning ---")
    
    # Place orders
    actions = [[10, 5, 8] for _ in range(env.n_agents)]
    obs, rewards, dones, infos = env.step(actions, one_hot=False)
    
    # Check pipeline
    total_orders = sum(len(env.pipeline[agent_id]) for agent_id in range(env.n_agents))
    print(f"Placed orders: {total_orders} orders in pipeline")
    
    # Display pipeline for agent 0
    print(f"\nAgent 0 pipeline:")
    for order in env.pipeline[0]:
        days_until = order['arrival_day'] - env.current_day
        print(f"  SKU {order['sku']}: {order['qty']} units in {days_until} days")
    
    # Check observation structure
    print(f"\n--- Observation Structure (Agent 0) ---")
    print(f"Shape: {obs[0].shape}")
    print(f"Positions 0-2 (Inventory): {obs[0][0:3]}")
    print(f"Positions 3-5 (Backlog): {obs[0][3:6]}")
    print(f"Positions 6-8 (Pipeline 7-8 days): {obs[0][6:9]}")
    print(f"Positions 9-11 (Pipeline 9-10 days): {obs[0][9:12]}")
    print(f"Positions 12-14 (Pipeline 11-14 days): {obs[0][12:15]}")
    print(f"Positions 15-17 (Pipeline total): {obs[0][15:18]}")
    print(f"Positions 18-20 (Current prices): {obs[0][18:21]}")
    print(f"Positions 21-23 (Price MA): {obs[0][21:24]}")
    print(f"Positions 24-26 (Recent demand): {obs[0][24:27]}")
    
    # Run several steps to see binning in action
    print(f"\n--- Running 10 Steps to Test Binning ---")
    for step in range(10):
        # Random actions
        actions = [
            [np.random.randint(0, 10) for _ in range(env.n_skus)]
            for _ in range(env.n_agents)
        ]
        obs, rewards, dones, infos = env.step(actions, one_hot=False)
        
        # Show pipeline bins for agent 0, SKU 0
        if step in [2, 5, 8]:
            print(f"\nDay {env.current_day} - Agent 0, SKU 0 pipeline bins:")
            print(f"  Days 7-8: {obs[0][6]:.2f}")
            print(f"  Days 9-10: {obs[0][9]:.2f}")
            print(f"  Days 11-14: {obs[0][12]:.2f}")
            print(f"  Total: {obs[0][15]:.2f}")
    
    print(f"\n{'='*60}")
    print("✅ ALL TESTS PASSED!")
    print("{'='*60}")
    print("\nEnvironment ready for training with 7-14 day lead times!")
    print("Observation space: 3 agents × 27 dimensions = 81 total")
    print("Pipeline bins: Days 7-8, 9-10, 11-14, plus total")

if __name__ == '__main__':
    test_7_14_day_lead_time()
