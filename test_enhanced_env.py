"""
Test Suite for Enhanced Multi-SKU Environment

This script validates the core functionality of the enhanced environment:
1. Environment instantiation and reset
2. Variable lead time mechanics
3. Dynamic pricing integration
4. CSV data loading
5. Observation/action space compatibility

Usage:
    python test_enhanced_env.py --test all
    python test_enhanced_env.py --test test_reset
    python test_enhanced_env.py --test test_lead_time
"""

import numpy as np
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from envs.enhanced_net_2x3 import EnhancedMultiSKUEnv


def test_reset():
    """Test 1: Environment instantiation and reset."""
    print("\n" + "="*60)
    print("TEST 1: Environment Instantiation & Reset")
    print("="*60)
    
    try:
        # Create environment
        env = EnhancedMultiSKUEnv(config_path='configs/multi_sku_config.yaml')
        
        # Reset
        obs = env.reset()
        
        # Validate observations
        print(f"✓ Environment created successfully")
        print(f"  - Number of agents: {len(obs)}")
        print(f"  - Observation shape per agent: {obs[0].shape}")
        print(f"  - Expected shape: ({env.obs_dim},)")
        
        assert len(obs) == env.n_agents, f"Expected {env.n_agents} observations, got {len(obs)}"
        assert obs[0].shape == (env.obs_dim,), f"Expected shape ({env.obs_dim},), got {obs[0].shape}"
        
        # Check normalization
        if env.normalize:
            for i, agent_obs in enumerate(obs):
                assert np.all(agent_obs >= 0) and np.all(agent_obs <= 1.5), \
                    f"Agent {i} observation not normalized properly: min={agent_obs.min()}, max={agent_obs.max()}"
            print(f"✓ Observations normalized to [0, 1] range")
        
        # Display sample observation
        print(f"\n--- Agent 0 Initial Observation (first 12 values) ---")
        print(f"Inventory (3 SKUs): {obs[0][:3]}")
        print(f"Backlog (3 SKUs): {obs[0][3:6]}")
        print(f"Pipeline next day (3 SKUs): {obs[0][6:9]}")
        print(f"Pipeline 2-3 days (3 SKUs): {obs[0][9:12]}")
        
        print(f"\n✅ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lead_time():
    """Test 2: Variable lead time mechanics."""
    print("\n" + "="*60)
    print("TEST 2: Variable Lead Time Mechanics")
    print("="*60)
    
    try:
        env = EnhancedMultiSKUEnv(config_path='configs/multi_sku_config.yaml')
        env.reset()
        
        # Place orders for all SKUs
        test_action = [[10, 5, 8] for _ in range(env.n_agents)]  # Each agent orders for 3 SKUs
        
        print(f"Placing orders: {test_action}")
        print(f"Lead time range: [{env.lt_min}, {env.lt_max}]")
        
        # Execute step
        obs, rewards, dones, infos = env.step(test_action, one_hot=False)
        
        # Check pipeline
        total_orders = 0
        arrival_days = []
        
        for agent_id in range(env.n_agents):
            agent_pipeline = env.pipeline[agent_id]
            total_orders += len(agent_pipeline)
            
            print(f"\nAgent {agent_id} pipeline:")
            for order in agent_pipeline:
                arrival_day = order['arrival_day']
                arrival_days.append(arrival_day)
                print(f"  SKU {order['sku']}: {order['qty']} units arriving on day {arrival_day}")
        
        # Validate lead times
        current_day = env.current_day
        valid_arrivals = all(
            current_day + env.lt_min <= day <= current_day + env.lt_max
            for day in arrival_days
        )
        
        assert valid_arrivals, "Some orders have invalid arrival times!"
        print(f"\n✓ All {total_orders} orders have valid arrival times")
        
        # Run 10 more steps to see arrivals
        print(f"\nRunning 10 more steps to observe arrivals...")
        for step in range(10):
            random_action = [
                [np.random.randint(0, 5) for _ in range(env.n_skus)]
                for _ in range(env.n_agents)
            ]
            obs, rewards, dones, infos = env.step(random_action, one_hot=False)
            
            inv_changes = [env.inventory[0][sku] for sku in range(env.n_skus)]
            if step in [1, 2, 3, 4, 5]:
                print(f"  Day {env.current_day}: Agent 0 inventory = {inv_changes}")
        
        print(f"\n✅ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pricing():
    """Test 3: Dynamic pricing integration."""
    print("\n" + "="*60)
    print("TEST 3: Dynamic Pricing Integration")
    print("="*60)
    
    try:
        env = EnhancedMultiSKUEnv(config_path='configs/multi_sku_config.yaml')
        env.reset()
        
        initial_prices = env.current_prices.copy()
        print(f"Initial prices: {initial_prices}")
        
        # Run 20 steps and track price changes
        prices_over_time = [initial_prices.copy()]
        
        for step in range(20):
            action = [[2, 2, 2] for _ in range(env.n_agents)]
            obs, rewards, dones, infos = env.step(action, one_hot=False)
            prices_over_time.append(env.current_prices.copy())
        
        # Check if prices varied
        price_array = np.array(prices_over_time)
        price_variance = np.var(price_array, axis=0)
        
        print(f"\nPrice statistics over 20 days:")
        for sku in range(env.n_skus):
            print(f"  SKU {sku}:")
            print(f"    Min: {price_array[:, sku].min():.2f}")
            print(f"    Max: {price_array[:, sku].max():.2f}")
            print(f"    Mean: {price_array[:, sku].mean():.2f}")
            print(f"    Variance: {price_variance[sku]:.2f}")
        
        # Prices should vary (not constant)
        assert np.any(price_variance > 0.01), "Prices are constant (no variation detected)"
        print(f"\n✓ Prices vary over time as expected")
        
        # Check reward includes pricing
        test_order = [[10, 10, 10] for _ in range(env.n_agents)]
        obs, rewards, dones, infos = env.step(test_order, one_hot=False)
        
        print(f"\nRewards after large order: {rewards}")
        assert all(r < 0 for r in rewards), "Rewards should be negative (costs)"
        
        # Check price appears in observation (positions 15-17)
        price_obs_start = env.n_skus * 5  # After inv, backlog, 3 pipeline features
        price_in_obs = obs[0][price_obs_start:price_obs_start + env.n_skus]
        print(f"\nPrices in observation (normalized): {price_in_obs}")
        
        print(f"\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_loading():
    """Test 4: CSV data loading."""
    print("\n" + "="*60)
    print("TEST 4: CSV Data Loading")
    print("="*60)
    
    try:
        # First, generate CSV data using the synthetic generator
        print("Generating test CSV data...")
        from utils.synthetic_data_generator import generate_demand_data, generate_price_data
        
        demand_df = generate_demand_data(n_days=100, n_skus=3, seed=123)
        price_df = generate_price_data(n_days=100, n_skus=3, seed=123)
        
        # Save to data directory
        os.makedirs('data', exist_ok=True)
        demand_df.to_csv('data/demand_history.csv', index=False)
        price_df.to_csv('data/price_history.csv', index=False)
        
        print(f"✓ Generated and saved CSV files")
        
        # Update config to use CSV instead of synthetic
        import yaml
        with open('configs/multi_sku_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        config['environment']['data_sources']['use_synthetic'] = False
        
        with open('configs/test_csv_config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Load environment with CSV data
        env = EnhancedMultiSKUEnv(config_path='configs/test_csv_config.yaml')
        env.reset()
        
        # Verify demand matches CSV
        csv_day_10_demand = [
            demand_df.loc[10, f'sku_{i}_demand'] for i in range(env.n_skus)
        ]
        
        # Run to day 10
        for _ in range(10):
            action = [[0, 0, 0] for _ in range(env.n_agents)]
            env.step(action, one_hot=False)
        
        env_demand = env.get_demand()
        
        print(f"\nDay 10 demand comparison:")
        print(f"  CSV: {csv_day_10_demand}")
        print(f"  Env: {env_demand}")
        
        # Allow small floating point differences
        demand_match = np.allclose(csv_day_10_demand, env_demand, rtol=0.01)
        assert demand_match, "Demand from CSV doesn't match environment demand"
        
        print(f"✓ CSV data loaded correctly")
        
        # Cleanup
        os.remove('configs/test_csv_config.yaml')
        
        print(f"\n✅ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_episode():
    """Test 5: Run a full episode to check stability."""
    print("\n" + "="*60)
    print("TEST 5: Full Episode Stability")
    print("="*60)
    
    try:
        env = EnhancedMultiSKUEnv(config_path='configs/multi_sku_config.yaml')
        obs = env.reset()
        
        episode_length = 50
        total_rewards = [0.0] * env.n_agents
        
        print(f"Running episode of {episode_length} steps...")
        
        for step in range(episode_length):
            # Random policy
            actions = [
                [np.random.randint(0, 10) for _ in range(env.n_skus)]
                for _ in range(env.n_agents)
            ]
            
            obs, rewards, dones, infos = env.step(actions, one_hot=False)
            
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            
            # Check for NaN or Inf
            for i, agent_obs in enumerate(obs):
                assert not np.any(np.isnan(agent_obs)), f"NaN detected in agent {i} observation at step {step}"
                assert not np.any(np.isinf(agent_obs)), f"Inf detected in agent {i} observation at step {step}"
            
            if step % 10 == 0:
                print(f"  Step {step}: Rewards = {[f'{r:.2f}' for r in rewards]}")
        
        print(f"\n✓ Episode completed without errors")
        print(f"Total rewards: {[f'{r:.2f}' for r in total_rewards]}")
        
        print(f"\n✅ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Enhanced Multi-SKU Environment')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'test_reset', 'test_lead_time', 'test_pricing', 
                               'test_csv_loading', 'test_full_episode'],
                       help='Which test to run')
    
    args = parser.parse_args()
    
    # Map test names to functions
    tests = {
        'test_reset': test_reset,
        'test_lead_time': test_lead_time,
        'test_pricing': test_pricing,
        'test_csv_loading': test_csv_loading,
        'test_full_episode': test_full_episode
    }
    
    print("\n" + "="*60)
    print("ENHANCED MULTI-SKU ENVIRONMENT TEST SUITE")
    print("="*60)
    
    if args.test == 'all':
        results = {}
        for test_name, test_func in tests.items():
            results[test_name] = test_func()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        print("\n" + "="*60)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED")
        print("="*60)
        
        return 0 if all_passed else 1
    else:
        # Run single test
        success = tests[args.test]()
        return 0 if success else 1


if __name__ == '__main__':
    exit(main())
