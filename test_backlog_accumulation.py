#!/usr/bin/env python
"""Quick test to see actual backlog values during first episode"""

import numpy as np
from envs.multi_dc_env import MultiDCInventoryEnv

env = MultiDCInventoryEnv()
obs = env.reset()

print("=" * 80)
print("TESTING BACKLOG ACCUMULATION")
print("=" * 80)

print("\n📊 COST PARAMETERS:")
print(f"Retailer backlog costs: {env.B_retailer}")
print(f"Retailer holding costs: {env.H_retailer}")
print(f"DC backlog costs: {env.B_dc}")

# Run 30 days with moderate actions
for day in range(30):
    actions = {
        0: np.array([15.0, 12.0, 18.0, 0.0, 0.0, 0.0]),  # DC 0
        1: np.array([15.0, 12.0, 18.0, 0.0, 0.0, 0.0]),  # DC 1
        2: np.array([10.0, 8.0, 12.0, 10.0, 8.0, 12.0]), # Retailer 0
        3: np.array([10.0, 8.0, 12.0, 10.0, 8.0, 12.0]), # Retailer 1
        4: np.array([10.0, 8.0, 12.0, 10.0, 8.0, 12.0]), # Retailer 2
    }
    
    obs, rewards, dones, infos = env.step(actions)
    
    if day % 5 == 0:
        print(f"\n📅 Day {day + 1}:")
        print(f"  DC 0 - Backlog: {env.backlog[0]}")
        print(f"  DC 1 - Backlog: {env.backlog[1]}")
        print(f"  Retailer 0 - Backlog: {env.backlog[2]}")
        print(f"  Retailer 1 - Backlog: {env.backlog[3]}")
        print(f"  Retailer 2 - Backlog: {env.backlog[4]}")
        print(f"  Rewards: DC0={rewards[0]:.1f}, DC1={rewards[1]:.1f}, R0={rewards[2]:.1f}, R1={rewards[3]:.1f}, R2={rewards[4]:.1f}")

# Calculate reward from backlog
print("\n" + "=" * 80)
print("FINAL STATE (Day 30):")
print("=" * 80)
for i, ret_id in enumerate(env.retailer_ids):
    total_backlog_cost = 0
    for sku in range(env.n_skus):
        cost = env.B_retailer[i][sku] * env.backlog[ret_id][sku]
        total_backlog_cost += cost
        print(f"Retailer {i} SKU {sku}: backlog={env.backlog[ret_id][sku]:.1f}, rate={env.B_retailer[i][sku]}, cost={cost:.1f}")
    print(f"  → Total backlog cost for Retailer {i}: {total_backlog_cost:.1f}")
    print()

print("\n🔍 ANALYSIS:")
print("If Retailer rewards are around -5000, backlog must be HUGE!")
print("Example: If reward = -5000 and backlog_rate = 0.6, then backlog ≈ 8333 units!")
print("This suggests demand is MUCH HIGHER than supply capacity.")
