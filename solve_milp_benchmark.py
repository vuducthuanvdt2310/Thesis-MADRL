import pulp
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
from envs.multi_dc_env import MultiDCInventoryEnv

def solve_pure_milp(horizon=50, time_limit_sec=10000):
    # ==========================================
    # MANUALLY EDIT STARTING INVENTORY HERE
    # ==========================================
    INIT_INV_DC = 200       # Starting inventory for DCs (per SKU)
    INIT_INV_RETAILER = 15  # Starting inventory for Retailers (per SKU)
    INIT_BACKLOG = 0        # Starting backlog for Retailers (per SKU)
    
    # MANUALLY EDIT DEMAND DISTRIBUTION HERE
    DEMAND_MEAN = [2.41, 1.99, 1.89]  # Mean demand per day for [SKU_0, SKU_1, SKU_2]
    DEMAND_STD = [2, 1.4, 2.2]   # Standard deviation for [SKU_0, SKU_1, SKU_2]
    # ==========================================

    # 1. Load Environment and Config
    env = MultiDCInventoryEnv()
    with open('configs/multi_dc_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    n_skus = config['environment']['n_skus']
    n_retailers = config['environment']['n_retailers']
    n_dcs = 2
    
    # 2. Stochastic Data Realization (Captured as parameters for the MILP)
    np.random.seed(42)
    # Realized Demand: Normal(mean, std)
    real_demand = {}
    for r in range(n_retailers):
        real_demand[r] = {t: np.maximum(0, np.random.normal(DEMAND_MEAN, DEMAND_STD)).astype(int) 
                          for t in range(horizon)}
    
    # Realized Lead Times: Uniform(min, max)
    lt_s_dc = np.random.randint(config['environment']['lead_time']['supplier_to_dc']['min'], 
                                config['environment']['lead_time']['supplier_to_dc']['max'] + 1, size=(n_dcs, horizon))
    lt_dc_r = 1 # Fixed in config
    
    # Realized Prices: Random Walk with Mean Reversion
    prices = np.zeros((n_skus, horizon))
    curr_p = np.array(config['pricing']['base_price'], dtype=float)
    for t in range(horizon):
        prices[:, t] = curr_p
        change = np.random.normal(0, config['pricing']['volatility'] * curr_p)
        curr_p += change
        curr_p += 0.1 * (np.array(config['pricing']['base_price']) - curr_p) # Mean reversion
        curr_p = np.clip(curr_p, config['pricing']['min_price'], config['pricing']['max_price'])

    # 3. Define MILP Model
    prob = pulp.LpProblem("Pure_System_Optimization", pulp.LpMinimize)
    
    # --- Decision Variables ---
    # Agents: 0,1 are DCs | 2..16 are Retailers
    agents = list(range(n_dcs + n_retailers))
    x = pulp.LpVariable.dicts("ord", (agents, range(n_skus), range(horizon)), 0, 500, pulp.LpInteger)
    y = pulp.LpVariable.dicts("bin", (agents, range(n_skus), range(horizon)), 0, 1, pulp.LpBinary)
    inv = pulp.LpVariable.dicts("inv", (agents, range(n_skus), range(horizon)), 0, 5000, pulp.LpInteger)
    back = pulp.LpVariable.dicts("bak", (agents, range(n_skus), range(horizon)), 0, 5000, pulp.LpInteger)
    ship = pulp.LpVariable.dicts("shp", (range(n_dcs), range(n_retailers), range(n_skus), range(horizon)), 0, 500, pulp.LpInteger)
    
    # DC Owed to Retailer (DC Backlog)
    dc_owed = pulp.LpVariable.dicts("dc_owed", (range(n_dcs), range(n_retailers), range(n_skus), range(horizon)), 0, 5000, pulp.LpInteger)
    
    # Fill Rate Tracker: 1 if demand fully met from stock today
    is_filled = pulp.LpVariable.dicts("fld", (range(n_retailers), range(n_skus), range(horizon)), 0, 1, pulp.LpBinary)

    # 4. Constraints
    for s in range(n_skus):
        # --- DC Math ---
        for d in range(n_dcs):
            for t in range(horizon):
                prob += x[d][s][t] <= 500 * y[d][s][t] # Order link
                arrival = pulp.lpSum([x[d][s][tp] for tp in range(t) if tp + lt_s_dc[d][tp] == t])
                assigned_r = config['dc_assignments'][f'dc_{d}']
                outflow = pulp.lpSum([ship[d][r][s][t] for r in assigned_r])
                prev_i = INIT_INV_DC if t == 0 else inv[d][s][t-1]
                prob += inv[d][s][t] == prev_i + arrival - outflow # Flow balance

        # --- Retailer Math ---
        for r in range(n_retailers):
            a_dc = 0 if r in config['dc_assignments']['dc_0'] else 1
            r_id = r + n_dcs
            for t in range(horizon):
                prob += x[r_id][s][t] <= 100 * y[r_id][s][t]
                
                # Link shipment from DC to retailer's order via dc_owed
                prev_owed = 0 if t == 0 else dc_owed[a_dc][r][s][t-1]
                prob += dc_owed[a_dc][r][s][t] == prev_owed + x[r_id][s][t] - ship[a_dc][r][s][t]
                prob += ship[a_dc][r][s][t] <= prev_owed + x[r_id][s][t]
                
                arrival = ship[a_dc][r][s][t-lt_dc_r] if t >= lt_dc_r else 0
                dem = real_demand[r][t][s]
                prev_i = INIT_INV_RETAILER if t == 0 else inv[r_id][s][t-1]
                prev_b = INIT_BACKLOG if t == 0 else back[r_id][s][t-1]
                
                # Flow balance: I - B = I_prev - B_prev + A - D
                prob += inv[r_id][s][t] - back[r_id][s][t] == prev_i - prev_b + arrival - dem
                
                # Fill Rate Logic: is_filled = 1 ONLY IF (prev_i + arrival) >= dem
                # Linearization: (prev_i + arrival) >= dem * is_filled
                prob += prev_i + arrival >= dem * is_filled[r][s][t]

    # 5. Objective (Multi-Objective: Minimize Cost + Maximize Fill Rate)
    obj = []
    w_h, w_b, w_o = config['rewards']['holding_weight'], config['rewards']['backlog_weight'], config['rewards']['ordering_weight']
    
    # Multi-Objective Weight: How much a 'Filled Order' is worth compared to costs
    # A value of 20.0 strongly prioritizes maintaining a high service level
    fill_rate_reward = 20.0 

    for t in range(horizon):
        for s in range(n_skus):
            for d in range(n_dcs):
                # DC Costs
                h = inv[d][s][t] * env.H_dc[d][s] * w_h
                f = y[d][s][t] * env.C_fixed_dc[d][s] * w_o
                v = x[d][s][t] * prices[s, t] * w_o
                
                # DC Backlog cost
                assigned_r = config['dc_assignments'][f'dc_{d}']
                b_dc = sum(dc_owed[d][r][s][t] for r in assigned_r) * env.B_dc[d][s] * w_b
                
                obj.append(h + b_dc + f + v)
            
            for r in range(n_retailers):
                # Retailer Costs
                r_id = r + n_dcs
                h = inv[r_id][s][t] * env.H_retailer[r][s] * w_h
                b = back[r_id][s][t] * env.B_retailer[r][s] * w_b
                f = y[r_id][s][t] * env.C_fixed_retailer[r][s] * w_o
                assigned_dc = 0 if r in config['dc_assignments']['dc_0'] else 1
                v = x[r_id][s][t] * env.C_var_retailer[r][assigned_dc][s] * w_o
                
                # Multi-Objective: Reward for fulfilling demand from stock
                fill_bonus = is_filled[r][s][t] * fill_rate_reward
                
                obj.append(h + b + f + v - fill_bonus)

    prob += pulp.lpSum(obj)

    # 6. Solve and Print
    print(f"Solving Pure MILP for {horizon} days...")
    solver = pulp.HiGHS(msg=True, timeLimit=time_limit_sec, gapRel=0.0)
    prob.solve(solver)
    
    # 7. Final Metrics
    if prob.status == 1:
        total_cost = pulp.value(prob.objective)
        
        # Calculate Fill Rate only for days with actual demand (matching environment logic)
        total_filled = 0
        total_orders = 0
        for r in range(n_retailers):
            for s in range(n_skus):
                for t in range(horizon):
                    if real_demand[r][t][s] > 0:
                        total_orders += 1
                        val = pulp.value(is_filled[r][s][t])
                        if val is not None and val > 0.5:
                            total_filled += 1
        
        fill_rate = (total_filled / total_orders) * 100 if total_orders > 0 else 100.0
        
        print("\nExtracting trajectory data and saving reports...")
        
        # Data Extraction for Excel and CSV
        tot_holding = 0.0
        tot_backlog = 0.0
        tot_ordering = 0.0
        avg_inv_sum = 0.0
        
        rows = []
        cum_orders_placed = {aid: 0 for aid in agents}
        cum_orders_from_stock = {aid: 0 for aid in agents}

        for t in range(horizon):
            for aid in agents:
                is_dc = aid < 2
                r_idx = aid - 2 if not is_dc else -1
                
                step_inv = sum((pulp.value(inv[aid][s][t]) or 0) for s in range(n_skus))
                avg_inv_sum += step_inv
                
                step_backlog = 0
                if not is_dc:
                    step_backlog = sum((pulp.value(back[aid][s][t]) or 0) for s in range(n_skus))
                else:
                    assigned_r = config['dc_assignments'][f'dc_{aid}']
                    step_backlog = sum((pulp.value(dc_owed[aid][r_id][s][t]) or 0) for s in range(n_skus) for r_id in assigned_r)
                
                step_placed = 0
                step_from_stock = 0
                
                row = {
                    'step': t + 1,
                    'agent_id': aid,
                    'agent': f'{"DC" if is_dc else "R"}_{aid}',
                    'inv': step_inv,
                    'backlog': step_backlog,
                }
                
                step_cost = 0.0
                for s in range(n_skus):
                    order_val = (pulp.value(x[aid][s][t]) or 0)
                    row[f'order_{s}'] = order_val
                    
                    if is_dc:
                        assigned_r = config['dc_assignments'][f'dc_{aid}']
                        dem_val = sum((pulp.value(x[r_id + n_dcs][s][t]) or 0) for r_id in assigned_r)
                        row[f'demand_{s}'] = dem_val
                        
                        hc = (pulp.value(inv[aid][s][t]) or 0) * env.H_dc[aid][s]
                        fc = (pulp.value(y[aid][s][t]) or 0) * env.C_fixed_dc[aid][s]
                        vc = order_val * prices[s, t]
                        
                        # DC Backlog extraction
                        bc_dc = sum((pulp.value(dc_owed[aid][r_id][s][t]) or 0) for r_id in assigned_r) * env.B_dc[aid][s]
                        
                        step_cost += (hc * w_h) + (bc_dc * w_b) + (fc + vc) * w_o
                        
                        tot_holding += hc
                        tot_ordering += (fc + vc)
                        tot_backlog += bc_dc
                    else:
                        dem_val = real_demand[r_idx][t][s]
                        row[f'demand_{s}'] = dem_val
                        if dem_val > 0:
                            step_placed += 1
                            val = pulp.value(is_filled[r_idx][s][t])
                            if val is not None and val > 0.5:
                                step_from_stock += 1
                                
                        hc = (pulp.value(inv[aid][s][t]) or 0) * env.H_retailer[r_idx][s]
                        bc = (pulp.value(back[aid][s][t]) or 0) * env.B_retailer[r_idx][s]
                        fc = (pulp.value(y[aid][s][t]) or 0) * env.C_fixed_retailer[r_idx][s]
                        assigned_dc = 0 if r_idx in config['dc_assignments']['dc_0'] else 1
                        vc = order_val * env.C_var_retailer[r_idx][assigned_dc][s]
                        
                        step_cost += (hc * w_h) + (bc * w_b) + (fc + vc) * w_o
                        
                        tot_holding += hc
                        tot_backlog += bc
                        tot_ordering += (fc + vc)
                
                row['reward'] = -step_cost
                row['orders_placed'] = step_placed
                row['orders_from_stock'] = step_from_stock
                row['step_sl_pct'] = round((step_from_stock / step_placed) * 100, 1) if step_placed > 0 else 100.0
                
                cum_orders_placed[aid] += step_placed
                cum_orders_from_stock[aid] += step_from_stock
                row['cum_placed'] = cum_orders_placed[aid]
                row['cum_from_stock'] = cum_orders_from_stock[aid]
                row['cum_sl_pct'] = round((cum_orders_from_stock[aid] / cum_orders_placed[aid]) * 100, 1) if cum_orders_placed[aid] > 0 else 100.0
                
                rows.append(row)
        
        # Save Outputs
        save_dir = Path("evaluation_results/milp_benchmark")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Excel Trajectory
        df = pd.DataFrame(rows)
        excel_path = save_dir / f"step_trajectory_milp_{horizon}d.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            pd.DataFrame([{'note': 'MILP Benchmark'}]).to_excel(writer, sheet_name='Scales', index=False)
        print(f"[OK] Saved step trajectory to: {excel_path}")
        
        # 2. Save CSV Metrics
        import csv
        csv_path = save_dir / f"results_milp_{horizon}d.csv"
        lost_sales = sum(cum_orders_placed[aid] - cum_orders_from_stock[aid] for aid in range(n_dcs, len(agents)))
        avg_inv = avg_inv_sum / (horizon * len(agents))
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                             'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'])
            
            true_total_cost = tot_holding + tot_backlog + tot_ordering
            writer.writerow([
                1,
                round(true_total_cost, 4),
                round(fill_rate, 4),
                round(lost_sales, 4),
                round(avg_inv, 4),
                round(tot_holding, 4),
                round(tot_backlog, 4),
                round(tot_ordering, 4)
            ])
        print(f"[OK] Saved metrics to: {csv_path}")

        print("\n" + "="*30)
        print(f"MILP RESULT ({horizon} DAYS)")
        print(f"Status: {pulp.LpStatus[prob.status]}")
        print(f"Objective Value (w/ penalties & fill bonus): {total_cost:.2f}")
        print(f"Normalized Daily Cost: {true_total_cost / config['environment']['max_days']:.4f}")
        print(f"System Fill Rate: {fill_rate:.2f}%")
        print("-" * 30)
        print("COST BREAKDOWN (True Environment Costs):")
        print(f"  Total Cost    : {true_total_cost:.2f}")
        print(f"  Holding Cost  : {tot_holding:.2f}")
        print(f"  Backlog Cost  : {tot_backlog:.2f}")
        print(f"  Ordering Cost : {tot_ordering:.2f}")
        print("="*30)

if __name__ == "__main__":
    solve_pure_milp(horizon=90)
