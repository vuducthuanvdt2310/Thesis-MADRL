import pulp
import numpy as np
import yaml
from envs.multi_dc_env import MultiDCInventoryEnv

def solve_pure_milp(horizon=90, time_limit_sec=10000):
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
        real_demand[r] = {t: np.maximum(0, np.random.normal(config['demand']['mean'], config['demand']['std'])).astype(int) 
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
    
    # Fill Rate Tracker: 1 if demand fully met from stock today
    is_filled = pulp.LpVariable.dicts("fld", (range(n_retailers), range(n_skus), range(horizon)), 0, 1, pulp.LpBinary)

    # Penalty Helpers
    dc_exc1 = pulp.LpVariable.dicts("de1", (range(n_dcs), range(n_skus), range(horizon)), 0, None)
    dc_exc2 = pulp.LpVariable.dicts("de2", (range(n_dcs), range(n_skus), range(horizon)), 0, None)
    ret_ss_s = pulp.LpVariable.dicts("rss", (range(n_retailers), range(n_skus), range(horizon)), 0, None)
    ret_exc = pulp.LpVariable.dicts("rex", (range(n_retailers), range(n_skus), range(horizon)), 0, None)

    # 4. Constraints
    for s in range(n_skus):
        # --- DC Math ---
        for d in range(n_dcs):
            target_dc = config['rewards']['target_stock_days_dc'] * (n_retailers/n_dcs) * config['demand']['mean'][s]
            for t in range(horizon):
                prob += x[d][s][t] <= 500 * y[d][s][t] # Order link
                arrival = pulp.lpSum([x[d][s][tp] for tp in range(t) if tp + lt_s_dc[d][tp] == t])
                assigned_r = config['dc_assignments'][f'dc_{d}']
                outflow = pulp.lpSum([ship[d][r][s][t] for r in assigned_r])
                prev_i = 500 if t == 0 else inv[d][s][t-1]
                prob += inv[d][s][t] == prev_i + arrival - outflow # Flow balance
                
                # Piecewise Excess
                prob += inv[d][s][t] - target_dc <= dc_exc1[d][s][t] + dc_exc2[d][s][t]
                prob += dc_exc1[d][s][t] <= target_dc

        # --- Retailer Math ---
        for r in range(n_retailers):
            a_dc = 0 if r in config['dc_assignments']['dc_0'] else 1
            ss_th = config['constraints']['safety_stock_threshold'][r][s]
            target_r = config['rewards']['target_stock_days_retailer'] * config['demand']['mean'][s]
            r_id = r + n_dcs
            for t in range(horizon):
                prob += x[r_id][s][t] <= 100 * y[r_id][s][t]
                arrival = ship[a_dc][r][s][t-lt_dc_r] if t >= lt_dc_r else 0
                dem = real_demand[r][t][s]
                prev_i = 30 if t == 0 else inv[r_id][s][t-1]
                prev_b = 0 if t == 0 else back[r_id][s][t-1]
                
                # Flow balance: I - B = I_prev - B_prev + A - D
                prob += inv[r_id][s][t] - back[r_id][s][t] == prev_i - prev_b + arrival - dem
                
                # Fill Rate Logic: is_filled = 1 ONLY IF (prev_i + arrival) >= dem
                # Linearization: (prev_i + arrival) >= dem * is_filled
                prob += prev_i + arrival >= dem * is_filled[r][s][t]
                
                # Penalties
                prob += ss_th - inv[r_id][s][t] <= ret_ss_s[r][s][t]
                prob += inv[r_id][s][t] - target_r <= ret_exc[r][s][t]

    # 5. Objective (Multi-Objective: Minimize Cost + Maximize Fill Rate)
    obj = []
    w_h, w_b, w_o = config['rewards']['holding_weight'], config['rewards']['backlog_weight'], config['rewards']['ordering_weight']
    
    # Multi-Objective Weight: How much a 'Filled Order' is worth compared to costs
    # Higher value = Higher Service Level priority
    fill_rate_reward = 20.0 

    for t in range(horizon):
        for s in range(n_skus):
            for d in range(n_dcs):
                # DC Costs
                h = inv[d][s][t] * env.H_dc[d][s] * w_h
                f = y[d][s][t] * env.C_fixed_dc[d][s] * w_o
                v = x[d][s][t] * prices[s, t] * w_o
                ex = (dc_exc1[d][s][t] * config['rewards']['excess_penalty_dc'] + 
                      dc_exc2[d][s][t] * config['rewards']['excess_penalty_dc'] * 3.0)
                obj.append(h + f + v + ex)
            
            for r in range(n_retailers):
                # Retailer Costs
                r_id = r + n_dcs
                h = inv[r_id][s][t] * env.H_retailer[r][s] * w_h
                b = back[r_id][s][t] * env.B_retailer[r][s] * w_b
                f = y[r_id][s][t] * env.C_fixed_retailer[r][s] * w_o
                v = x[r_id][s][t] * env.C_var_retailer[r][0 if r in config['dc_assignments']['dc_0'] else 1][s] * w_o
                ss = ret_ss_s[r][s][t] * config['constraints']['safety_stock_penalty']
                ex = ret_exc[r][s][t] * config['rewards']['excess_penalty_retailer']
                
                # Revenue: units sold = demand - (backlog_today - backlog_yesterday)
                prev_b = 0 if t == 0 else back[r_id][s][t-1]
                sold = real_demand[r][t][s] - (back[r_id][s][t] - prev_b)
                rev = sold * config['rewards']['sale_revenue_retailer']
                
                # Fill Bonus: Incentive to meet orders from stock
                fill_bonus = is_filled[r][s][t] * fill_rate_reward
                
                obj.append(h + b + f + v + ss + ex - rev - fill_bonus)

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
        
        print("\n" + "="*30)
        print(f"MILP RESULT ({horizon} DAYS)")
        print(f"Status: {pulp.LpStatus[prob.status]}")
        print(f"Total Objective Cost: {total_cost:.2f}")
        print(f"Normalized Daily Cost: {total_cost / config['environment']['max_days']:.4f}")
        print(f"System Fill Rate: {fill_rate:.2f}%")
        print("="*30)

if __name__ == "__main__":
    solve_pure_milp(horizon=90)
