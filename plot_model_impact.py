import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def clean_cost(series):
    if series.dtype == 'object':
        return series.str.replace(' ', '').str.replace(',', '').astype(float)
    return series

def get_total_costs(filepath):
    try:
        df = pd.read_csv(filepath)
        holding = clean_cost(df['Total_Holding_Cost']).mean()
        backlog = clean_cost(df['Total_Backlog_Cost']).mean()
        ordering = clean_cost(df['Total_Ordering_Cost']).mean()
        return holding, backlog, ordering
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0, 0

def get_excess_stock_and_order_spikes(filepath):
    try:
        df = pd.read_excel(filepath)
        
        # Excess Stock Calculation
        demand_cols = [c for c in df.columns if 'demand' in c and 'norm' not in c]
        df['total_demand'] = df[demand_cols].sum(axis=1)
        df['excess_stock'] = np.maximum(0, df['inv'] - df['total_demand'])
        avg_excess_stock = df['excess_stock'].mean()
        
        # Maximum Order Spikes for DCs (Orders received from Retailers)
        # Assuming retailers place orders that DCs process. 
        # The sum of orders_placed by all retailers per step is the daily order quantity DCs process.
        r_df = df[~df['agent'].str.startswith('DC')]
        daily_orders = r_df.groupby('step')['orders_placed'].sum()
        max_daily_order = daily_orders.max()
        order_volatility = daily_orders.std()
        
        return avg_excess_stock, max_daily_order, order_volatility
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0, 0

def main():
    base_dir = r"d:\thuan\thesis\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management"
    results_dir = os.path.join(base_dir, "evaluation_results")
    
    # Paths
    paths_csv = {
        "GNN-HAPPO": os.path.join(results_dir, "eval_gnn", "results_gnn_happo.csv"),
        "Standard HAPPO": os.path.join(results_dir, "eval_base", "results_standard_happo.csv"),
        "MAPPO": os.path.join(results_dir, "eval_mappo", "results_mappo.csv"),
        "(s,S) Heuristic": os.path.join(results_dir, "basestock_v1", "results_ss_heuristic.csv")
    }
    
    paths_excel = {
        "GNN-HAPPO": os.path.join(results_dir, "eval_gnn", "step_trajectory_ep1.xlsx"),
        "Standard HAPPO": os.path.join(results_dir, "eval_base", "step_trajectory_ep1.xlsx"),
        "MAPPO": os.path.join(results_dir, "eval_mappo", "step_trajectory_ep1.xlsx"),
        "(s,S) Heuristic": os.path.join(results_dir, "basestock_v1", "step_trajectory_ep1.xlsx")
    }
    
    # 1. Stacked Column Chart Total Cost
    models = list(paths_csv.keys())
    holding_costs = []
    backlog_costs = []
    ordering_costs = []
    
    for model in models:
        h, b, o = get_total_costs(paths_csv[model])
        holding_costs.append(h)
        backlog_costs.append(b)
        ordering_costs.append(o)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p1 = ax.bar(models, holding_costs, label='Holding Cost')
    p2 = ax.bar(models, backlog_costs, bottom=holding_costs, label='Backlog Cost')
    p3 = ax.bar(models, ordering_costs, bottom=np.array(holding_costs) + np.array(backlog_costs), label='Ordering Cost')
    
    ax.set_ylabel("Cost ('000 VND)")
    ax.set_title('Cost Breakdown across Models')
    ax.legend()
    
    plt.savefig(os.path.join(results_dir, "cost_breakdown_chart.png"))
    plt.close()
    print("Saved cost breakdown chart to cost_breakdown_chart.png")

    # 2 & 3. Data extraction for Excess Stock and Order Spikes
    excel_models = list(paths_excel.keys())
    excess_stocks = []
    max_orders = []
    volatilities = []
    
    for model in excel_models:
        e, m, v = get_excess_stock_and_order_spikes(paths_excel[model])
        excess_stocks.append(e)
        max_orders.append(m)
        volatilities.append(v)
        
    # 2. Horizontal Bar Chart for Average Excess Stock
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(excel_models))
    ax.barh(y_pos, excess_stocks, align='center', color=['skyblue', 'lightgreen', 'orange', 'salmon'])
    ax.set_yticks(y_pos, labels=excel_models)
    ax.set_xlabel('Average Excess Stock Volume')
    ax.set_title('Average Excess Stock Volume Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "excess_stock_chart.png"))
    plt.close()
    print("Saved excess stock chart to excess_stock_chart.png")
    
    # 3. Clustered Bar Chart for Maximum Order Spikes and Volatility
    x = np.arange(len(excel_models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, max_orders, width, label='Max Daily Orders Processed')
    rects2 = ax.bar(x + width/2, volatilities, width, label='Order Volatility (Std Dev)')
    
    ax.set_ylabel('Quantity')
    ax.set_title('Order Spikes and Volatility')
    ax.set_xticks(x, excel_models)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "order_spikes_chart.png"))
    plt.close()
    print("Saved order spikes chart to order_spikes_chart.png")

if __name__ == "__main__":
    main()
