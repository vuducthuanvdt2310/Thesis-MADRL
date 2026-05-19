import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
data_path = 'data/price_history.csv'
df = pd.read_csv(data_path)

skus = ['sku_0_price', 'sku_1_price', 'sku_2_price']

# Create output directory for plots if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

for i, sku in enumerate(skus):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get the price data
    data = df[sku]
    
    # Panel 1: Random walk trajectory
    ax1.plot(df['day'], data, color='coral', linewidth=1.5, label='Price')
    ax1.set_title(f'{sku} Random Walk Trajectory')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.5)
    ax1.legend()
    
    # Panel 2: Distribution of price
    ax2.hist(data, bins=30, density=True, alpha=0.7, color='coral', edgecolor='black', label='Price Histogram')
    ax2.set_title(f'{sku} Price Distribution')
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Density')
    ax2.grid(axis='y', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{sku}_random_walk_distribution.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

print(f"Plots saved successfully in the '{output_dir}' directory.")
