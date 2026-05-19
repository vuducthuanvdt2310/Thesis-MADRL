import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Load data
data_path = 'data/demand_history.csv'
df = pd.read_csv(data_path)

skus = ['sku_0_demand', 'sku_1_demand', 'sku_2_demand']

# Create output directory for plots if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

for i, sku in enumerate(skus):
    plt.figure(figsize=(8, 6))
    
    # Get the demand data
    data = df[sku]
    
    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(data)
    
    # Plot the histogram
    # Using bins=range to center bins on integers since demand is discrete
    min_val = int(data.min())
    max_val = int(data.max())
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Demand Histogram')
    
    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label=f'Normal PDF\n$\mu={mu:.2f}$, $\sigma={std:.2f}$')
    
    # Add title and labels
    plt.title(f'Demand Distribution for {sku}')
    plt.xlabel('Demand Quantity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{sku}_normal_distribution.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

print(f"Plots saved successfully in the '{output_dir}' directory.")
