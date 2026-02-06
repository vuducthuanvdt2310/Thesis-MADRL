import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse

def plot_progress(results_dir):
    """
    Plots the training progress from progress.csv files found in the results directory.
    """
    print(f"Searching for progress.csv in: {results_dir}")
    
    found_files = []
    
    # Walk through directory to find all progress.csv files
    for root, dirs, files in os.walk(results_dir):
        if "progress.csv" in files:
            full_path = os.path.join(root, "progress.csv")
            # Get the experiment name (folder name) for the legend
            label = os.path.basename(root) 
            found_files.append((full_path, label))
            
    if not found_files:
        print("No progress.csv files found! Make sure you have run the training with the updated runner.")
        print("Also check that your path is correct.")
        return

    plt.figure(figsize=(10, 6))
    
    for file_path, label in found_files:
        try:
            df = pd.read_csv(file_path)
            
            # Determine X-axis (Steps preferred, Episode fallback)
            if 'steps' in df.columns:
                x_data = df['steps']
                x_label = "Total Environment Steps"
            else:
                x_data = df['episode']
                x_label = "Episode"

            plt.plot(x_data, df['reward'], marker='o', label=label)
            
            # Annotate the max reward
            max_reward = df['reward'].max()
            max_idx = df['reward'].idxmax()
            max_x = x_data.iloc[max_idx]
            
            plt.annotate(f'Max: {max_reward:.2f}', 
                         xy=(max_x, max_reward), 
                         xytext=(max_x, max_reward + 5),
                         arrowprops=dict(facecolor='black', shrink=0.05))
                         
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    plt.title(f"Training Progress ({x_label} vs Reward)")
    plt.xlabel(x_label)
    plt.ylabel("Average Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = "training_progress.png"
    plt.savefig(output_file)
    print(f"\nSuccess! Graph saved to: {os.path.abspath(output_file)}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training progress from CSV logs.")
    parser.add_argument("--dir", type=str, default="results", help="Path to results directory")
    args = parser.parse_args()
    
    plot_progress(args.dir)
