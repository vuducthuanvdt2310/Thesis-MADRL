"""
Synthetic Data Generator for Multi-SKU Multi-Echelon Inventory Environment

Standalone utility to pre-generate CSV files for demand and pricing data.
Can be used to create reproducible datasets for training and testing.

Usage:
    python utils/synthetic_data_generator.py --output_dir data --days 365 --n_skus 3
"""

import numpy as np
import pandas as pd
import argparse
import os
from typing import List, Tuple


def generate_demand_data(
    n_days: int,
    n_skus: int,
    base_demands: List[float] = None,
    variances: List[float] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic demand data using random walk with mean reversion.
    
    Args:
        n_days: Number of days to generate
        n_skus: Number of SKUs
        base_demands: List of base demand values per SKU
        variances: List of demand volatilities per SKU
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with demand data
    """
    np.random.seed(seed)
    
    # Default parameters if not provided
    if base_demands is None:
        base_demands = [25, 15, 10]  # Different base demands
    if variances is None:
        variances = [15, 10, 8]  # Different volatilities
    
    data = {'day': list(range(n_days))}
    
    for sku in range(n_skus):
        demand_series = []
        current_demand = base_demands[sku % len(base_demands)]
        
        for day in range(n_days):
            # Random noise
            noise = np.random.normal(0, variances[sku % len(variances)])
            
            # Mean reversion (10% pull toward base demand)
            mean_reversion = 0.1 * (base_demands[sku % len(base_demands)] - current_demand)
            
            # Update demand (ensure non-negative)
            current_demand = max(0, current_demand + noise + mean_reversion)
            
            # Add occasional demand spikes (5% chance)
            if np.random.random() < 0.05:
                current_demand *= 1.5
            
            demand_series.append(int(current_demand))
        
        data[f'sku_{sku}_demand'] = demand_series
    
    df = pd.DataFrame(data)
    
    print(f"Generated demand data:")
    print(f"  - Days: {n_days}")
    print(f"  - SKUs: {n_skus}")
    print(f"  - Mean demands: {[df[f'sku_{i}_demand'].mean() for i in range(n_skus)]}")
    
    return df


def generate_price_data(
    n_days: int,
    n_skus: int,
    base_prices: List[float] = None,
    min_prices: List[float] = None,
    max_prices: List[float] = None,
    volatility: float = 0.1,
    seasonal_amplitude: float = 0.2,
    seasonal_period: int = 90,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic pricing data with seasonal patterns and noise.
    
    Formula: price = base_price * (1 + seasonal_factor) * (1 + noise)
    
    Args:
        n_days: Number of days to generate
        n_skus: Number of SKUs
        base_prices: List of base prices per SKU
        min_prices: List of minimum allowed prices
        max_prices: List of maximum allowed prices
        volatility: Standard deviation of price noise
        seasonal_amplitude: Amplitude of seasonal variation (0.2 = ±20%)
        seasonal_period: Period of seasonal cycle in days
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with price data
    """
    np.random.seed(seed + 100)  # Different seed from demand
    
    # Default parameters
    if base_prices is None:
        base_prices = [10.0, 7.0, 15.0]
    if min_prices is None:
        min_prices = [5.0, 3.0, 8.0]
    if max_prices is None:
        max_prices = [20.0, 15.0, 30.0]
    
    data = {'day': list(range(n_days))}
    
    for sku in range(n_skus):
        price_series = []
        base_price = base_prices[sku % len(base_prices)]
        min_price = min_prices[sku % len(min_prices)]
        max_price = max_prices[sku % len(max_prices)]
        
        for day in range(n_days):
            # Seasonal pattern (sine wave)
            seasonal_factor = seasonal_amplitude * np.sin(2 * np.pi * day / seasonal_period)
            
            # Random noise
            noise = np.random.normal(0, volatility)
            
            # Calculate price
            price = base_price * (1 + seasonal_factor) * (1 + noise)
            
            # Clip to bounds
            price = np.clip(price, min_price, max_price)
            
            price_series.append(round(price, 2))
        
        data[f'sku_{sku}_price'] = price_series
    
    df = pd.DataFrame(data)
    
    print(f"Generated price data:")
    print(f"  - Days: {n_days}")
    print(f"  - SKUs: {n_skus}")
    print(f"  - Mean prices: {[df[f'sku_{i}_price'].mean() for i in range(n_skus)]}")
    
    return df


def create_train_test_split(
    demand_df: pd.DataFrame,
    price_df: pd.DataFrame,
    split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        demand_df: Full demand DataFrame
        price_df: Full price DataFrame
        split_ratio: Fraction of data to use for training
    
    Returns:
        train_demand, test_demand, train_price, test_price
    """
    n_days = len(demand_df)
    split_idx = int(n_days * split_ratio)
    
    train_demand = demand_df[:split_idx].copy()
    test_demand = demand_df[split_idx:].copy()
    
    train_price = price_df[:split_idx].copy()
    test_price = price_df[split_idx:].copy()
    
    # Reset day indices for test set
    test_demand['day'] = range(len(test_demand))
    test_price['day'] = range(len(test_price))
    
    print(f"Split data: {split_idx} train days, {n_days - split_idx} test days")
    
    return train_demand, test_demand, train_price, test_price


def main():
    """Main function to generate and save synthetic data."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic demand and price data for multi-SKU environment'
    )
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save CSV files')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days to generate')
    parser.add_argument('--n_skus', type=int, default=3,
                        help='Number of SKUs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--split', action='store_true',
                        help='Create train/test split')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Synthetic Data Generator")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate data
    print("\n[1/2] Generating demand data...")
    demand_df = generate_demand_data(
        n_days=args.days,
        n_skus=args.n_skus,
        seed=args.seed
    )
    
    print("\n[2/2] Generating price data...")
    price_df = generate_price_data(
        n_days=args.days,
        n_skus=args.n_skus,
        seed=args.seed
    )
    
    # Save data
    if args.split:
        print("\nCreating train/test split...")
        train_demand, test_demand, train_price, test_price = create_train_test_split(
            demand_df, price_df
        )
        
        # Save training data
        train_dir = os.path.join(args.output_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        train_demand.to_csv(os.path.join(train_dir, 'demand_history.csv'), index=False)
        train_price.to_csv(os.path.join(train_dir, 'price_history.csv'), index=False)
        
        # Save testing data
        test_dir = os.path.join(args.output_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        test_demand.to_csv(os.path.join(test_dir, 'demand_history.csv'), index=False)
        test_price.to_csv(os.path.join(test_dir, 'price_history.csv'), index=False)
        
        print(f"\n✓ Data saved to {args.output_dir}/train/ and {args.output_dir}/test/")
    else:
        demand_path = os.path.join(args.output_dir, 'demand_history.csv')
        price_path = os.path.join(args.output_dir, 'price_history.csv')
        
        demand_df.to_csv(demand_path, index=False)
        price_df.to_csv(price_path, index=False)
        
        print(f"\n✓ Data saved:")
        print(f"  - {demand_path}")
        print(f"  - {price_path}")
    
    # Display sample data
    print("\n--- Sample Demand Data (first 5 days) ---")
    print(demand_df.head())
    
    print("\n--- Sample Price Data (first 5 days) ---")
    print(price_df.head())
    
    print("\n✓ Data generation completed successfully!")


if __name__ == '__main__':
    main()
