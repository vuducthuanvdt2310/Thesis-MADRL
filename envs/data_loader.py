"""
Data Loader for Multi-SKU Multi-Echelon Inventory Environment

This module handles loading historical demand and pricing data from CSV files,
with fallback to synthetic data generation.
"""

import numpy as np
import pandas as pd
import yaml
import os
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """
    Centralized data management for the enhanced multi-SKU environment.
    
    Responsibilities:
    - Load configuration from YAML
    - Load demand/price history from CSV or generate synthetic data
    - Provide demand and pricing data for each simulation day
    - Validate data integrity
    """
    
    def __init__(self, config_path: str = 'configs/multi_sku_config.yaml'):
        """
        Initialize data loader with configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract key parameters
        self.n_skus = self.config['environment']['n_skus']
        self.max_days = self.config['environment']['max_days']
        
        # Initialize data containers
        self.demand_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None
        
        # Load or generate data
        self._load_data()
    
    def _load_data(self):
        """Load data from CSV files or generate synthetic data."""
        use_synthetic = self.config['environment']['data_sources']['use_synthetic']
        
        if use_synthetic:
            print("[DataLoader] Using synthetic data generation...")
            self.demand_data = self._generate_synthetic_demand()
            self.price_data = self._generate_synthetic_prices()
        else:
            # Try to load from CSV
            demand_file = self.config['environment']['data_sources']['demand_file']
            price_file = self.config['environment']['data_sources']['price_file']
            
            try:
                print(f"[DataLoader] Loading demand from {demand_file}...")
                self.demand_data = pd.read_csv(demand_file)
                self._validate_demand_data()
                
                print(f"[DataLoader] Loading prices from {price_file}...")
                self.price_data = pd.read_csv(price_file)
                self._validate_price_data()
                
                print("[DataLoader] CSV data loaded successfully.")
            except FileNotFoundError as e:
                print(f"[DataLoader] CSV file not found: {e}")
                print("[DataLoader] Falling back to synthetic data...")
                self.demand_data = self._generate_synthetic_demand()
                self.price_data = self._generate_synthetic_prices()
    
    def _validate_demand_data(self):
        """Validate demand CSV has correct columns and sufficient data."""
        required_cols = ['day'] + [f'sku_{i}_demand' for i in range(self.n_skus)]
        
        for col in required_cols:
            if col not in self.demand_data.columns:
                raise ValueError(f"Missing required column in demand CSV: {col}")
        
        if len(self.demand_data) < self.max_days:
            raise ValueError(
                f"Demand CSV has {len(self.demand_data)} rows, "
                f"but simulation requires {self.max_days} days"
            )
    
    def _validate_price_data(self):
        """Validate price CSV has correct columns and sufficient data."""
        required_cols = ['day'] + [f'sku_{i}_price' for i in range(self.n_skus)]
        
        for col in required_cols:
            if col not in self.price_data.columns:
                raise ValueError(f"Missing required column in price CSV: {col}")
        
        if len(self.price_data) < self.max_days:
            raise ValueError(
                f"Price CSV has {len(self.price_data)} rows, "
                f"but simulation requires {self.max_days} days"
            )
    
    def _generate_synthetic_demand(self) -> pd.DataFrame:
        """
        Generate synthetic demand data using random process.
        
        Uses a simplified version of the Merton jump-diffusion process
        similar to the existing generator.py logic.
        
        Returns:
            DataFrame with columns: day, sku_0_demand, sku_1_demand, ...
        """
        data = {'day': list(range(self.max_days))}
        
        # Different demand characteristics per SKU
        base_demands = [25, 15, 10]  # Default base demands
        variances = [15, 10, 8]      # Different volatility per SKU
        
        for sku in range(self.n_skus):
            # Random walk with mean reversion
            demand_series = []
            current_demand = base_demands[sku % len(base_demands)]
            
            for day in range(self.max_days):
                # Add noise and mean reversion
                noise = np.random.normal(0, variances[sku % len(variances)])
                mean_reversion = 0.1 * (base_demands[sku % len(base_demands)] - current_demand)
                
                current_demand = max(0, current_demand + noise + mean_reversion)
                demand_series.append(int(current_demand))
            
            data[f'sku_{sku}_demand'] = demand_series
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_prices(self) -> pd.DataFrame:
        """
        Generate synthetic price data with seasonal patterns.
        
        Formula: price = base_price * (1 + seasonal_factor) * (1 + noise)
        
        Returns:
            DataFrame with columns: day, sku_0_price, sku_1_price, ...
        """
        data = {'day': list(range(self.max_days))}
        
        base_prices = self.config['pricing']['base_price']
        volatility = self.config['pricing']['volatility']
        
        for sku in range(self.n_skus):
            price_series = []
            base_price = base_prices[sku]
            
            for day in range(self.max_days):
                # Seasonal pattern (90-day cycle)
                seasonal_factor = 0.2 * np.sin(2 * np.pi * day / 90)
                
                # Random noise
                noise = np.random.normal(0, volatility)
                
                # Calculate price
                price = base_price * (1 + seasonal_factor) * (1 + noise)
                
                # Clip to min/max bounds
                min_price = self.config['pricing']['min_price'][sku]
                max_price = self.config['pricing']['max_price'][sku]
                price = np.clip(price, min_price, max_price)
                
                price_series.append(round(price, 2))
            
            data[f'sku_{sku}_price'] = price_series
        
        return pd.DataFrame(data)
    
    def get_demand(self, day: int) -> List[float]:
        """
        Get demand for all SKUs on a specific day.
        
        Args:
            day: Simulation day (0-indexed)
        
        Returns:
            List of demand quantities [sku_0_demand, sku_1_demand, ...]
        """
        if day >= len(self.demand_data):
            day = day % len(self.demand_data)  # Wrap around if simulation exceeds data
        
        return [
            float(self.demand_data.loc[day, f'sku_{i}_demand'])
            for i in range(self.n_skus)
        ]
    
    def get_prices(self, day: int) -> List[float]:
        """
        Get procurement prices for all SKUs on a specific day.
        
        Args:
            day: Simulation day (0-indexed)
        
        Returns:
            List of unit prices [sku_0_price, sku_1_price, ...]
        """
        if day >= len(self.price_data):
            day = day % len(self.price_data)  # Wrap around if simulation exceeds data
        
        return [
            float(self.price_data.loc[day, f'sku_{i}_price'])
            for i in range(self.n_skus)
        ]
    
    def export_to_csv(self, output_dir: str = 'data'):
        """
        Export current demand and price data to CSV files.
        
        Useful for saving generated synthetic data for reproducibility.
        
        Args:
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        demand_path = os.path.join(output_dir, 'demand_history.csv')
        price_path = os.path.join(output_dir, 'price_history.csv')
        
        self.demand_data.to_csv(demand_path, index=False)
        self.price_data.to_csv(price_path, index=False)
        
        print(f"[DataLoader] Data exported to {output_dir}/")
        print(f"  - {demand_path}")
        print(f"  - {price_path}")


def test_data_loader():
    """Simple test function for DataLoader."""
    print("="*60)
    print("Testing DataLoader with synthetic data...")
    print("="*60)
    
    # Create dummy config for testing
    test_config = {
        'environment': {
            'n_skus': 3,
            'max_days': 100,
            'data_sources': {
                'use_synthetic': True
            }
        },
        'pricing': {
            'base_price': [10.0, 7.0, 15.0],
            'min_price': [5.0, 3.0, 8.0],
            'max_price': [20.0, 15.0, 30.0],
            'volatility': 0.1
        }
    }
    
    # Save test config
    os.makedirs('configs', exist_ok=True)
    config_path = 'configs/test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    # Test loader
    loader = DataLoader(config_path)
    
    # Test data retrieval
    print("\n--- Day 0 Data ---")
    print(f"Demand: {loader.get_demand(0)}")
    print(f"Prices: {loader.get_prices(0)}")
    
    print("\n--- Day 50 Data ---")
    print(f"Demand: {loader.get_demand(50)}")
    print(f"Prices: {loader.get_prices(50)}")
    
    # Export data
    loader.export_to_csv('data')
    
    print("\n✓ DataLoader test completed successfully!")
    
    # Cleanup
    os.remove(config_path)


if __name__ == '__main__':
    test_data_loader()
