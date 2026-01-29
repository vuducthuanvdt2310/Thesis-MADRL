"""
Enhanced Multi-SKU Multi-Echelon Inventory Management Environment

This module extends the original net_2x3.py environment to support:
1. Multiple SKUs (products) per agent
2. Variable lead times (stochastic delivery)
3. Dynamic pricing with time-varying procurement costs
4. CSV data integration for demand and pricing

Key Design Choices:
- Vectorized "fat-agent" architecture: Each location manages all SKUs
- Tagged pipeline structure for variable lead time management
- Observable pricing in state space for agents to learn procurement timing
- Backward compatible interface with HAPPO training framework
"""

import numpy as np
import os
import yaml
from typing import List, Tuple, Dict, Optional
from .data_loader import DataLoader


class EnhancedMultiSKUEnv:
    """
    Multi-SKU Multi-Echelon Inventory Environment with Advanced Features
    
    This environment simulates a 3-echelon supply chain (Retailer, Distributor, Manufacturer)
    managing multiple products (SKUs) simultaneously with realistic constraints:
    - Variable lead times (Uniform random distribution)
    - Dynamic procurement pricing   
    - Fixed + variable ordering costs
    """
    
    def __init__(self, config_path: str = 'configs/multi_sku_config.yaml'):
        """
        Initialize the enhanced multi-SKU environment.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load data and configuration
        self.data_loader = DataLoader(config_path)
        self.config = self.data_loader.config
        
        # Environment parameters
        self.n_skus = self.config['environment']['n_skus']
        self.n_agents = self.config['environment']['n_agents']  # 3 echelons
        self.max_days = self.config['environment']['max_days']
        
        # Lead time parameters
        self.lt_min = self.config['environment']['lead_time']['min']
        self.lt_max = self.config['environment']['lead_time']['max']
        self.lt_distribution = self.config['environment']['lead_time']['distribution']
        
        # Cost matrices (n_agents × n_skus)
        self.H = np.array(self.config['costs']['holding_cost'], dtype=np.float32)
        self.B = np.array(self.config['costs']['backlog_cost'], dtype=np.float32)
        self.C_FIXED = np.array(self.config['costs']['fixed_order_cost'], dtype=np.float32)
        
        # Normalization bounds
        self.max_inventory = self.config['normalization']['max_inventory']
        self.max_backlog = self.config['normalization']['max_backlog']
        self.max_pipeline = self.config['normalization']['max_pipeline']
        self.max_demand = self.config['normalization']['max_demand']
        
        # Action/Observation space dimensions
        # Observation: 27 values per agent (9 features × 3 SKUs)
        #   - inventory (3), backlog (3), pipeline_7_to_8_days (3), 
        #   - pipeline_9_to_10_days (3), pipeline_11_to_14_days (3), pipeline_total (3),
        #   - current_price (3), price_ma (3), recent_demand (3)
        self.obs_dim = self.n_skus * 9
        
        # Action: MultiDiscrete([21, 21, 21]) - order 0-20 units per SKU
        self.action_dim_per_sku = 21
        self.action_dim = self.action_dim_per_sku  # For compatibility
        
        # State variables
        self.inventory: np.ndarray = None  # Shape: (n_agents, n_skus)
        self.backlog: np.ndarray = None    # Shape: (n_agents, n_skus)
        self.pipeline: List[List[Dict]] = None  # List of pipeline orders per agent
        
        # Price tracking
        self.current_prices: np.ndarray = None  # Shape: (n_skus,)
        self.price_history: List[List[float]] = None  # For moving average
        
        # Episode tracking
        self.current_day: int = 0
        self.step_num: int = 0
        self.train_mode: bool = True
        self.normalize: bool = True
        
        # Last actions for reward calculation
        self.last_actions: np.ndarray = None
        
        # Demand history for observations
        self.demand_history: List[List[float]] = None
        
        # Agent number for compatibility
        self.agent_num = self.n_agents
        self.eposide_max_steps = self.max_days
        
        print(f"[EnhancedEnv] Initialized with {self.n_agents} agents, "
              f"{self.n_skus} SKUs, {self.max_days} days")
        print(f"[EnhancedEnv] Lead time: Uniform[{self.lt_min}, {self.lt_max}]")
        print(f"[EnhancedEnv] Observation dim: {self.obs_dim}, Action dim per SKU: {self.action_dim_per_sku}")
    
    def reset(self, train: bool = True, normalize: bool = True) -> List[np.ndarray]:
        """
        Reset the environment to initial state.
        
        Args:
            train: If True, use training mode; else evaluation mode
            normalize: If True, normalize observations to [0, 1]
        
        Returns:
            List of initial observations for each agent
        """
        self.current_day = 0
        self.step_num = 0
        self.train_mode = train
        self.normalize = normalize
        
        # Reset state variables
        self.inventory = np.full((self.n_agents, self.n_skus), 10.0, dtype=np.float32)
        self.backlog = np.zeros((self.n_agents, self.n_skus), dtype=np.float32)
        self.pipeline = [[] for _ in range(self.n_agents)]
        
        # Initialize prices
        self.current_prices = np.array(self.data_loader.get_prices(0), dtype=np.float32)
        self.price_history = [[] for _ in range(self.n_skus)]
        for sku in range(self.n_skus):
            self.price_history[sku].append(self.current_prices[sku])
        
        # Initialize demand history
        self.demand_history = [[] for _ in range(self.n_skus)]
        
        # Initialize last actions
        self.last_actions = np.zeros((self.n_agents, self.n_skus), dtype=np.float32)
        
        # Get initial observations
        observations = self._get_observations()
        
        return observations
    
    def step(self, actions: List[np.ndarray], one_hot: bool = False) -> Tuple[
        List[np.ndarray], List[float], List[bool], List[Dict]
    ]:
        """
        Execute one time step of the environment.
        
        Args:
            actions: List of actions for each agent
                     Each action is either one-hot encoded or integer array [sku0_qty, sku1_qty, sku2_qty]
            one_hot: If True, actions are one-hot encoded and need to be decoded
        
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            dones: List of done flags for each agent
            infos: List of info dicts for each agent
        """
        self.current_day += 1
        self.step_num += 1
        
        # Decode actions if one-hot
        if one_hot:
            # Convert one-hot to multi-discrete indices
            decoded_actions = []
            for agent_action in actions:
                # Assuming agent_action is concatenated one-hots: [sku0_onehot, sku1_onehot, sku2_onehot]
                sku_actions = []
                for sku in range(self.n_skus):
                    start_idx = sku * self.action_dim_per_sku
                    end_idx = start_idx + self.action_dim_per_sku
                    sku_action = np.argmax(agent_action[start_idx:end_idx])
                    sku_actions.append(sku_action)
                decoded_actions.append(np.array(sku_actions))
            actions = decoded_actions
        else:
            # Actions are already in the correct format
            actions = [np.array(a) if not isinstance(a, np.ndarray) else a for a in actions]
        
        # Store actions for reward calculation
        self.last_actions = np.array(actions, dtype=np.float32)
        
        # === PHASE 1: PLACE ORDERS (with variable lead time) ===
        self._process_orders(actions)
        
        # === PHASE 2: PROCESS ARRIVALS ===
        self._process_arrivals()
        
        # === PHASE 3: PROCESS DEMAND (at retailer level) ===
        self._process_demand()
        
        # === PHASE 4: ECHELON FLOW (distributor → retailer, manufacturer → distributor) ===
        self._process_echelon_flow()
        
        # === PHASE 5: UPDATE PRICING ===
        self._update_pricing()
        
        # === PHASE 6: CALCULATE REWARDS ===
        rewards = self._calculate_rewards()
        
        # === PHASE 7: GET OBSERVATIONS ===
        observations = self._get_observations()
        
        # === PHASE 8: CHECK TERMINATION ===
        done = self.current_day >= self.max_days
        dones = [done for _ in range(self.n_agents)]
        
        # Info (for debugging/logging)
        infos = [{} for _ in range(self.n_agents)]
        
        return observations, rewards, dones, infos
    
    def _process_orders(self, actions: List[np.ndarray]):
        """
        Process new orders with variable lead time.
        
        Args:
            actions: List of action arrays, each shape (n_skus,)
        """
        for agent_id in range(self.n_agents):
            for sku in range(self.n_skus):
                order_qty = int(actions[agent_id][sku])
                
                if order_qty > 0:
                    # Sample lead time
                    if self.lt_distribution == "uniform":
                        lead_time = np.random.randint(self.lt_min, self.lt_max + 1)
                    else:  # Deterministic fallback
                        lead_time = self.lt_min
                    
                    arrival_day = self.current_day + lead_time
                    
                    # Add to pipeline with tag
                    self.pipeline[agent_id].append({
                        'sku': sku,
                        'qty': order_qty,
                        'arrival_day': arrival_day
                    })
    
    def _process_arrivals(self):
        """Process orders arriving today and update inventory."""
        for agent_id in range(self.n_agents):
            # Filter orders arriving today
            arrived = [order for order in self.pipeline[agent_id] 
                      if order['arrival_day'] == self.current_day]
            
            # Add to inventory
            for order in arrived:
                self.inventory[agent_id][order['sku']] += order['qty']
            
            # Remove arrived orders from pipeline
            self.pipeline[agent_id] = [order for order in self.pipeline[agent_id]
                                       if order['arrival_day'] > self.current_day]
    
    def _process_demand(self):
        """Process customer demand at the retailer level (agent 0)."""
        retailer_id = 0
        
        # Get demand from data loader
        demand = self.data_loader.get_demand(min(self.current_day, self.max_days - 1))
        
        # Update demand history
        for sku in range(self.n_skus):
            self.demand_history[sku].append(demand[sku])
        
        # Fulfill demand or create backlog
        for sku in range(self.n_skus):
            if self.inventory[retailer_id][sku] >= demand[sku]:
                # Fulfill demand
                self.inventory[retailer_id][sku] -= demand[sku]
            else:
                # Create backlog
                shortage = demand[sku] - self.inventory[retailer_id][sku]
                self.inventory[retailer_id][sku] = 0
                self.backlog[retailer_id][sku] += shortage
    
    def _process_echelon_flow(self):
        """
        Process inventory flow between echelons.
        
        Flow: Manufacturer (2) → Distributor (1) → Retailer (0)
        Each upstream echelon tries to fulfill downstream backlog.
        """
        # Distributor → Retailer
        distributor_id = 1
        retailer_id = 0
        for sku in range(self.n_skus):
            if self.backlog[retailer_id][sku] > 0:
                # How much can distributor ship?
                shipment = min(self.backlog[retailer_id][sku], 
                              self.inventory[distributor_id][sku])
                
                self.inventory[distributor_id][sku] -= shipment
                self.backlog[retailer_id][sku] -= shipment
                self.inventory[retailer_id][sku] += shipment
        
        # Manufacturer → Distributor
        manufacturer_id = 2
        for sku in range(self.n_skus):
            # Distributor creates backlog when it ships to retailer
            # (Simplified: assume distributor immediately requests replenishment)
            # This part depends on your specific echelon logic
            pass  # Can extend based on original net_2x3.py logic
    
    def _update_pricing(self):
        """Update current procurement prices from data."""
        day_idx = min(self.current_day, self.max_days - 1)
        self.current_prices = np.array(self.data_loader.get_prices(day_idx), dtype=np.float32)
        
        # Update price history for moving average
        for sku in range(self.n_skus):
            self.price_history[sku].append(self.current_prices[sku])
    
    def _calculate_rewards(self) -> List[float]:
        """
        Calculate rewards for all agents.
        
        Reward = - (holding_cost + backlog_cost + ordering_cost)
        
        Returns:
            List of rewards for each agent
        """
        rewards = []
        
        for agent_id in range(self.n_agents):
            total_cost = 0.0
            
            for sku in range(self.n_skus):
                # Holding cost
                holding_cost = self.H[agent_id][sku] * max(0, self.inventory[agent_id][sku])
                
                # Backlog cost
                backlog_cost = self.B[agent_id][sku] * max(0, self.backlog[agent_id][sku])
                
                # Ordering cost (fixed + variable)
                order_qty = self.last_actions[agent_id][sku]
                if order_qty > 0:
                    fixed_cost = self.C_FIXED[agent_id][sku]
                    # Use price from when order was placed (current_day - 1)
                    price_idx = max(0, min(self.current_day - 1, len(self.price_history[sku]) - 1))
                    variable_cost = self.price_history[sku][price_idx] * order_qty
                    ordering_cost = fixed_cost + variable_cost
                else:
                    ordering_cost = 0
                
                total_cost += holding_cost + backlog_cost + ordering_cost
            
            # Negative cost = reward
            rewards.append(-total_cost)
        
        return rewards
    
    def _get_observations(self) -> List[np.ndarray]:
        """
        Build observations for all agents.
        
        Observation structure (27 values per agent):
        - Inventory levels (3 SKUs)
        - Backlog levels (3 SKUs)
        - Pipeline arriving in 7-8 days (3 SKUs)
        - Pipeline arriving in 9-10 days (3 SKUs)
        - Pipeline arriving in 11-14 days (3 SKUs)
        - Total pipeline in transit (3 SKUs)
        - Current procurement prices (3 SKUs)
        - Price 5-day moving average (3 SKUs)
        - Recent demand average (3 SKUs)
        
        Returns:
            List of observation arrays for each agent
        """
        observations = []
        
        for agent_id in range(self.n_agents):
            obs = []
            
            for sku in range(self.n_skus):
                # 1. Inventory
                inv = self.inventory[agent_id][sku]
                obs.append(inv / self.max_inventory if self.normalize else inv)
                
            for sku in range(self.n_skus):
                # 2. Backlog
                bl = self.backlog[agent_id][sku]
                obs.append(bl / self.max_backlog if self.normalize else bl)
            
            for sku in range(self.n_skus):
                # 3. Pipeline arriving in 7-8 days (short-term for 7-14 day LT)
                short_term_arrivals = sum(
                    order['qty'] for order in self.pipeline[agent_id]
                    if order['sku'] == sku and 
                    self.current_day + 7 <= order['arrival_day'] <= self.current_day + 8
                )
                obs.append(short_term_arrivals / self.max_demand if self.normalize else short_term_arrivals)
            
            for sku in range(self.n_skus):
                # 4. Pipeline arriving in 9-10 days (medium-term)
                medium_term_arrivals = sum(
                    order['qty'] for order in self.pipeline[agent_id]
                    if order['sku'] == sku and 
                    self.current_day + 9 <= order['arrival_day'] <= self.current_day + 10
                )
                obs.append(medium_term_arrivals / self.max_demand if self.normalize else medium_term_arrivals)
            
            for sku in range(self.n_skus):
                # 5. Pipeline arriving in 11-14 days (long-term)
                long_term_arrivals = sum(
                    order['qty'] for order in self.pipeline[agent_id]
                    if order['sku'] == sku and 
                    self.current_day + 11 <= order['arrival_day'] <= self.current_day + 14
                )
                obs.append(long_term_arrivals / self.max_demand if self.normalize else long_term_arrivals)
            
            for sku in range(self.n_skus):
                # 6. Total pipeline in transit
                total_pipeline = sum(
                    order['qty'] for order in self.pipeline[agent_id]
                    if order['sku'] == sku
                )
                obs.append(total_pipeline / self.max_pipeline if self.normalize else total_pipeline)
            
            for sku in range(self.n_skus):
                # 7. Current price
                price = self.current_prices[sku]
                max_price = self.config['pricing']['max_price'][sku]
                obs.append(price / max_price if self.normalize else price)
            
            for sku in range(self.n_skus):
                # 8. Price moving average (5-day)
                price_ma = self._get_price_ma(sku, window=5)
                max_price = self.config['pricing']['max_price'][sku]
                obs.append(price_ma / max_price if self.normalize else price_ma)
            
            for sku in range(self.n_skus):
                # 9. Recent demand average (3-day, only for retailer)
                if agent_id == 0:  # Retailer observes demand
                    demand_avg = self._get_demand_avg(sku, window=3)
                else:  # Upstream agents see 0 (or could see downstream backlog)
                    demand_avg = 0
                obs.append(demand_avg / self.max_demand if self.normalize else demand_avg)
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def _get_price_ma(self, sku: int, window: int = 5) -> float:
        """Calculate moving average of price for a SKU."""
        if len(self.price_history[sku]) < window:
            return np.mean(self.price_history[sku])
        else:
            return np.mean(self.price_history[sku][-window:])
    
    def _get_demand_avg(self, sku: int, window: int = 3) -> float:
        """Calculate moving average of demand for a SKU."""
        if len(self.demand_history[sku]) == 0:
            return 0.0
        elif len(self.demand_history[sku]) < window:
            return np.mean(self.demand_history[sku])
        else:
            return np.mean(self.demand_history[sku][-window:])
    
    # === Compatibility methods for existing training framework ===
    
    def get_demand(self) -> List[float]:
        """Get current demand (for logging/debugging)."""
        if len(self.demand_history[0]) > 0:
            return [self.demand_history[sku][-1] for sku in range(self.n_skus)]
        else:
            return [0.0] * self.n_skus
    
    def get_inventory(self) -> List[float]:
        """Get current inventory levels (for logging)."""
        inv = []
        for agent_id in range(self.n_agents):
            for sku in range(self.n_skus):
                inv.append(self.inventory[agent_id][sku])
        return inv
    
    def get_orders(self) -> List[float]:
        """Get current orders (for logging)."""
        orders = []
        for agent_id in range(self.n_agents):
            for sku in range(self.n_skus):
                orders.append(self.last_actions[agent_id][sku])
        return orders


# Alias for backward compatibility
Env = EnhancedMultiSKUEnv
