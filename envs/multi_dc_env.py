"""
Multi-DC 2-Echelon Inventory Management Environment

Topology:
    Supplier (Unlimited) → 2 DCs (Agents 0,1) → 3 Retailers (Agents 2,3,4)

Key Features:
- Variable lead times: Uniform[7, 14] days for both supplier→DC and DC→retailer
- Continuous action spaces for realistic ordering
- Retailer multi-source ordering: Choose which DC to order from
- Proportional rationing when DCs have insufficient stock
- Dynamic market pricing for DC orders
- Heterogeneous agents (DCs vs Retailers have different obs/action spaces)
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
import os


class MultiDCInventoryEnv:
    """
    Multi-Agent 2-Echelon Inventory Environment with Multiple Distribution Centers
    
    Compatible with standard MARL interfaces (can be wrapped for PettingZoo)
    """
    
    def __init__(self, config_path: str = 'configs/multi_dc_config.yaml'):
        """Initialize the multi-DC environment."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load demand data from CSV
        import pandas as pd
        demand_path = 'data/demand_history.csv'
        self.demand_df = pd.read_csv(demand_path)
        print(f"[MultiDCEnv] Loaded demand data: {len(self.demand_df)} days")
        
        # Network topology
        self.n_dcs = 2  # Distribution Centers
        self.n_retailers = self.config['environment'].get('n_retailers', 3)  # Retail locations
        self.n_agents = self.n_dcs + self.n_retailers  # Total: 17 agents (if 15 retailers)
        self.n_skus = self.config['environment']['n_skus']
        
        # Agent IDs
        self.dc_ids = list(range(self.n_dcs))  # [0, 1]
        self.retailer_ids = list(range(self.n_dcs, self.n_agents))  # [2, 3, 4]
        
        # Lead time parameters
        # Supplier → DC lead times
        self.lt_supplier_to_dc_min = self.config['environment']['lead_time']['supplier_to_dc']['min']
        self.lt_supplier_to_dc_max = self.config['environment']['lead_time']['supplier_to_dc']['max']
        
        # DC → Retailer lead times
        self.lt_dc_to_retailer_min = self.config['environment']['lead_time']['dc_to_retailer']['min']
        self.lt_dc_to_retailer_max = self.config['environment']['lead_time']['dc_to_retailer']['max']
        
        # Episode settings
        self.max_days = self.config['environment']['max_days']
        self.current_day = 0
        
        # Cost parameters
        self._load_cost_parameters()
        self._load_reward_parameters()
        
        # Load constraints
        self._load_constraints()
        
        # State variables
        self.inventory = {}  # {agent_id: np.array(n_skus)}
        self.backlog = {}
        self.pipeline = {}  # {agent_id: List[order_dicts]}
        self.last_actions = {i: 0.0 for i in range(self.n_agents)}  # Track last order qty
        
        # Market price tracking (for DCs ordering from supplier)
        self.market_prices = None
        self.price_history = None
        
        # Demand tracking
        self.demand_history = None
        
        # Observation/Action space definitions
        self._define_spaces()
        
        # For MARL compatibility
        self.agent_num = self.n_agents
        self.agents = [f'agent_{i}' for i in range(self.n_agents)]
        
        print(f"[MultiDCEnv] Initialized: {self.n_dcs} DCs, {self.n_retailers} Retailers, {self.n_skus} SKUs")
    
    def _load_cost_parameters(self):
        """Load cost matrices from config."""
        costs = self.config['costs']
        
        # DC costs (2 DCs × 3 SKUs)
        self.H_dc = np.array(costs['holding_cost_dc'], dtype=np.float32)
        self.B_dc = np.array(costs['backlog_cost_dc'], dtype=np.float32)
        self.C_fixed_dc = np.array(costs['fixed_order_cost_dc'], dtype=np.float32)
        
        # Retailer costs (3 Retailers × 3 SKUs)
        self.H_retailer = np.array(costs['holding_cost_retailer'], dtype=np.float32)
        self.B_retailer = np.array(costs['backlog_cost_retailer'], dtype=np.float32)
        self.C_fixed_retailer = np.array(costs['fixed_order_cost_retailer'], dtype=np.float32)
        
        # Variable costs for ordering from DCs (per retailer, per DC, per SKU)
        # Shape: (n_retailers, n_dcs, n_skus)
        self.C_var_retailer = np.array(costs['variable_cost_retailer'], dtype=np.float32)
        
        # Market price parameters
        self.base_market_price = np.array(self.config['pricing']['base_price'], dtype=np.float32)
        self.price_volatility = self.config['pricing']['volatility']
        self.price_bounds = {
            'min': np.array(self.config['pricing']['min_price'], dtype=np.float32),
            'max': np.array(self.config['pricing']['max_price'], dtype=np.float32)
        }

    def _load_reward_parameters(self):
        """Load reward parameters from config."""
        if 'rewards' in self.config and 'survival_reward' in self.config['rewards']:
            self.survival_reward = float(self.config['rewards']['survival_reward'])
        else:
            self.survival_reward = 0.0
            
        if 'rewards' in self.config and 'termination_penalty' in self.config['rewards']:
            self.termination_penalty = float(self.config['rewards']['termination_penalty'])
        else:
            self.termination_penalty = 0.0

    def _load_constraints(self):
        """Load constraints from config."""
        if 'constraints' in self.config:
            self.on_shelf_quantity = np.array(self.config['constraints']['on_shelf_quantity'], dtype=np.float32)
        else:
            # Default to 0 if not specified
            self.on_shelf_quantity = np.zeros((self.n_retailers, self.n_skus), dtype=np.float32)
    
    def _define_spaces(self):
        """Define observation and action spaces for each agent type."""
        
        # DC Observation: 27D (9 features × 3 SKUs)
        self.obs_dim_dc = self.n_skus * 9
        self.obs_space_dc = spaces.Box(0, 1, (self.obs_dim_dc,), dtype=np.float32)
        
        # Retailer Observation: 36D (12 features × 3 SKUs)
        # Reduced from 14 to 12 by removing "2+ days" pipeline features (DC→Retailer is 1 day fixed)
        self.obs_dim_retailer = self.n_skus * 12
        self.obs_space_retailer = spaces.Box(0, 1, (self.obs_dim_retailer,), dtype=np.float32)
        
        
        # === UNIFORM ACTION SPACES FOR HAPPO COMPATIBILITY ===
        # All agents have 6D continuous actions for uniformity
        # DC agents: only use first 3 dimensions (order qty per SKU from supplier)
        #            last 3 dimensions are ignored/masked
        # Retailer agents: use all 6 dimensions (DC0_SKU0-2, DC1_SKU0-2)
        
        self.action_dim = 6  # Uniform for all agents
        self.action_dim_dc_used = 3  # DCs only use first 3
        self.action_dim_retailer = 6  # Retailers use all 6
        
        # Uniform action space for all agents
        self.action_space = spaces.Box(0, 70, (self.action_dim,), dtype=np.float32)
        
        # Combined spaces (uniform for compatibility)
        self.observation_spaces = {
            i: self.obs_space_dc if i in self.dc_ids else self.obs_space_retailer
            for i in range(self.n_agents)
        }
        
        self.action_spaces = {
            i: self.action_space  # Same for all agents now
            for i in range(self.n_agents)
        }
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment to initial state."""
        self.current_day = 0
        
        # Reset state for all agents
        for agent_id in range(self.n_agents):
            self.inventory[agent_id] = np.full(self.n_skus, 50.0, dtype=np.float32)
            self.backlog[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.pipeline[agent_id] = []
        
        # Initialize market prices
        self.market_prices = self.base_market_price.copy()
        self.price_history = [[] for _ in range(self.n_skus)]
        for sku in range(self.n_skus):
            self.price_history[sku].append(self.market_prices[sku])
        
        # Initialize demand history (for retailers)
        self.demand_history = [[] for _ in range(self.n_skus)]
        
        # Reload demand data (in case changed)
        import pandas as pd
        demand_path = 'data/demand_history.csv'
        self.demand_df = pd.read_csv(demand_path)
        
        # Get initial observations
        observations = self._get_observations()
        
        return observations
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, dict]
    ]:
        """
        Execute one time step.
        
        Args:
            actions: {agent_id: action_array}
                - ALL actions shape (6,) for uniformity
                - DC actions: Only first 3 values used (order qty per SKU from supplier)
                              Last 3 values ignored
                - Retailer actions: All 6 values used [DC0_SKU0, DC0_SKU1, DC0_SKU2, DC1_SKU0, DC1_SKU1, DC1_SKU2]
        
        Returns:
            observations, rewards, dones, infos
        """
        self.current_day += 1
        
        # Clip actions to valid ranges
        actions = self._clip_actions(actions)
        
        # Track total orders for logging
        self.last_actions = {i: 0.0 for i in range(self.n_agents)}
        
        # === PHASE 1: Retailers place orders to DCs ===
        retailer_orders = self._process_retailer_orders(actions)
        
        # === PHASE 2: DCs fulfill retailer orders (with rationing) ===
        self._fulfill_retailer_orders(retailer_orders)
        
        # === PHASE 3: DCs place orders to supplier ===
        self._process_dc_orders(actions)
        
        # === PHASE 4: Process arrivals (both DC and retailer pipelines) ===
        self._process_arrivals()
        
        # === PHASE 5: Customer demand at retailers ===
        self._process_customer_demand()
        
        # === PHASE 6: Update market prices ===
        self._update_market_prices()
        
        # === PHASE 7: Calculate rewards ===
        rewards = self._calculate_rewards(actions)
        
        # === PHASE 8: Get observations ===
        observations = self._get_observations()
        
        # === PHASE 9: Check termination ===
        # Check max days
        time_limit_reached = self.current_day >= self.max_days
        
        # Check on-shelf quantity constraint
        # Terminate if ANY SKU at ANY retailer drops below the threshold
        constraint_violated = False
        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            for sku in range(self.n_skus):
                if self.inventory[retailer_id][sku] < self.on_shelf_quantity[retailer_idx][sku]:
                    constraint_violated = True
                    break
            if constraint_violated:
                break
        
        # Apply termination penalty if violated
        if constraint_violated:
            for i in range(self.n_agents):
                rewards[i] -= self.termination_penalty

        done = time_limit_reached
        dones = {i: done for i in range(self.n_agents)}
        
        infos = {i: {} for i in range(self.n_agents)}
        
        return observations, rewards, dones, infos
    
    def _clip_actions(self, actions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Clip actions to valid ranges and ensure non-negative."""
        clipped = {}
        for agent_id, action in actions.items():
            action = np.clip(action, 0, None)  # No negative orders
            
            if agent_id in self.dc_ids:
                # DC: max 70 per SKU
                clipped[agent_id] = np.clip(action, 0, 1000)
            else:
                # Retailer: max 60 per order
                clipped[agent_id] = np.clip(action, 0, 60)
        
        return clipped
    
    def _process_retailer_orders(self, actions: Dict[int, np.ndarray]) -> Dict:
        """
        Parse retailer actions and register orders with DCs.
        
        Returns:
            retailer_orders: {dc_id: {retailer_id: {sku: qty}}}
        """
        retailer_orders = {dc_id: {} for dc_id in self.dc_ids}
        
        for retailer_id in self.retailer_ids:
            action = actions[retailer_id]  # Shape: (6,)
            
            # Parse: [DC0_SKU0, DC0_SKU1, DC0_SKU2, DC1_SKU0, DC1_SKU1, DC1_SKU2]
            dc0_orders = action[0:3]  # Orders to DC_0
            dc1_orders = action[3:6]  # Orders to DC_1
            
            # Register with DC_0
            retailer_orders[0][retailer_id] = {
                sku: float(dc0_orders[sku]) for sku in range(self.n_skus)
            }
            
            # Register with DC_1
            retailer_orders[1][retailer_id] = {
                sku: float(dc1_orders[sku]) for sku in range(self.n_skus)
            }
            
            # Track total order for logging
            self.last_actions[retailer_id] = np.sum(dc0_orders) + np.sum(dc1_orders)
        
        return retailer_orders
        
    def get_orders(self) -> Dict[int, float]:
        """Return total quantity ordered by each agent in the last step."""
        return self.last_actions
    
    def _fulfill_retailer_orders(self, retailer_orders: Dict):
        """
        DCs fulfill retailer orders with proportional rationing if needed.
        
        Args:
            retailer_orders: {dc_id: {retailer_id: {sku: qty}}}
        """
        for dc_id in self.dc_ids:
            dc_orders = retailer_orders[dc_id]
            
            for sku in range(self.n_skus):
                # Calculate total demand for this SKU
                total_demand = sum(
                    retailer_orders[sku] for retailer_orders in dc_orders.values()
                )
                
                available = self.inventory[dc_id][sku]
                
                if total_demand == 0:
                    continue  # No orders
                
                if available >= total_demand:
                    # Fulfill all orders completely
                    for retailer_id, orders in dc_orders.items():
                        qty = orders[sku]
                        if qty > 0:
                            self._ship_to_retailer(dc_id, retailer_id, sku, qty, lead_time_sample=True)
                            self.inventory[dc_id][sku] -= qty
                else:
                    # RATIONING: Proportional fulfillment
                    for retailer_id, orders in dc_orders.items():
                        qty_ordered = orders[sku]
                        
                        if qty_ordered > 0:
                            # Calculate proportion
                            ratio = qty_ordered / total_demand
                            fulfilled_qty = available * ratio
                            unfulfilled_qty = qty_ordered - fulfilled_qty
                            
                            # Ship fulfilled portion
                            if fulfilled_qty > 0:
                                self._ship_to_retailer(dc_id, retailer_id, sku, fulfilled_qty, lead_time_sample=True)
                            
                            # Create backlog for unfulfilled
                            if unfulfilled_qty > 0:
                                self.backlog[dc_id][sku] += unfulfilled_qty
                    
                    # DC inventory depleted
                    self.inventory[dc_id][sku] = 0
    
    def _ship_to_retailer(self, dc_id: int, retailer_id: int, sku: int, qty: float, lead_time_sample: bool = True):
        """
        Ship from DC to retailer (goes into retailer's pipeline).
        
        Args:
            dc_id: Source DC
            retailer_id: Destination retailer
            sku: SKU index
            qty: Quantity to ship
            lead_time_sample: If True, sample lead time from DC→Retailer distribution; else instant
        """
        if lead_time_sample:
            # Use DC→Retailer lead time (typically 1 day in current config)
            lead_time = np.random.randint(self.lt_dc_to_retailer_min, self.lt_dc_to_retailer_max + 1)
        else:
            lead_time = 0  # Instant
        
        arrival_day = self.current_day + lead_time
        
        self.pipeline[retailer_id].append({
            'sku': sku,
            'qty': qty,
            'arrival_day': arrival_day,
            'source': f'DC_{dc_id}'
        })
    
    def _process_dc_orders(self, actions: Dict[int, np.ndarray]):
        """DCs place orders to supplier (unlimited capacity)."""
        for dc_id in self.dc_ids:
            action = actions[dc_id]  # Shape: (6,) but only use first 3
            
            for sku in range(self.n_skus):
                order_qty = float(action[sku])  # Only access first 3 elements (SKU 0, 1, 2)
                
                if order_qty > 0:
                    # Sample lead time from supplier→DC distribution (Uniform[7, 14] in current config)
                    lead_time = np.random.randint(self.lt_supplier_to_dc_min, self.lt_supplier_to_dc_max + 1)
                    arrival_day = self.current_day + lead_time
                    
                    # Add to DC pipeline
                    self.pipeline[dc_id].append({
                        'sku': sku,
                        'qty': order_qty,
                        'arrival_day': arrival_day,
                        'source': 'supplier'
                    })
            
            # Track total order for logging
            # Note: DC actions are only first 3 elements
            self.last_actions[dc_id] = np.sum(action[:self.n_skus])
    
    def _process_arrivals(self):
        """Process all pipeline arrivals for all agents."""
        for agent_id in range(self.n_agents):
            # Find orders arriving today
            arrived = [order for order in self.pipeline[agent_id]
                      if order['arrival_day'] == self.current_day]
            
            # Add to inventory
            for order in arrived:
                self.inventory[agent_id][order['sku']] += order['qty']
            
            # Remove arrived orders from pipeline
            self.pipeline[agent_id] = [order for order in self.pipeline[agent_id]
                                       if order['arrival_day'] > self.current_day]
    
    def _process_customer_demand(self):
        """Process customer demand at retailer level."""
        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            # Get demand for this retailer
            demand = self._get_demand(retailer_idx, self.current_day)
            
            # Update demand history
            for sku in range(self.n_skus):
                if len(self.demand_history[sku]) <= retailer_idx:
                    self.demand_history[sku].append([])
                self.demand_history[sku][retailer_idx].append(demand[sku])
            
            # Fulfill or create backlog
            for sku in range(self.n_skus):
                if self.inventory[retailer_id][sku] >= demand[sku]:
                    self.inventory[retailer_id][sku] -= demand[sku]
                else:
                    shortage = demand[sku] - self.inventory[retailer_id][sku]
                    self.inventory[retailer_id][sku] = 0
                    self.backlog[retailer_id][sku] += shortage
    
    def _get_demand(self, retailer_idx: int, day: int) -> np.ndarray:
        """
        Get customer demand for a specific retailer on a specific day.
        
        Args:
            retailer_idx: Index among retailers (0, 1, 2)
            day: Current simulation day
        
        Returns:
            demand: np.array of shape (n_skus,)
        """
        # Load demand from CSV file
        # Use modulo to wrap around if episode goes beyond available data
        day_idx = day % len(self.demand_df)
        
        # Get base demand from CSV
        row = self.demand_df.iloc[day_idx]
        base_demand = np.array([
            row['sku_0_demand'],
            row['sku_1_demand'],
            row['sku_2_demand']
        ], dtype=np.float32)
        
        
        retailer_multipliers = [1.0] * self.n_retailers
        demand = base_demand * retailer_multipliers[retailer_idx]
        
        return demand
    
    def _update_market_prices(self):
        """Update market prices with volatility."""
        for sku in range(self.n_skus):
            # Random walk with mean reversion
            change = np.random.normal(0, self.price_volatility * self.base_market_price[sku])
            new_price = self.market_prices[sku] + change
            
            # Mean reversion
            reversion_force = 0.1 * (self.base_market_price[sku] - new_price)
            new_price += reversion_force
            
            # Clip to bounds
            new_price = np.clip(new_price, self.price_bounds['min'][sku], self.price_bounds['max'][sku])
            
            self.market_prices[sku] = new_price
            self.price_history[sku].append(new_price)
    
    def _calculate_rewards(self, actions: Dict[int, np.ndarray]) -> Dict[int, float]:
        """Calculate rewards (negative costs) for all agents."""
        rewards = {}
        
        # DC rewards
        for dc_id in self.dc_ids:
            total_cost = 0.0
            
            for sku in range(self.n_skus):
                # Holding cost
                holding = self.H_dc[dc_id][sku] * self.inventory[dc_id][sku]
                
                # Backlog cost
                backlog = self.B_dc[dc_id][sku] * self.backlog[dc_id][sku]
                
                # Ordering cost (Fixed + Market_Price * Qty)
                order_qty = actions[dc_id][sku]
                if order_qty > 0:
                    # Use price from when order was placed (current market price)
                    market_price = self.market_prices[sku]
                    ordering = self.C_fixed_dc[dc_id][sku] + (market_price * order_qty)
                else:
                    ordering = 0
                
                total_cost += holding + backlog + ordering
            
            rewards[dc_id] = -total_cost + self.survival_reward
        
        # Retailer rewards
        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            total_cost = 0.0
            
            for sku in range(self.n_skus):
                # Holding cost
                holding = self.H_retailer[retailer_idx][sku] * self.inventory[retailer_id][sku]
                
                # Backlog cost
                backlog = self.B_retailer[retailer_idx][sku] * self.backlog[retailer_id][sku]
                
                # Ordering costs from both DCs
                action = actions[retailer_id]
                order_dc0 = action[sku]  # Order to DC_0 for this SKU
                order_dc1 = action[3 + sku]  # Order to DC_1 for this SKU
                
                ordering = 0
                if order_dc0 > 0:
                    var_cost = self.C_var_retailer[retailer_idx][0][sku]
                    ordering += self.C_fixed_retailer[retailer_idx][sku] + (var_cost * order_dc0)
                
                if order_dc1 > 0:
                    var_cost = self.C_var_retailer[retailer_idx][1][sku]
                    ordering += self.C_fixed_retailer[retailer_idx][sku] + (var_cost * order_dc1)
                
                total_cost += holding + backlog + ordering
            
            rewards[retailer_id] = -total_cost + self.survival_reward
        
        return rewards
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Build observations for all agents."""
        observations = {}
        
        # DC observations
        for dc_id in self.dc_ids:
            observations[dc_id] = self._get_dc_observation(dc_id)
        
        # Retailer observations
        for retailer_id in self.retailer_ids:
            observations[retailer_id] = self._get_retailer_observation(retailer_id)
        
        return observations
    
    def _get_dc_observation(self, dc_id: int) -> np.ndarray:
        """
        Build observation for a DC agent (27 dimensions).
        
        Features per SKU (9 total):
        1. Inventory
        2. Backlog
        3. Pipeline 7-8 days
        4. Pipeline 9-10 days
        5. Pipeline 11-14 days
        6. Total pipeline
        7. Current market price
        8. Market price 5-day MA
        9. Aggregate retailer demand signal
        """
        obs = []
        
        for sku in range(self.n_skus):
            # 1. Inventory (normalized)
            obs.append(self.inventory[dc_id][sku] / 1000.0)
            
            # 2. Backlog (normalized)
            obs.append(self.backlog[dc_id][sku] / 50.0)
            
            # 3-5. Pipeline bins
            pipeline_7_8 = sum(o['qty'] for o in self.pipeline[dc_id]
                              if o['sku'] == sku and self.current_day + 7 <= o['arrival_day'] <= self.current_day + 8)
            pipeline_9_10 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 9 <= o['arrival_day'] <= self.current_day + 10)
            pipeline_11_14 = sum(o['qty'] for o in self.pipeline[dc_id]
                                if o['sku'] == sku and self.current_day + 11 <= o['arrival_day'] <= self.current_day + 14)
            
            obs.append(pipeline_7_8 / 50.0)
            obs.append(pipeline_9_10 / 50.0)
            obs.append(pipeline_11_14 / 50.0)
            
            # 6. Total pipeline
            total_pipeline = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku)
            obs.append(total_pipeline / 1000.0)
            
            # 7. Current market price (normalized)
            obs.append(self.market_prices[sku] / self.price_bounds['max'][sku])
            
            # 8. Price MA (5-day)
            price_ma = np.mean(self.price_history[sku][-5:]) if len(self.price_history[sku]) >= 5 else self.market_prices[sku]
            obs.append(price_ma / self.price_bounds['max'][sku])
            
            # 9. Aggregate retailer demand (signal of downstream needs)
            # Simplified: use average recent retailer backlog as proxy
            avg_retailer_backlog = np.mean([self.backlog[r][sku] for r in self.retailer_ids])
            obs.append(avg_retailer_backlog / 50.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_retailer_observation(self, retailer_id: int) -> np.ndarray:
        """
        Build observation for a retailer agent (36 dimensions).
        
        Features per SKU (12 total):
        1. Own inventory
        2. Own backlog
        3. DC_0 inventory (visibility)
        4. DC_1 inventory (visibility)
        5. DC_0 backlog (reliability signal)
        6. DC_1 backlog (reliability signal)
        7-8. DC variable cost indicators
        9. Pipeline from DC_0 (arriving in 0-1 days)
        10. Pipeline from DC_1 (arriving in 0-1 days)
        11. Total own pipeline
        12. Recent demand (3-day average)
        
        Note: Removed "2+ days" pipeline features since DC→Retailer lead time is fixed at 1 day
        """
        obs = []
        retailer_idx = self.retailer_ids.index(retailer_id)
        
        for sku in range(self.n_skus):
            # 1. Own inventory
            obs.append(self.inventory[retailer_id][sku] / 100.0)
            
            # 2. Own backlog
            obs.append(self.backlog[retailer_id][sku] / 50.0)
            
            # 3. DC_0 inventory (DC visibility for routing decision)
            obs.append(self.inventory[0][sku] / 100.0)
            
            # 4. DC_1 inventory
            obs.append(self.inventory[1][sku] / 100.0)
            
            # 5. DC_0 backlog (reliability signal)
            obs.append(self.backlog[0][sku] / 50.0)
            
            # 6. DC_1 backlog
            obs.append(self.backlog[1][sku] / 50.0)
            
            # 7-8. Variable cost from each DC (normalized)
            cost_dc0 = self.C_var_retailer[retailer_idx][0][sku]
            cost_dc1 = self.C_var_retailer[retailer_idx][1][sku]
            obs.append(cost_dc0 / 20.0)
            obs.append(cost_dc1 / 20.0)
            
            # 9-10. Pipeline from each DC (arriving in 0-1 days)
            # With 1-day fixed lead time, this captures all relevant incoming shipments
            pipeline_dc0_0_1 = sum(o['qty'] for o in self.pipeline[retailer_id]
                                  if o['sku'] == sku and o['source'] == 'DC_0' and
                                  self.current_day <= o['arrival_day'] <= self.current_day + 1)
            pipeline_dc1_0_1 = sum(o['qty'] for o in self.pipeline[retailer_id]
                                  if o['sku'] == sku and o['source'] == 'DC_1' and
                                  self.current_day <= o['arrival_day'] <= self.current_day + 1)
            
            obs.append(pipeline_dc0_0_1 / 30.0)
            obs.append(pipeline_dc1_0_1 / 30.0)
            
            # 11. Total own pipeline
            total_pipeline = sum(o['qty'] for o in self.pipeline[retailer_id] if o['sku'] == sku)
            obs.append(total_pipeline / 100.0)
            
            # 12. Recent demand (3-day average)
            if retailer_idx < len(self.demand_history[sku]) and len(self.demand_history[sku][retailer_idx]) > 0:
                recent_demand = np.mean(self.demand_history[sku][retailer_idx][-3:])
            else:
                recent_demand = 0
            obs.append(recent_demand / 50.0)
        
        return np.array(obs, dtype=np.float32)


# Alias for compatibility
Env = MultiDCInventoryEnv
