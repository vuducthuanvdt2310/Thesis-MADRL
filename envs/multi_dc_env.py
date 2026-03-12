"""
Multi-DC 2-Echelon Inventory Management Environment

Topology:
    Supplier (Unlimited) → 2 DCs (Agents 0,1) → 3 Retailers (Agents 2,3,4)

Key Features:
- Variable lead times: Uniform[15, 25] days for supplier→DC only
- Fixed lead time: 1 day for DC→Retailer (deterministic, simplifies retailer training)
- Continuous action spaces for realistic ordering
- Retailer multi-source ordering: Choose which DC to order from
- Proportional rationing when DCs have insufficient stock
- Dynamic market pricing for DC orders
- Bulk-order discount for DC→Supplier orders: higher quantities earn a tiered price discount
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
        self.retailer_ids = list(range(self.n_dcs, self.n_agents))  # [2..16]
        
        # DC-to-Retailer exclusive assignment
        # Loaded from config; maps dc_id -> [retailer agent_ids]
        raw_assignments = self.config.get('dc_assignments', None)
        if raw_assignments:
            self.dc_assignments = {
                0: [self.n_dcs + idx for idx in raw_assignments['dc_0']],
                1: [self.n_dcs + idx for idx in raw_assignments['dc_1']],
            }
        else:
            # Default: split retailers evenly between the two DCs
            half = self.n_retailers // 2
            self.dc_assignments = {
                0: list(range(self.n_dcs, self.n_dcs + half)),
                1: list(range(self.n_dcs + half, self.n_agents)),
            }
        # Reverse map: retailer_id -> assigned dc_id
        self.retailer_to_dc: Dict[int, int] = {}
        for dc_id, r_list in self.dc_assignments.items():
            for r_id in r_list:
                self.retailer_to_dc[r_id] = dc_id
        # print(f"[MultiDCEnv] DC assignments: DC0 -> {self.dc_assignments[0]}, DC1 -> {self.dc_assignments[1]}")
        
        # Pre-compute per-retailer demand stats (mean, std per SKU) from history CSV.
        # Done after n_retailers and n_skus are set.
        self._compute_retailer_demand_stats()
        
        # Lead time parameters
        # Supplier → DC lead times
        self.lt_supplier_to_dc_min = self.config['environment']['lead_time']['supplier_to_dc']['min']
        self.lt_supplier_to_dc_max = self.config['environment']['lead_time']['supplier_to_dc']['max']
        
        # DC → Retailer lead time: fixed at 1 day (deterministic)
        # Variable lead time only applies to Supplier → DC
        self.lt_dc_to_retailer = 1
        
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

        # Fulfilled demand tracking (reset each step) — used for sale revenue bonus
        # demand_fulfilled[retailer_id] = units sold this step (array over SKUs)
        # dc_fulfilled[dc_id] = units shipped to retailers this step (array over SKUs)
        self.demand_fulfilled = {}
        self.dc_fulfilled = {}

        # Per-step demand before fulfillment — used for SL calculation
        self.step_demand = {}   # {retailer_id: np.array(n_skus)}

        # Rolling service level tracking
        # sl_history[retailer_id] = deque of (fulfilled, demanded) tuples (last 7 steps)
        self.sl_history = {}   # {retailer_id: {'fulfilled': list, 'demanded': list}}
        self.episode_sl = 0.0  # Tracks whole-episode SL for infos

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
        rewards_cfg = self.config.get('rewards', {})

        self.survival_reward = float(rewards_cfg.get('survival_reward', 0.0))
        self.termination_penalty = float(rewards_cfg.get('termination_penalty', 0.0))

        # --- Sale revenue bonus ---
        # Retailers earn +sale_revenue_retailer per unit of customer demand fulfilled.
        # DCs earn +sale_revenue_dc per unit successfully shipped to retailers.
        # This creates a POSITIVE gradient toward high service level, solving the
        # 'do-nothing' local optimum where agents minimize costs by not ordering.
        self.sale_revenue_retailer = float(rewards_cfg.get('sale_revenue_retailer', 5.0))
        self.sale_revenue_dc = float(rewards_cfg.get('sale_revenue_dc', 2.0))

        # --- Bulk discount tiers for DC→Supplier orders ---
        discount_cfg = rewards_cfg.get('bulk_discount', {})
        self.bulk_discount_tiers = discount_cfg.get('tiers', [
            {'threshold': 30,  'discount_rate': 0.05},
            {'threshold': 60,  'discount_rate': 0.10},
            {'threshold': 100, 'discount_rate': 0.15},
            {'threshold': 150, 'discount_rate': 0.20},
        ])
        self.bulk_discount_tiers = sorted(
            self.bulk_discount_tiers, key=lambda t: t['threshold'], reverse=True
        )

    def _load_constraints(self):
        """Load constraints from config."""
        if 'constraints' in self.config:
            self.on_shelf_quantity = np.array(self.config['constraints']['on_shelf_quantity'], dtype=np.float32)
        else:
            # Default to 0 if not specified
            self.on_shelf_quantity = np.zeros((self.n_retailers, self.n_skus), dtype=np.float32)

        # Safety-stock threshold: continuous penalty when inventory < threshold
        # Separate from on_shelf_quantity (termination); this is a soft incentive.
        constraints_cfg = self.config.get('constraints', {})
        if 'safety_stock_threshold' in constraints_cfg:
            self.safety_stock_threshold = np.array(constraints_cfg['safety_stock_threshold'], dtype=np.float32)
        else:
            self.safety_stock_threshold = np.zeros((self.n_retailers, self.n_skus), dtype=np.float32)

        # Penalty per unit below the safety-stock level, per step
        self.safety_stock_penalty = float(constraints_cfg.get('safety_stock_penalty', 0.0))
    
    def _define_spaces(self):
        """Define observation and action spaces for each agent type."""
        
        # DC Observation: 28D = (9 features × 3 SKUs) + 1 global SL feature
        # Extra feature: 7-day rolling system service level [0, 1]
        self.obs_dim_dc = self.n_skus * 9 + 1
        self.obs_space_dc = spaces.Box(0, 1, (self.obs_dim_dc,), dtype=np.float32)

        # Retailer Observation: 22D = (7 features × 3 SKUs) + 1 own rolling SL feature
        # Extra feature: 7-day rolling service level for this retailer [0, 1]
        self.obs_dim_retailer = self.n_skus * 7 + 1
        self.obs_space_retailer = spaces.Box(0, 1, (self.obs_dim_retailer,), dtype=np.float32)
        
        # === UNIFORM ACTION SPACES FOR HAPPO COMPATIBILITY ===
        # All agents have 6D continuous actions for uniformity.
        # DC agents:       use action[0:3] (order qty per SKU from supplier)
        #                  action[3:6] ignored
        # Retailer agents: use action[0:3] (order qty per SKU from their ASSIGNED DC)
        #                  action[3:6] ignored
        
        self.action_dim = 6            # Uniform for all agents
        self.action_dim_dc_used = 3    # DCs only use first 3
        self.action_dim_retailer = 3   # Retailers only use first 3 (assigned DC)
        
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
        # DCs starting inventory: each DC serves 7-8 retailers demanding ~1.41 units/step/SKU
        # = ~10.5 units/step/SKU total drain.  Supplier lead time is now 7-14 days, so the
        # DC needs a 7-14 day buffer = ~75-150 units.  We use 100 as a safe midpoint.
        # Retailers start at 20 (demand ~1.4/step; 20 units gives ~14-day runway while pipeline
        # builds up given the 1-day DC→Retailer lead time).
        for agent_id in range(self.n_agents):
            if agent_id in self.dc_ids:
                self.inventory[agent_id] = np.full(self.n_skus, 100.0, dtype=np.float32)
            else:
                self.inventory[agent_id] = np.full(self.n_skus, 20.0, dtype=np.float32)
            self.backlog[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.pipeline[agent_id] = []
        
        # Initialize market prices
        self.market_prices = self.base_market_price.copy()
        self.price_history = [[] for _ in range(self.n_skus)]
        for sku in range(self.n_skus):
            self.price_history[sku].append(self.market_prices[sku])
        
        # Initialize demand history (for retailers)
        self.demand_history = [[] for _ in range(self.n_skus)]

        # Initialize per-step fulfilled-demand trackers
        for agent_id in range(self.n_agents):
            self.demand_fulfilled[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.dc_fulfilled[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand[agent_id] = np.zeros(self.n_skus, dtype=np.float32)

        # Initialize rolling service level history
        for retailer_id in self.retailer_ids:
            self.sl_history[retailer_id] = {'fulfilled': [], 'demanded': []}
        self.episode_sl = 0.0

        # demand stats are pre-computed at __init__; no need to reload CSV every episode

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
        
        # === PHASE 10: Compute step and episode service level ===
        step_total_demand    = 0.0
        step_total_fulfilled = 0.0
        for retailer_id in self.retailer_ids:
            step_total_demand    += float(np.sum(self.step_demand[retailer_id]))
            step_total_fulfilled += float(np.sum(self.demand_fulfilled[retailer_id]))

        step_sl = (step_total_fulfilled / step_total_demand) if step_total_demand > 0 else 1.0

        # Running episode average: exponential moving with simple accumulation
        alpha = 1.0 / self.current_day  # decays as more steps accumulate
        self.episode_sl = self.episode_sl + alpha * (step_sl - self.episode_sl)

        infos = {i: {
            'step_service_level':    step_sl,
            'episode_service_level': self.episode_sl,
        } for i in range(self.n_agents)}
        
        return observations, rewards, dones, infos
    
    def _clip_actions(self, actions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Clip actions to valid order quantity bounds.

        Calibrated from demand_history.csv (365 days, 3 SKUs):
          SKU_0: mean=1.41, std=1.99  |  SKU_1: mean=1.06, std=1.28  |  SKU_2: mean=0.77, std=1.06

        WHY min=0 (removed forced minimums):
          The old min=150 (DC) and min=50 (retailer) forced huge orders every step.
          With a 365-day episode, this caused:
            DC accumulates ~151,000 units over the episode → holding costs ~700M per DC.
            Total reward: ~-1.5 BILLION even without any service failures.
          The agent had zero gradient from ordering quantity alone — same outcome no matter what.
          With min=0, agents genuinely choose order quantities based on state.
          The sale_revenue bonus (5.0/unit fulfilled) provides the incentive to order proactively.

        RETAILER  (DC→Retailer lead time = 1 day):
          max=100 covers ~70x average daily demand — allows building a large safety stock buffer.

        DC  (Supplier→DC lead time = 7-14 days, each DC serves ~8 retailers):
          max=1000 covers ~14 days x ~11.3 units/day drain = ~158 units in one order.
          Agents may place multiple orders per lead-time period; cap prevents runaway inventory.
        """
        clipped = {}
        for agent_id, action in actions.items():
            if agent_id in self.dc_ids:
                clipped[agent_id] = np.clip(action, 0, 500)   # DC: 0 to 1000 units
            else:
                clipped[agent_id] = np.clip(action, 0, 70)    # Retailer: 0 to 100 units
        return clipped



    
    def _process_retailer_orders(self, actions: Dict[int, np.ndarray]) -> Dict:
        """
        Parse retailer actions and register orders with their ASSIGNED DC only.
        
        Each retailer exclusively serves one DC (see self.retailer_to_dc).
        action[0:3] = order quantities per SKU for the assigned DC.
        action[3:6] = unused (kept for uniform 6D action space).
        
        Returns:
            retailer_orders: {dc_id: {retailer_id: {sku: qty}}}
        """
        retailer_orders = {dc_id: {} for dc_id in self.dc_ids}
        
        for retailer_id in self.retailer_ids:
            action = actions[retailer_id]  # Shape: (6,) but only [0:3] used
            assigned_dc = self.retailer_to_dc[retailer_id]
            
            # action[0:3] → order from assigned DC per SKU
            retailer_orders[assigned_dc][retailer_id] = {
                sku: float(action[sku]) for sku in range(self.n_skus)
            }
            # action[3:6] is deliberately unused (maintained for HAPPO buffer uniformity)
            
            # Track total order for logging
            self.last_actions[retailer_id] = float(np.sum(action[:self.n_skus]))
        
        return retailer_orders
        
    def get_orders(self) -> Dict[int, float]:
        """Return total quantity ordered by each agent in the last step."""
        return self.last_actions
    
    def _fulfill_retailer_orders(self, retailer_orders: Dict):
        """
        DCs fulfill retailer orders with proportional rationing if needed.
        Tracks dc_fulfilled[dc_id][sku] for the sale revenue bonus.

        Args:
            retailer_orders: {dc_id: {retailer_id: {sku: qty}}}
        """
        # Reset DC fulfilled tracker for this step
        for dc_id in self.dc_ids:
            self.dc_fulfilled[dc_id] = np.zeros(self.n_skus, dtype=np.float32)

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
                            self.dc_fulfilled[dc_id][sku] += qty  # track fulfilled
                else:
                    # RATIONING: Proportional fulfillment
                    for retailer_id, orders in dc_orders.items():
                        qty_ordered = orders[sku]

                        if qty_ordered > 0:
                            ratio = qty_ordered / total_demand
                            fulfilled_qty = available * ratio
                            unfulfilled_qty = qty_ordered - fulfilled_qty

                            if fulfilled_qty > 0:
                                self._ship_to_retailer(dc_id, retailer_id, sku, fulfilled_qty, lead_time_sample=True)
                                self.dc_fulfilled[dc_id][sku] += fulfilled_qty  # track fulfilled

                            if unfulfilled_qty > 0:
                                self.backlog[dc_id][sku] += unfulfilled_qty

                    # DC inventory depleted
                    self.inventory[dc_id][sku] = 0
    
    def _ship_to_retailer(self, dc_id: int, retailer_id: int, sku: int, qty: float, lead_time_sample: bool = True):
        """
        Ship from DC to retailer (goes into retailer's pipeline).
        
        DC→Retailer lead time is fixed at 1 day (deterministic).
        Variable lead time applies only to Supplier→DC.
        
        Args:
            dc_id: Source DC
            retailer_id: Destination retailer
            sku: SKU index
            qty: Quantity to ship
            lead_time_sample: If True, use fixed 1-day DC→Retailer lead time; else instant (0)
        """
        if lead_time_sample:
            lead_time = self.lt_dc_to_retailer  # Fixed: 1 day
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
        """Process customer demand at retailer level.

        Key change: existing backlog is cleared FIRST using available inventory
        before new demand is served.  This makes the MDP recoverable — agents
        learn that replenishing stock erases past penalties, creating a clear
        incentive to order aggressively after a stockout.

        Tracks demand_fulfilled and step_demand for service-level calculation.
        """
        # Reset per-step trackers
        for retailer_id in self.retailer_ids:
            self.demand_fulfilled[retailer_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand[retailer_id]       = np.zeros(self.n_skus, dtype=np.float32)

        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            demand = self._get_demand(retailer_idx, self.current_day)

            # Record gross demand for SL denominator
            self.step_demand[retailer_id] = demand.copy()

            # Update demand history
            for sku in range(self.n_skus):
                if len(self.demand_history[sku]) <= retailer_idx:
                    self.demand_history[sku].append([])
                self.demand_history[sku][retailer_idx].append(demand[sku])

            for sku in range(self.n_skus):
                avail = self.inventory[retailer_id][sku]

                # ── Step A: Clear existing backlog with available inventory ──
                # If there is backlog from prior stockouts and we now have stock,
                # fulfil the backlog first.  This removes the permanent-penalty
                # death-spiral and teaches agents that restocking pays off.
                if self.backlog[retailer_id][sku] > 0 and avail > 0:
                    backlog_cleared = min(self.backlog[retailer_id][sku], avail)
                    avail -= backlog_cleared
                    self.backlog[retailer_id][sku] -= backlog_cleared
                    # Count cleared backlog as fulfilled demand (service recovery)
                    self.demand_fulfilled[retailer_id][sku] += backlog_cleared

                # ── Step B: Serve today's fresh customer demand ──
                if avail >= demand[sku]:
                    avail -= demand[sku]
                    self.demand_fulfilled[retailer_id][sku] += demand[sku]  # fully met
                else:
                    # Partial fulfillment on today's demand
                    self.demand_fulfilled[retailer_id][sku] += avail
                    shortage = demand[sku] - avail
                    avail = 0
                    self.backlog[retailer_id][sku] += shortage

                self.inventory[retailer_id][sku] = avail

            # ── Update rolling SL history (last 7 steps) for observation feature ──
            step_demanded  = float(np.sum(self.step_demand[retailer_id]))
            step_fulfilled = float(np.sum(self.demand_fulfilled[retailer_id]))
            self.sl_history[retailer_id]['fulfilled'].append(step_fulfilled)
            self.sl_history[retailer_id]['demanded'].append(step_demanded)
            # Keep only most recent 7 steps
            if len(self.sl_history[retailer_id]['fulfilled']) > 7:
                self.sl_history[retailer_id]['fulfilled'].pop(0)
                self.sl_history[retailer_id]['demanded'].pop(0)
    
    def _compute_retailer_demand_stats(self):
        """
        Compute a single shared demand distribution (mean, std per SKU)
        from the demand history CSV. All retailers sample from this same
        Normal distribution, so demand is stochastic but identically
        distributed across retailers.
        """
        sku_cols = ['sku_0_demand', 'sku_1_demand', 'sku_2_demand']
        self.demand_mean = np.array([self.demand_df[c].mean() for c in sku_cols], dtype=np.float32)
        self.demand_std  = np.maximum(
            np.array([self.demand_df[c].std() for c in sku_cols], dtype=np.float32),
            0.1  # ensure std > 0
        )
        print(f"[MultiDCEnv] Demand distribution (shared across all retailers):")
        print(f"  mean={np.round(self.demand_mean, 2)}  std={np.round(self.demand_std, 2)}")

    def _get_demand(self, retailer_idx: int, day: int) -> np.ndarray:
        """
        Sample customer demand from Normal(mean, std) fitted to the CSV history.
        All retailers share the same distribution; demand is clipped to >= 0.

        Args:
            retailer_idx: Index among retailers (unused — same distribution for all)
            day: Current simulation day (unused — i.i.d. sampling)

        Returns:
            demand: np.array of shape (n_skus,), non-negative
        """
        demand = np.random.normal(self.demand_mean, self.demand_std).astype(np.float32)
        demand = np.maximum(demand, 0.0)
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
    
    def _get_bulk_discount_rate(self, total_order_qty: float) -> float:
        """
        Return the applicable bulk-discount rate for a DC order to the supplier.

        The discount is tiered: the highest threshold that the total order quantity
        (summed across all SKUs) exceeds determines the discount rate.

        Args:
            total_order_qty: Sum of order quantities across all SKUs.

        Returns:
            discount_rate: A value in [0, 1] representing the fraction of the
                           variable (market-price) ordering cost that is returned
                           as a reward bonus.
        """
        for tier in self.bulk_discount_tiers:  # already sorted descending by threshold
            if total_order_qty >= tier['threshold']:
                return float(tier['discount_rate'])
        return 0.0  # no discount for small orders

    def _calculate_rewards(self, actions: Dict[int, np.ndarray]) -> Dict[int, float]:
        """Calculate rewards for all agents.

        Reward = sale_revenue_bonus - holding_cost - backlog_cost - ordering_cost

        The SALE REVENUE BONUS is the key signal for service level:
          - Retailers earn +sale_revenue_retailer per unit of customer demand fulfilled.
          - DCs earn +sale_revenue_dc per unit shipped to retailers.
        This makes "fill demand" the primary objective, not just "minimize cost".
        Backlog costs provide a secondary penalty to discourage stockouts.

        Normalization:
          All per-step rewards are divided by max_days so the cumulative episode reward
          stays in a stable numerical range regardless of episode length.
          Without this, backlog costs grow quadratically with episode length
          (backlog accumulates linearly → cost per step grows linearly → total scales as T²),
          making value function estimation and policy gradients unstable for long episodes.
        """
        rewards = {}

        # ---- DC rewards ----
        for dc_id in self.dc_ids:
            total_cost = 0.0
            variable_order_cost = 0.0

            total_order_qty = float(np.sum(actions[dc_id][:self.n_skus]))
            discount_rate = self._get_bulk_discount_rate(total_order_qty)

            for sku in range(self.n_skus):
                holding  = self.H_dc[dc_id][sku] * self.inventory[dc_id][sku]
                backlog  = self.B_dc[dc_id][sku] * self.backlog[dc_id][sku]

                order_qty = actions[dc_id][sku]
                if order_qty > 0:
                    market_price = self.market_prices[sku]
                    var_cost = market_price * order_qty
                    ordering = self.C_fixed_dc[dc_id][sku] + var_cost
                    variable_order_cost += var_cost
                else:
                    ordering = 0.0

                total_cost += holding + backlog + ordering

            # Bulk discount on ordering cost
            bulk_discount_bonus = discount_rate * variable_order_cost

            # Sale revenue for units shipped to retailers this step
            dc_sale_revenue = self.sale_revenue_dc * float(np.sum(self.dc_fulfilled[dc_id]))

            rewards[dc_id] = -total_cost + bulk_discount_bonus + dc_sale_revenue

        # ---- Retailer rewards ----
        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            total_cost = 0.0
            assigned_dc = self.retailer_to_dc[retailer_id]

            for sku in range(self.n_skus):
                holding = self.H_retailer[retailer_idx][sku] * self.inventory[retailer_id][sku]
                backlog = self.B_retailer[retailer_idx][sku] * self.backlog[retailer_id][sku]

                action = actions[retailer_id]
                order_qty = float(action[sku])

                ordering = 0.0
                if order_qty > 0:
                    var_cost = self.C_var_retailer[retailer_idx][assigned_dc][sku]
                    ordering = self.C_fixed_retailer[retailer_idx][sku] + (var_cost * order_qty)

                # Safety-stock penalty: smooth gradient to maintain buffer stock
                ss_threshold = self.safety_stock_threshold[retailer_idx][sku]
                if ss_threshold > 0:
                    shortfall = max(0.0, ss_threshold - self.inventory[retailer_id][sku])
                    ordering += self.safety_stock_penalty * shortfall

                total_cost += holding + backlog + ordering

            # Sale revenue bonus: +R per unit of customer demand actually fulfilled
            retailer_sale_revenue = self.sale_revenue_retailer * float(
                np.sum(self.demand_fulfilled[retailer_id])
            )

            rewards[retailer_id] = -total_cost + retailer_sale_revenue + self.survival_reward

        # === Normalize all rewards by episode length ===
        # Prevents quadratic reward blow-up from cumulative backlog costs in long episodes.
        # Per-step reward stays in a stable range regardless of max_days.
        # All relative cost ratios are preserved — incentive structure is unchanged.
        norm = float(self.max_days)
        rewards = {agent_id: r / norm for agent_id, r in rewards.items()}

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
        Build observation for a DC agent (28D = 9 features × 3 SKUs + 1 global SL).

        Features per SKU (27D):
        1. Inventory
        2. Backlog
        3. Pipeline 7-9 days
        4. Pipeline 10-12 days
        5. Pipeline 13-14 days
        6. Total pipeline
        7. Current market price
        8. Aggregate retailer backlog  (demand-pressure signal)
        9. Aggregate retailer inventory

        Global feature (1D):
        10. 7-day rolling system service level [0, 1]  ← NEW
        """
        obs = []
        
        for sku in range(self.n_skus):
            # 1. DC Inventory
            # DCs accumulate stock across many lead-time cycles (max L/T=25 days, max order=70).
            # Realistic ceiling: ~1000 units (25 days x 70 / ~2 cycle overlap).
            obs.append(self.inventory[dc_id][sku] / 1000.0)
            
            # 2. DC Backlog (failure-state signal; chronic large backlog = bad policy)
            obs.append(self.backlog[dc_id][sku] / 100.0)
            
            # 3-5. Pipeline bins aligned with Uniform[7, 14] lead time (new config).
            # Arrivals expected between day+7 and day+14.
            pipeline_7_9 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 7 <= o['arrival_day'] <= self.current_day + 9)
            pipeline_10_12 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 10 <= o['arrival_day'] <= self.current_day + 12)
            pipeline_13_14 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 13 <= o['arrival_day'] <= self.current_day + 14)
            
            obs.append(pipeline_7_9  / 70.0)   # 3-day window early
            obs.append(pipeline_10_12 / 70.0)  # 3-day window mid
            obs.append(pipeline_13_14 / 70.0)  # 2-day window late
            
            # 6. Total DC pipeline
            # Max lead time=14, max order=70/step -> ceiling ~980; use 1000 for safety.
            total_pipeline = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku)
            obs.append(total_pipeline / 1000.0)
            
            # 7. Current market price (normalized by its own price ceiling)
            obs.append(self.market_prices[sku] / self.price_bounds['max'][sku])
            
            # 8. Aggregate retailer backlog (downstream demand pressure on DC)
            avg_retailer_backlog = np.mean([self.backlog[r][sku] for r in self.retailer_ids])
            obs.append(avg_retailer_backlog / 100.0)

            # 9. Aggregate retailer inventory (proactive restock signal)
            avg_retailer_inventory = np.mean([self.inventory[r][sku] for r in self.retailer_ids])
            obs.append(avg_retailer_inventory / 150.0)

        # 10. [Global] 7-day rolling SYSTEM service level (across all retailers)
        # This directly tells the DC how well the downstream chain is being served,
        # which is the most actionable signal for proactive replenishment.
        total_f = sum(sum(self.sl_history[r]['fulfilled']) for r in self.retailer_ids)
        total_d = sum(sum(self.sl_history[r]['demanded'])  for r in self.retailer_ids)
        rolling_sl = (total_f / total_d) if total_d > 0 else 1.0
        obs.append(float(np.clip(rolling_sl, 0.0, 1.0)))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_retailer_observation(self, retailer_id: int) -> np.ndarray:
        """
        Build observation for a retailer agent (22D = 7 features × 3 SKUs + 1 own SL).

        Each retailer exclusively orders from its ASSIGNED DC.
        DC→Retailer lead time is fixed at 1 day.

        Features per SKU (21D):
        1. Own inventory
        2. Own backlog
        3. Assigned DC inventory
        4. Assigned DC backlog
        5. Pipeline arriving tomorrow (day+1)
        6. Total own pipeline
        7. Recent demand (3-day average)

        Global feature (1D):
        8. Own 7-day rolling service level [0, 1]  ← NEW
        [Removed: Variable ordering cost (static), pipeline_day2/day3 (always 0)]
        """
        obs = []
        retailer_idx = self.retailer_ids.index(retailer_id)
        assigned_dc   = self.retailer_to_dc[retailer_id]
        
        for sku in range(self.n_skus):
            # 1. Own inventory
            obs.append(self.inventory[retailer_id][sku] / 150.0)
            
            # 2. Own backlog (failure-state cap)
            obs.append(self.backlog[retailer_id][sku] / 100.0)
            
            # 3. Assigned DC inventory (ceiling matches DC's own observation scale ~1000)
            obs.append(self.inventory[assigned_dc][sku] / 1000.0)
            
            # 4. Assigned DC backlog (reliability signal)
            obs.append(self.backlog[assigned_dc][sku] / 100.0)
            
            # 5. Pipeline arriving tomorrow (DC→Retailer L/T = 1 day fixed)
            pipeline_day1 = sum(o['qty'] for o in self.pipeline[retailer_id]
                                if o['sku'] == sku and
                                o['arrival_day'] == self.current_day + 1)
            obs.append(pipeline_day1 / 70.0)

            # 6. Total own pipeline (with 1-day L/T, this ≈ pipeline_day1)
            total_pipeline = sum(o['qty'] for o in self.pipeline[retailer_id] if o['sku'] == sku)
            obs.append(total_pipeline / 70.0)
            
            # 7. Recent demand (3-day average)
            demand_cap = float(self.demand_mean[sku] + 3.0 * self.demand_std[sku])
            if retailer_idx < len(self.demand_history[sku]) and len(self.demand_history[sku][retailer_idx]) > 0:
                recent_demand = np.mean(self.demand_history[sku][retailer_idx][-3:])
            else:
                recent_demand = 0
            obs.append(recent_demand / demand_cap)

        # 8. [Own] 7-day rolling service level for this retailer
        # Provides a direct "am I stocking shelves?" policy quality signal.
        # Raw backlog tells HOW BAD the situation is; SL rate tells HOW WELL
        # the retailer has been serving customers. Both together are more
        # informative than either alone.
        hist = self.sl_history[retailer_id]
        if len(hist['demanded']) > 0:
            total_d = sum(hist['demanded'])
            total_f = sum(hist['fulfilled'])
            own_sl = (total_f / total_d) if total_d > 0 else 1.0
        else:
            own_sl = 1.0  # assume perfect at episode start (no data yet)
        obs.append(float(np.clip(own_sl, 0.0, 1.0)))
        
        return np.array(obs, dtype=np.float32)


# Alias for compatibility
Env = MultiDCInventoryEnv
