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
        self.demand_fulfilled = {}

        # Per-step demand before fulfillment — used for SL calculation
        self.step_demand = {}   # {retailer_id: np.array(n_skus)}
        # Per-step fresh-demand-met (separate from backlog clearance for correct SL)
        self.step_demand_met = {}  # {retailer_id: np.array(n_skus)}

        # ── DC per-retailer backlog (replaces flat dc backlog for downstream shortfalls) ──
        # dc_retailer_backlog[dc_id][retailer_id] = {sku: float}  (units owed to that retailer)
        self.dc_retailer_backlog = {}

        # ── DC Cycle Service Level tracking ──
        # Cycle SL = (orders received - orders that triggered any backlog) / orders received
        # Tracked per DC across the entire episode.
        self.dc_cycle_sl_orders_received  = {}  # {dc_id: int}  total retailer orders sent to DC
        self.dc_cycle_sl_orders_no_backlog = {}  # {dc_id: int}  orders fulfilled without any backlog

        # Rolling service level tracking
        # sl_history[retailer_id] = deque of (fulfilled, demanded) tuples (last 7 steps)
        self.sl_history = {}   # {retailer_id: {'fulfilled': list, 'demanded': list}}
        self.episode_sl = 0.0  # Tracks whole-episode SL for infos

        # ── Order-count Fill Rate tracking (retailer level) ──
        # An "order" = one (retailer, sku) demand event per step.
        # Fulfilled from on-hand = demand[sku] covered ENTIRELY without adding to backlog.
        # step trackers (reset each step)
        self.step_orders_placed      = {}  # {retailer_id: {sku: int}}  orders placed this step
        self.step_orders_from_stock  = {}  # {retailer_id: {sku: int}}  fully met from on-hand
        # episode accumulators (reset each episode)
        self.ep_orders_placed     = 0  # cumulative across all retailers, SKUs, steps
        self.ep_orders_from_stock = 0  # cumulative orders fully met from on-hand

        # Market price tracking (for DCs ordering from supplier)
        self.market_prices = None
        self.price_history = None

        # Demand tracking
        self.demand_history = None

        # ── Step 3: Previous-potential tracking for Look-Back PBRS ────────────────
        # prev_potential[agent_id][sku] = Φ(s_{t-1}, a_{t-1})  (scalar float)
        # Initialised to 0.0 and reset every episode so that F = Φ(s_1,a_1) − 0
        # on the very first step (a mild positive or negative nudge, then fast learning).
        self.prev_potential: Dict[int, Dict[int, float]] = {
            agent_id: {sku: 0.0 for sku in range(self.n_skus)}
            for agent_id in range(self.n_agents)
        }
        
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

        self.termination_penalty = float(rewards_cfg.get('termination_penalty', 0.0))

        # --- Sale revenue bonus ---
        # Retailers earn +sale_revenue_retailer per unit of customer demand fulfilled.
        # DCs earn +sale_revenue_dc per unit successfully shipped to retailers.
        # This creates a POSITIVE gradient toward high service level, solving the
        # 'do-nothing' local optimum where agents minimize costs by not ordering.
        self.sale_revenue_retailer = float(rewards_cfg.get('sale_revenue_retailer', 5.0))

        # --- Alpha-weighted global reward mixing ---
        # Each agent's final reward = (1 - alpha) * local + alpha * global_mean.
        # alpha=0.0  → purely local (current default, backward-compatible).
        # alpha=0.2  → recommended starting point from literature.
        # alpha=1.0  → purely global (all agents share the mean reward).
        self.reward_alpha = float(rewards_cfg.get('alpha', 0.0))

        # --- Excess inventory penalty ---
        # Applied when inventory > target_stock_days × mean_daily_demand.
        # Discourages hoarding far beyond what demands requires.
        self.target_stock_days_retailer = float(rewards_cfg.get('target_stock_days_retailer', 7))
        self.target_stock_days_dc       = float(rewards_cfg.get('target_stock_days_dc', 14))
        self.excess_penalty_retailer    = float(rewards_cfg.get('excess_penalty_retailer', 0.0))
        self.excess_penalty_dc          = float(rewards_cfg.get('excess_penalty_dc', 0.0))
        # Pre-compute per-SKU target levels (populated after demand stats are computed).
        # Updated in reset() once demand stats are available.
        self._retailer_target_stock = None
        self._dc_target_stock       = None

        # ── Step 1 & 5: Look-Back Potential-Based Reward Shaping ──────────────────
        # k   : scaling factor; maps |heuristic - action| → potential magnitude.
        #        Set relative to the reward normalisation (÷ max_days) so that
        #        shaping is in the same numerical range as the base reward.
        # window : rolling window (steps) for heuristic demand estimation.
        # shaping_weight : annealing multiplier; starts at 1.0 and decays to 0.
        shaping_cfg = rewards_cfg.get('heuristic_shaping', {})
        self.shaping_k      = float(shaping_cfg.get('k', 0.5))
        self.shaping_window = int(shaping_cfg.get('demand_window', 14))
        self.shaping_weight = float(shaping_cfg.get('initial_weight', 1.0))
        self.shaping_enabled = bool(shaping_cfg.get('enabled', True))

    def _load_constraints(self):
        """Load constraints from config."""
        # Safety-stock threshold: continuous penalty when inventory < threshold
        # This is a soft incentive.
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
        
        # === UNIFORM 3D ACTION SPACE ===
        # All agents use 3 continuous actions — one order quantity per SKU.
        # DCs:      action[sku] = units to order from the supplier
        # Retailers: action[sku] = units to order from their assigned DC
        # With each retailer assigned to exactly one DC, 3D is sufficient for everyone.
        
        self.action_dim = 3            # One per SKU (n_skus = 3)
        
        # Action space bounds for DCs and Retailers
        self.action_space_dc = spaces.Box(0, 1000, (self.action_dim,), dtype=np.float32)
        self.action_space_retailer = spaces.Box(0, 10, (self.action_dim,), dtype=np.float32)

        # Uniform action space for compatibility
        self.action_space = spaces.Box(0, 1000, (self.action_dim,), dtype=np.float32)
        
        # Combined spaces (uniform for compatibility)
        self.observation_spaces = {
            i: self.obs_space_dc if i in self.dc_ids else self.obs_space_retailer
            for i in range(self.n_agents)
        }
        
        self.action_spaces = {
            i: self.action_space_dc if i in self.dc_ids else self.action_space_retailer
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
                self.inventory[agent_id] = np.full(self.n_skus, 2000.0, dtype=np.float32)
            else:
                self.inventory[agent_id] = np.full(self.n_skus, 30.0, dtype=np.float32)
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
            self.step_demand[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand_met[agent_id] = np.zeros(self.n_skus, dtype=np.float32)

        # Initialize DC per-retailer backlog and cycle-SL counters
        for dc_id in self.dc_ids:
            self.dc_retailer_backlog[dc_id] = {
                r_id: {sku: 0.0 for sku in range(self.n_skus)}
                for r_id in self.retailer_ids
            }
            self.dc_cycle_sl_orders_received[dc_id]   = 0
            self.dc_cycle_sl_orders_no_backlog[dc_id] = 0

        # Initialize rolling service level history
        for retailer_id in self.retailer_ids:
            self.sl_history[retailer_id] = {'fulfilled': [], 'demanded': []}
        self.episode_sl = 0.0

        # Initialize order-count fill rate trackers
        for retailer_id in self.retailer_ids:
            self.step_orders_placed[retailer_id]     = {sku: 0 for sku in range(self.n_skus)}
            self.step_orders_from_stock[retailer_id] = {sku: 0 for sku in range(self.n_skus)}
        self.ep_orders_placed     = 0
        self.ep_orders_from_stock = 0

        # ── Reset previous-potential to zero at episode start ──────────────────
        for agent_id in range(self.n_agents):
            for sku in range(self.n_skus):
                self.prev_potential[agent_id][sku] = 0.0

        # demand stats are pre-computed at __init__; no need to reload CSV every episode

        # Pre-compute excess-inventory target stock levels (per SKU).
        # Retailer target: fixed at starting inventory (30 units/SKU).
        # DC target: target_stock_days_dc × (n_retailers_per_dc × mean daily demand per SKU).
        self._retailer_target_stock = np.full(self.n_skus, 30.0, dtype=np.float32)  # = starting inventory
        # Each DC serves a different number of retailers; use the larger group as conservative bound.
        avg_retailers_per_dc = self.n_retailers / self.n_dcs
        self._dc_target_stock = (
            self.target_stock_days_dc * avg_retailers_per_dc * self.demand_mean
        )  # shape: (n_skus,)

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
                - ALL actions shape (3,): one order quantity per SKU
                - DC actions:      action[sku] = units to order from supplier
                - Retailer actions: action[sku] = units to order from assigned DC
        
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
        
        # === PHASE 7: Calculate rewards (includes look-back PBRS shaping) ===
        rewards = self._calculate_rewards(actions)

        # === PHASE 7b: Update prev_potential AFTER reward is computed ════════════
        # Must happen AFTER _calculate_rewards() reads self.prev_potential for F,
        # and BEFORE the next step overwrites actions.
        if self.shaping_enabled and self.shaping_weight > 0.0:
            for agent_id in range(self.n_agents):
                for sku in range(self.n_skus):
                    rec = self._get_heuristic_recommendation(agent_id, sku)
                    action_sku = float(actions[agent_id][sku])
                    self.prev_potential[agent_id][sku] = self._compute_potential(rec, action_sku)
        
        # === PHASE 8: Get observations ===
        observations = self._get_observations()
        
        # === PHASE 9: Check termination ===
        # Check max days
        time_limit_reached = self.current_day >= self.max_days
        
       

        done = time_limit_reached
        dones = {i: done for i in range(self.n_agents)}
        
        # === PHASE 10: Compute step and episode service level ===
        # Order-count Fill Rate: orders fully met from on-hand / total orders placed
        # (Quantity-based step_demand_met kept for rolling obs signal — not changed)
        step_orders_placed     = 0
        step_orders_from_stock = 0
        for retailer_id in self.retailer_ids:
            for sku in range(self.n_skus):
                step_orders_placed     += self.step_orders_placed[retailer_id][sku]
                step_orders_from_stock += self.step_orders_from_stock[retailer_id][sku]

        step_sl = (
            step_orders_from_stock / step_orders_placed
            if step_orders_placed > 0 else 1.0
        )

        # Accumulate into episode fill rate
        self.ep_orders_placed     += step_orders_placed
        self.ep_orders_from_stock += step_orders_from_stock
        episode_fill_rate = (
            self.ep_orders_from_stock / self.ep_orders_placed
            if self.ep_orders_placed > 0 else 1.0
        )

        # Cycle Service Level per DC
        dc_cycle_sl = {}
        for dc_id in self.dc_ids:
            received   = self.dc_cycle_sl_orders_received[dc_id]
            no_backlog = self.dc_cycle_sl_orders_no_backlog[dc_id]
            dc_cycle_sl[dc_id] = (no_backlog / received * 100.0) if received > 0 else 100.0

        infos = {i: {
            'step_service_level':    step_sl,          # order-count fill rate this step
            'episode_service_level': episode_fill_rate, # cumulative order-count fill rate
            'dc_cycle_service_level': dc_cycle_sl,
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
                clipped[agent_id] = np.clip(action, 0, 5000)   # DC: 0 to 5000 units
            else:
                clipped[agent_id] = np.clip(action, 0, 10)    # Retailer: 0 to 100 units
        return clipped



    
    def _process_retailer_orders(self, actions: Dict[int, np.ndarray]) -> Dict:
        """
        Retailers place orders to their assigned DC.

        Each retailer is permanently assigned to one DC (see dc_assignments in config).
        action shape is 3D: action[sku] = units to order from the assigned DC.

        Returns:
            retailer_orders: {dc_id: {retailer_id: {sku: qty}}}
        """
        retailer_orders = {dc_id: {} for dc_id in self.dc_ids}
        
        for retailer_id in self.retailer_ids:
            action = actions[retailer_id]  # Shape: (3,) — one value per SKU
            assigned_dc = self.retailer_to_dc[retailer_id]
            
            # action[sku] → order qty for that SKU from assigned DC
            retailer_orders[assigned_dc][retailer_id] = {
                sku: float(action[sku]) for sku in range(self.n_skus)
            }
            
            # Track total order for logging
            self.last_actions[retailer_id] = float(np.sum(action[:self.n_skus]))
        
        return retailer_orders
        
    def get_orders(self) -> Dict[int, float]:
        """Return total quantity ordered by each agent in the last step."""
        return self.last_actions
    
    def _fulfill_retailer_orders(self, retailer_orders: Dict):
        """
        DCs fulfill retailer orders per-retailer with a per-retailer backlog.

        Fulfillment rules:
          - If the DC has enough stock: ship the full order; no backlog added.
          - If the DC is short: ship all available stock proportionally; the
            unfulfilled remainder is added to dc_retailer_backlog[dc_id][retailer_id]
            (per-retailer, NOT aggregated into a single flat DC backlog).

        Cycle Service Level is updated here:
          - Each (retailer, sku) order that arrives counts as one order received.
          - An order is "fulfilled without backlog" only if the entire qty is shipped
            immediately (no partial).

        Args:
            retailer_orders: {dc_id: {retailer_id: {sku: qty}}}
        """
        for dc_id in self.dc_ids:
            dc_orders = retailer_orders[dc_id]

            for sku in range(self.n_skus):
                # Total requested qty for this SKU across all retailers of this DC
                total_demand = sum(
                    orders[sku] for orders in dc_orders.values()
                )

                if total_demand == 0:
                    continue  # No orders this step for this sku

                available = self.inventory[dc_id][sku]

                if available >= total_demand:
                    # ── Full fulfillment ──
                    for retailer_id, orders in dc_orders.items():
                        qty = orders[sku]
                        if qty > 0:
                            self._ship_to_retailer(dc_id, retailer_id, sku, qty,
                                                   lead_time_sample=True)
                            self.inventory[dc_id][sku] -= qty
                            # Cycle SL: order fulfilled without any backlog
                            self.dc_cycle_sl_orders_received[dc_id]   += 1
                            self.dc_cycle_sl_orders_no_backlog[dc_id] += 1
                else:
                    # ── Proportional rationing — per-retailer backlog ──
                    for retailer_id, orders in dc_orders.items():
                        qty_ordered = orders[sku]
                        if qty_ordered == 0:
                            continue

                        # Count as one order received
                        self.dc_cycle_sl_orders_received[dc_id] += 1

                        ratio = qty_ordered / total_demand
                        # Each retailer gets a proportional share of what's available
                        fulfilled_qty   = available * ratio
                        unfulfilled_qty = qty_ordered - fulfilled_qty

                        if fulfilled_qty > 0:
                            self._ship_to_retailer(dc_id, retailer_id, sku,
                                                   fulfilled_qty, lead_time_sample=True)

                        if unfulfilled_qty > 0:
                            # Add to per-retailer backlog (NOT the flat DC backlog)
                            self.dc_retailer_backlog[dc_id][retailer_id][sku] += unfulfilled_qty
                            # This order triggered a backlog — do NOT increment no_backlog counter

                    # DC inventory for this sku is fully depleted
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
        """Process all pipeline arrivals for all agents.

        For DCs: after adding arriving stock, automatically fulfil any
        outstanding per-retailer backlog (dc_retailer_backlog) before new
        retailer orders are placed.  This ensures that retailers who were
        shorted in a previous step receive their goods as soon as the DC
        is restocked.
        """
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

            # ── DC restocking: auto-fulfill outstanding per-retailer backlogs ──
            if agent_id in self.dc_ids:
                dc_id = agent_id
                for retailer_id in self.retailer_ids:
                    for sku in range(self.n_skus):
                        owed = self.dc_retailer_backlog[dc_id][retailer_id][sku]
                        if owed <= 0:
                            continue
                        avail = self.inventory[dc_id][sku]
                        if avail <= 0:
                            continue
                        # Ship as much of the backlog as currently available
                        ship_qty = min(owed, avail)
                        self._ship_to_retailer(dc_id, retailer_id, sku,
                                               ship_qty, lead_time_sample=True)
                        self.inventory[dc_id][sku]                            -= ship_qty
                        self.dc_retailer_backlog[dc_id][retailer_id][sku]    -= ship_qty
    
    def _process_customer_demand(self):
        """Process customer demand at retailer level.

        Service-Level calculation design:
          - `step_demand`     : today's FRESH customer demand (denominator of SL).
          - `step_demand_met` : units of TODAY'S fresh demand actually fulfilled
                               (numerator of Fill-Rate SL).  Backlog clearance
                               is intentionally excluded from the numerator so
                               that SL measures the fraction of new demand served,
                               not historical recovery.
          - `demand_fulfilled`: total units served this step (fresh + backlog
                               clearance) — used for the SALE REVENUE bonus only.

        Backlog is still cleared first (before serving fresh demand) so that
        agents learn that restocking recovers past shortfalls.
        """
        # Reset per-step trackers
        for retailer_id in self.retailer_ids:
            self.demand_fulfilled[retailer_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand_met[retailer_id]   = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand[retailer_id]        = np.zeros(self.n_skus, dtype=np.float32)
            # Reset order-count fill rate step trackers
            self.step_orders_placed[retailer_id]     = {sku: 0 for sku in range(self.n_skus)}
            self.step_orders_from_stock[retailer_id] = {sku: 0 for sku in range(self.n_skus)}

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
                # Backlog clearance earns sale revenue but is NOT counted in SL
                # numerator (step_demand_met) because it satisfies PAST demand,
                # not today's fresh demand.
                if self.backlog[retailer_id][sku] > 0 and avail > 0:
                    backlog_cleared = min(self.backlog[retailer_id][sku], avail)
                    avail -= backlog_cleared
                    self.backlog[retailer_id][sku] -= backlog_cleared
                    # Only goes to demand_fulfilled (revenue), NOT step_demand_met (SL)
                    self.demand_fulfilled[retailer_id][sku] += backlog_cleared

                # ── Order-count Fill Rate: check BEFORE touching inventory for fresh demand ──
                # avail at this point = on-hand stock available for today's fresh order
                # (after old backlog was already cleared above).
                if demand[sku] > 0:
                    self.step_orders_placed[retailer_id][sku] = 1  # one order this step per sku
                    if avail >= demand[sku]:
                        # Demand fully covered from on-hand — no backlog needed
                        self.step_orders_from_stock[retailer_id][sku] = 1
                    else:
                        # Demand partially or fully unmet — backlog will be added
                        self.step_orders_from_stock[retailer_id][sku] = 0

                # ── Step B: Serve today's fresh customer demand ──
                if avail >= demand[sku]:
                    avail -= demand[sku]
                    # Fresh demand served: counted in BOTH revenue AND SL numerator
                    self.demand_fulfilled[retailer_id][sku] += demand[sku]
                    self.step_demand_met[retailer_id][sku]   += demand[sku]
                else:
                    # Partial: avail < demand
                    self.demand_fulfilled[retailer_id][sku] += avail
                    self.step_demand_met[retailer_id][sku]   += avail
                    shortage = demand[sku] - avail
                    avail = 0
                    self.backlog[retailer_id][sku] += shortage

                self.inventory[retailer_id][sku] = avail

            # ── Update rolling SL history (last 7 steps) for observation feature ──
            # Use only fresh-demand numerator/denominator for the rolling SL signal.
            step_demanded  = float(np.sum(self.step_demand[retailer_id]))
            step_fresh_met = float(np.sum(self.step_demand_met[retailer_id]))
            self.sl_history[retailer_id]['fulfilled'].append(step_fresh_met)
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

            for sku in range(self.n_skus):
                holding  = self.H_dc[dc_id][sku] * self.inventory[dc_id][sku]
                # DC backlog cost: penalise the total quantity still owed to retailers
                # (dc_retailer_backlog replaced the old flat backlog[dc_id] which is now 0)
                total_owed_sku = sum(
                    self.dc_retailer_backlog[dc_id][r_id][sku]
                    for r_id in self.dc_assignments[dc_id]
                )
                backlog  = self.B_dc[dc_id][sku] * total_owed_sku

                order_qty = actions[dc_id][sku]
                if order_qty > 0:
                    market_price = self.market_prices[sku]
                    var_cost = market_price * order_qty
                    ordering = self.C_fixed_dc[dc_id][sku] + var_cost
                else:
                    ordering = 0.0

                # Excess inventory penalty for DCs: discourages stocking far beyond
                # what's needed to cover one lead-time cycle of retailer demand.
                if self.excess_penalty_dc > 0.0 and self._dc_target_stock is not None:
                    excess = max(0.0, self.inventory[dc_id][sku] - self._dc_target_stock[sku])
                    ordering += self.excess_penalty_dc * excess

                total_cost += holding + backlog + ordering

            rewards[dc_id] = -total_cost

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

                # Excess inventory penalty: discourages holding far above target level.
                # Only kicks in when inventory > target_stock (so no impact during low-stock).
                if self.excess_penalty_retailer > 0.0 and self._retailer_target_stock is not None:
                    excess = max(0.0, self.inventory[retailer_id][sku] - self._retailer_target_stock[sku])
                    ordering += self.excess_penalty_retailer * excess

                total_cost += holding + backlog + ordering

            # Sale revenue bonus: +R per unit of customer demand actually fulfilled
            retailer_sale_revenue = self.sale_revenue_retailer * float(
                np.sum(self.demand_fulfilled[retailer_id])
            )

            rewards[retailer_id] = -total_cost + retailer_sale_revenue

        # === Step 4: Look-Back Potential-Based Reward Shaping ════════════════
        # F(s_{t-1},a_{t-1} → s_t,a_t) = Φ(s_t,a_t) − Φ(s_{t-1},a_{t-1})
        # R'(agent) = R(agent) + shaping_weight × Σ_sku  F_sku
        # Computed BEFORE normalisation so that k is directly comparable with
        # unnormalised cost magnitudes; same /max_days applied to both.
        if self.shaping_enabled and self.shaping_weight > 0.0:
            for agent_id in range(self.n_agents):
                shaping_bonus = 0.0
                for sku in range(self.n_skus):
                    rec        = self._get_heuristic_recommendation(agent_id, sku)
                    action_sku = float(actions[agent_id][sku])
                    phi_curr   = self._compute_potential(rec, action_sku)
                    phi_prev   = self.prev_potential[agent_id][sku]
                    shaping_bonus += phi_curr - phi_prev  # F for this SKU
                rewards[agent_id] += self.shaping_weight * shaping_bonus

        # === Normalize all rewards by episode length ===
        # Prevents quadratic reward blow-up from cumulative backlog costs in long episodes.
        # Per-step reward stays in a stable range regardless of max_days.
        # All relative cost ratios are preserved — incentive structure is unchanged.
        norm = float(self.max_days)
        rewards = {agent_id: r / norm for agent_id, r in rewards.items()}

        # === Alpha-weighted local + global reward mixing ===
        # Blends each agent's purely local reward with the system-wide mean reward.
        # This encourages cooperative behaviour without losing the sharp local gradient.
        # Controlled by self.reward_alpha (set via config key 'rewards.alpha'):
        #   0.0 → purely local (default, backward-compatible)
        #   0.2 → recommended starting point
        #   1.0 → purely global
        if self.reward_alpha > 0.0:
            global_mean = sum(rewards.values()) / len(rewards)
            rewards = {
                agent_id: (1 - self.reward_alpha) * r + self.reward_alpha * global_mean
                for agent_id, r in rewards.items()
            }

        return rewards

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1 – Teacher Heuristic: Dynamic Base-Stock recommendation
    # ═══════════════════════════════════════════════════════════════════════════
    def _get_heuristic_recommendation(self, agent_id: int, sku: int) -> float:
        """
        Dynamic Base-Stock (Order-Up-To) heuristic for a given agent and SKU.

        Formula:
            a†(s) = max(0,  μ_recent × L  +  z × σ_recent × √L  − IP)

        where:
            μ_recent  = rolling mean demand over the last `shaping_window` steps
            σ_recent  = rolling std  demand over the last `shaping_window` steps
            L         = lead time (max, conservative)
            z         = safety factor (≈ 1.28 → ~90 % service target)
            IP        = inventory position = on-hand + pipeline − backlog

        For DCs the "demand" is the aggregate retailer demand observed via
        dc_fulfilled (units shipped); for retailers it is customer demand.
        The heuristic is purely based on already-visible state — no look-ahead.
        """
        z = 1.95  # safety factor (~90% cycle service level)

        if agent_id in self.dc_ids:
            # ── DC: use historical shipments to retailers as demand proxy ──────
            # Fallback to global demand_mean × n_assigned_retailers when
            # not enough history is available.
            assigned_retailers = self.dc_assignments[agent_id]
            n_assigned = max(len(assigned_retailers), 1)
            mu   = float(self.demand_mean[sku]) * n_assigned
            sigma = float(self.demand_std[sku])  * n_assigned

            lead_time = float(self.lt_supplier_to_dc_max)  # conservative upper bound

            # Inventory position = on-hand - owed to retailers + in-pipeline
            on_hand  = float(self.inventory[agent_id][sku])
            owed     = sum(
                self.dc_retailer_backlog[agent_id][r_id][sku]
                for r_id in assigned_retailers
            )
            pipeline = sum(
                o['qty'] for o in self.pipeline[agent_id] if o['sku'] == sku
            )
            ip = on_hand - owed + pipeline

        else:
            # ── Retailer: use recent customer demand history ──────────────────
            retailer_idx = self.retailer_ids.index(agent_id)
            if (retailer_idx < len(self.demand_history[sku]) and
                    len(self.demand_history[sku][retailer_idx]) >= 2):
                hist = self.demand_history[sku][retailer_idx][-self.shaping_window:]
                mu    = float(np.mean(hist))
                sigma = float(np.std(hist)) if len(hist) > 1 else float(self.demand_std[sku])
            else:
                mu    = float(self.demand_mean[sku])
                sigma = float(self.demand_std[sku])

            lead_time = 7  # fixed 1-day

            on_hand  = float(self.inventory[agent_id][sku])
            backlog  = float(self.backlog[agent_id][sku])
            pipeline = sum(
                o['qty'] for o in self.pipeline[agent_id] if o['sku'] == sku
            )
            ip = on_hand - backlog + pipeline

        # Order-up-to level: mean demand over lead time + safety buffer
        out_level = mu * lead_time + z * sigma * float(np.sqrt(max(lead_time, 1)))
        recommendation = max(0.0, out_level - ip)
        return float(recommendation)

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2 – Potential Function Φ(s, a)
    # ═══════════════════════════════════════════════════════════════════════════
    def _compute_potential(self, recommendation: float, action: float) -> float:
        """
        Φ(s, a) = -k · |a†(s) - a|

        The potential is maximised (least negative) when the agent's action
        exactly matches the heuristic recommendation.  It decreases linearly
        as the agent deviates in either direction.
        """
        return -self.shaping_k * abs(recommendation - action)

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5 – Annealing helper (called by the external training loop)
    # ═══════════════════════════════════════════════════════════════════════════
    def decay_shaping_weight(self, decay_rate: float = 0.995) -> float:
        """
        Multiply the shaping weight by `decay_rate` (clamped to [0, 1]).

        Call once per episode from the training loop:
            env.decay_shaping_weight(decay_rate=0.998)

        Args:
            decay_rate: Multiplicative factor in (0, 1].  Default 0.995
                        anneals from 1.0 → ~0.007 over 1,000 episodes.

        Returns:
            Updated shaping_weight (for logging).
        """
        self.shaping_weight = max(0.0, self.shaping_weight * decay_rate)
        return self.shaping_weight

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
            
            # 2. DC total owed to retailers (per-retailer backlog sum for this SKU)
            # Replaces flat backlog[dc_id] which is always 0 after the per-retailer backlog migration.
            # A non-zero value means the DC shipped partial orders; tells the agent to reorder fast.
            total_owed_sku = sum(
                self.dc_retailer_backlog[dc_id][r_id][sku]
                for r_id in self.dc_assignments[dc_id]
            )
            obs.append(total_owed_sku / 100.0)
            
            # 3-5. Pipeline bins aligned with Uniform[7, 14] lead time (new config).
            # Arrivals expected between day+7 and day+14.
            pipeline_7_9 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 7 <= o['arrival_day'] <= self.current_day + 9)
            pipeline_10_12 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 10 <= o['arrival_day'] <= self.current_day + 12)
            pipeline_13_14 = sum(o['qty'] for o in self.pipeline[dc_id]
                               if o['sku'] == sku and self.current_day + 13 <= o['arrival_day'] <= self.current_day + 14)
            
            obs.append(pipeline_7_9  / 500.0)   # 3-day window early  (norm by DC max order cap)
            obs.append(pipeline_10_12 / 500.0)  # 3-day window mid
            obs.append(pipeline_13_14 / 500.0)  # 2-day window late
            
            # 6. Total DC pipeline
            # Max lead time=14, max DC order=500/step → ceiling ~7000; use 7000 for safety.
            total_pipeline = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku)
            obs.append(total_pipeline / 7000.0)
            
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
