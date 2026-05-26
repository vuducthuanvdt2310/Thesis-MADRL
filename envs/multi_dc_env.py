import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
import os

class MultiDCInventoryEnv:
    def __init__(self, config_path: str = 'configs/multi_dc_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.n_dcs = self.config['environment'].get('n_dcs', 2)
        self.n_retailers = self.config['environment'].get('n_retailers', 3)
        self.n_agents = self.n_dcs + self.n_retailers
        self.n_skus = self.config['environment']['n_skus']
        self.dc_ids = list(range(self.n_dcs))
        self.retailer_ids = list(range(self.n_dcs, self.n_agents))
        raw_assignments = self.config.get('dc_assignments', None)
        if raw_assignments:
            self.dc_assignments = {}
            for dc_id in range(self.n_dcs):
                key = f'dc_{dc_id}'
                if key in raw_assignments:
                    self.dc_assignments[dc_id] = [self.n_dcs + idx for idx in raw_assignments[key]]
                else:
                    self.dc_assignments[dc_id] = []
        else:
            per_dc = self.n_retailers // self.n_dcs
            self.dc_assignments = {}
            for dc_id in range(self.n_dcs):
                start = self.n_dcs + dc_id * per_dc
                end = self.n_dcs + (dc_id + 1) * per_dc if dc_id < self.n_dcs - 1 else self.n_agents
                self.dc_assignments[dc_id] = list(range(start, end))
        self.retailer_to_dc: Dict[int, int] = {}
        for dc_id, r_list in self.dc_assignments.items():
            for r_id in r_list:
                self.retailer_to_dc[r_id] = dc_id
        self._compute_retailer_demand_stats()
        self.lt_supplier_to_dc_min = self.config['environment']['lead_time']['supplier_to_dc']['min']
        self.lt_supplier_to_dc_max = self.config['environment']['lead_time']['supplier_to_dc']['max']
        self.lt_dc_to_retailer = 1
        self.max_days = self.config['environment']['max_days']
        self.current_day = 0
        self._load_cost_parameters()
        self._load_reward_parameters()
        self._load_constraints()
        self.inventory = {}
        self.backlog = {}
        self.pipeline = {}
        self.last_actions = {i: 0.0 for i in range(self.n_agents)}
        self.demand_fulfilled = {}
        self.step_demand = {}
        self.step_demand_met = {}
        self.dc_retailer_backlog = {}
        self.dc_cycle_sl_orders_received = {}
        self.dc_cycle_sl_orders_no_backlog = {}
        self.sl_history = {}
        self.episode_sl = 0.0
        self.step_orders_placed = {}
        self.step_orders_from_stock = {}
        self.ep_orders_placed = 0
        self.ep_orders_from_stock = 0
        self.market_prices = None
        self.price_history = None
        self.demand_history = None
        self.prev_potential: Dict[int, Dict[int, float]] = {
            agent_id: {sku: 0.0 for sku in range(self.n_skus)}
            for agent_id in range(self.n_agents)
        }
        self._define_spaces()
        self.agent_num = self.n_agents
        self.agents = [f'agent_{i}' for i in range(self.n_agents)]

    def _load_cost_parameters(self):
        costs = self.config['costs']
        self.H_dc = np.array(costs['holding_cost_dc'], dtype=np.float32)
        self.B_dc = np.array(costs['backlog_cost_dc'], dtype=np.float32)
        self.C_fixed_dc = np.array(costs['fixed_order_cost_dc'], dtype=np.float32)
        self.H_retailer = np.array(costs['holding_cost_retailer'], dtype=np.float32)
        self.B_retailer = np.array(costs['backlog_cost_retailer'], dtype=np.float32)
        self.C_fixed_retailer = np.array(costs['fixed_order_cost_retailer'], dtype=np.float32)
        self.C_var_retailer = np.array(costs['variable_cost_retailer'], dtype=np.float32)
        self.base_market_price = np.array(self.config['pricing']['base_price'], dtype=np.float32)
        self.price_volatility = self.config['pricing']['volatility']
        self.price_bounds = {
            'min': np.array(self.config['pricing']['min_price'], dtype=np.float32),
            'max': np.array(self.config['pricing']['max_price'], dtype=np.float32)
        }

    def _load_reward_parameters(self):
        rewards_cfg = self.config.get('rewards', {})
        self.termination_penalty = float(rewards_cfg.get('termination_penalty', 0.0))
        self.sale_revenue_retailer = float(rewards_cfg.get('sale_revenue_retailer', 5.0))
        self.reward_alpha = float(rewards_cfg.get('alpha', 0.0))
        self.target_stock_days_retailer = float(rewards_cfg.get('target_stock_days_retailer', 7))
        self.target_stock_days_dc = float(rewards_cfg.get('target_stock_days_dc', 14))
        self.excess_penalty_retailer = float(rewards_cfg.get('excess_penalty_retailer', 0.0))
        self.excess_penalty_dc = float(rewards_cfg.get('excess_penalty_dc', 0.0))
        self._retailer_target_stock = None
        self._dc_target_stock = None
        self.holding_weight = float(rewards_cfg.get('holding_weight', 0.3))
        self.backlog_weight = float(rewards_cfg.get('backlog_weight', 1.5))
        self.ordering_weight = float(rewards_cfg.get('ordering_weight', 0.1))
        shaping_cfg = rewards_cfg.get('heuristic_shaping', {})
        self.shaping_k = float(shaping_cfg.get('k', 0.5))
        self.shaping_window = int(shaping_cfg.get('demand_window', 14))
        self.shaping_weight = float(shaping_cfg.get('initial_weight', 1.0))
        self.shaping_enabled = bool(shaping_cfg.get('enabled', True))

    def _load_constraints(self):
        constraints_cfg = self.config.get('constraints', {})
        if 'safety_stock_threshold' in constraints_cfg:
            self.safety_stock_threshold = np.array(constraints_cfg['safety_stock_threshold'], dtype=np.float32)
        else:
            self.safety_stock_threshold = np.zeros((self.n_retailers, self.n_skus), dtype=np.float32)
        self.safety_stock_penalty = float(constraints_cfg.get('safety_stock_penalty', 0.0))

    def _define_spaces(self):
        self.obs_dim_dc = self.n_skus * 9 + 1
        self.obs_space_dc = spaces.Box(0, 1, (self.obs_dim_dc,), dtype=np.float32)
        self.obs_dim_retailer = self.n_skus * 7 + 1
        self.obs_space_retailer = spaces.Box(0, 1, (self.obs_dim_retailer,), dtype=np.float32)
        self.action_dim = 3
        self.action_space_dc = spaces.Box(0, 100, (self.action_dim,), dtype=np.float32)
        self.action_space_retailer = spaces.Box(0, 15, (self.action_dim,), dtype=np.float32)
        self.action_space = spaces.Box(0, 100, (self.action_dim,), dtype=np.float32)
        self.observation_spaces = {
            i: self.obs_space_dc if i in self.dc_ids else self.obs_space_retailer
            for i in range(self.n_agents)
        }
        self.action_spaces = {
            i: self.action_space_dc if i in self.dc_ids else self.action_space_retailer
            for i in range(self.n_agents)
        }

    def reset(self) -> Dict[int, np.ndarray]:
        self.current_day = 0
        for agent_id in range(self.n_agents):
            if agent_id in self.dc_ids:
                self.inventory[agent_id] = np.full(self.n_skus, 500.0, dtype=np.float32)
            else:
                self.inventory[agent_id] = np.full(self.n_skus, 30.0, dtype=np.float32)
            self.backlog[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.pipeline[agent_id] = []
        self.market_prices = self.base_market_price.copy()
        self.price_history = [[] for _ in range(self.n_skus)]
        for sku in range(self.n_skus):
            self.price_history[sku].append(self.market_prices[sku])
        self.demand_history = [[] for _ in range(self.n_skus)]
        for agent_id in range(self.n_agents):
            self.demand_fulfilled[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand_met[agent_id] = np.zeros(self.n_skus, dtype=np.float32)
        for dc_id in self.dc_ids:
            self.dc_retailer_backlog[dc_id] = {
                r_id: {sku: 0.0 for sku in range(self.n_skus)}
                for r_id in self.retailer_ids
            }
            self.dc_cycle_sl_orders_received[dc_id] = 0
            self.dc_cycle_sl_orders_no_backlog[dc_id] = 0
        for retailer_id in self.retailer_ids:
            self.sl_history[retailer_id] = {'fulfilled': [], 'demanded': []}
        self.episode_sl = 0.0
        for retailer_id in self.retailer_ids:
            self.step_orders_placed[retailer_id] = {sku: 0 for sku in range(self.n_skus)}
            self.step_orders_from_stock[retailer_id] = {sku: 0 for sku in range(self.n_skus)}
        self.ep_orders_placed = 0
        self.ep_orders_from_stock = 0
        for agent_id in range(self.n_agents):
            for sku in range(self.n_skus):
                self.prev_potential[agent_id][sku] = 0.0
        self._retailer_target_stock = np.full(self.n_skus, 30.0, dtype=np.float32)
        avg_retailers_per_dc = self.n_retailers / self.n_dcs
        self._dc_target_stock = (self.target_stock_days_dc * avg_retailers_per_dc * self.demand_mean)
        observations = self._get_observations()
        return observations

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        self.current_day += 1
        actions = self._clip_actions(actions)
        self.last_actions = {i: 0.0 for i in range(self.n_agents)}
        retailer_orders = self._process_retailer_orders(actions)
        self._fulfill_retailer_orders(retailer_orders)
        self._process_dc_orders(actions)
        self._process_arrivals()
        self._process_customer_demand()
        self._update_market_prices()
        rewards = self._calculate_rewards(actions)
        if self.shaping_enabled and self.shaping_weight > 0.0:
            for agent_id in range(self.n_agents):
                for sku in range(self.n_skus):
                    rec = self._get_heuristic_recommendation(agent_id, sku)
                    action_sku = float(actions[agent_id][sku])
                    self.prev_potential[agent_id][sku] = self._compute_potential(rec, action_sku)
        observations = self._get_observations()
        time_limit_reached = self.current_day >= self.max_days
        done = time_limit_reached
        dones = {i: done for i in range(self.n_agents)}
        step_orders_placed = 0
        step_orders_from_stock = 0
        for retailer_id in self.retailer_ids:
            for sku in range(self.n_skus):
                step_orders_placed += self.step_orders_placed[retailer_id][sku]
                step_orders_from_stock += self.step_orders_from_stock[retailer_id][sku]
        step_sl = (step_orders_from_stock / step_orders_placed if step_orders_placed > 0 else 1.0)
        self.ep_orders_placed += step_orders_placed
        self.ep_orders_from_stock += step_orders_from_stock
        episode_fill_rate = (self.ep_orders_from_stock / self.ep_orders_placed if self.ep_orders_placed > 0 else 1.0)
        dc_cycle_sl = {}
        for dc_id in self.dc_ids:
            received = self.dc_cycle_sl_orders_received[dc_id]
            no_backlog = self.dc_cycle_sl_orders_no_backlog[dc_id]
            dc_cycle_sl[dc_id] = (no_backlog / received * 100.0) if received > 0 else 100.0
        infos = {i: {
            'step_service_level': step_sl,
            'episode_service_level': episode_fill_rate,
            'dc_cycle_service_level': dc_cycle_sl,
        } for i in range(self.n_agents)}
        return observations, rewards, dones, infos

    def _clip_actions(self, actions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        clipped = {}
        for agent_id, action in actions.items():
            if agent_id in self.dc_ids:
                clipped[agent_id] = np.clip(action, 0, 100)
            else:
                clipped[agent_id] = np.clip(action, 0, 15)
        return clipped

    def _process_retailer_orders(self, actions: Dict[int, np.ndarray]) -> Dict:
        retailer_orders = {dc_id: {} for dc_id in self.dc_ids}
        for retailer_id in self.retailer_ids:
            action = actions[retailer_id]
            assigned_dc = self.retailer_to_dc[retailer_id]
            retailer_orders[assigned_dc][retailer_id] = {
                sku: float(action[sku]) for sku in range(self.n_skus)
            }
            self.last_actions[retailer_id] = float(np.sum(action[:self.n_skus]))
        return retailer_orders

    def get_orders(self) -> Dict[int, float]:
        return self.last_actions

    def _fulfill_retailer_orders(self, retailer_orders: Dict):
        for dc_id in self.dc_ids:
            dc_orders = retailer_orders[dc_id]
            for sku in range(self.n_skus):
                total_demand = sum(orders[sku] for orders in dc_orders.values())
                if total_demand == 0:
                    continue
                available = self.inventory[dc_id][sku]
                if available >= total_demand:
                    for retailer_id, orders in dc_orders.items():
                        qty = orders[sku]
                        if qty > 0:
                            self._ship_to_retailer(dc_id, retailer_id, sku, qty, lead_time_sample=True)
                            self.inventory[dc_id][sku] -= qty
                            self.dc_cycle_sl_orders_received[dc_id] += 1
                            self.dc_cycle_sl_orders_no_backlog[dc_id] += 1
                else:
                    for retailer_id, orders in dc_orders.items():
                        qty_ordered = orders[sku]
                        if qty_ordered == 0:
                            continue
                        self.dc_cycle_sl_orders_received[dc_id] += 1
                        ratio = qty_ordered / total_demand
                        fulfilled_qty = available * ratio
                        unfulfilled_qty = qty_ordered - fulfilled_qty
                        if fulfilled_qty > 0:
                            self._ship_to_retailer(dc_id, retailer_id, sku, fulfilled_qty, lead_time_sample=True)
                        if unfulfilled_qty > 0:
                            self.dc_retailer_backlog[dc_id][retailer_id][sku] += unfulfilled_qty
                    self.inventory[dc_id][sku] = 0

    def _ship_to_retailer(self, dc_id: int, retailer_id: int, sku: int, qty: float, lead_time_sample: bool = True):
        if lead_time_sample:
            lead_time = self.lt_dc_to_retailer
        else:
            lead_time = 0
        arrival_day = self.current_day + lead_time
        self.pipeline[retailer_id].append({
            'sku': sku,
            'qty': qty,
            'arrival_day': arrival_day,
            'source': f'DC_{dc_id}'
        })

    def _process_dc_orders(self, actions: Dict[int, np.ndarray]):
        for dc_id in self.dc_ids:
            action = actions[dc_id]
            for sku in range(self.n_skus):
                order_qty = float(action[sku])
                if order_qty > 0:
                    lead_time = np.random.randint(self.lt_supplier_to_dc_min, self.lt_supplier_to_dc_max + 1)
                    arrival_day = self.current_day + lead_time
                    self.pipeline[dc_id].append({
                        'sku': sku,
                        'qty': order_qty,
                        'arrival_day': arrival_day,
                        'source': 'supplier'
                    })
            self.last_actions[dc_id] = np.sum(action[:self.n_skus])

    def _process_arrivals(self):
        for agent_id in range(self.n_agents):
            arrived = [order for order in self.pipeline[agent_id] if order['arrival_day'] == self.current_day]
            for order in arrived:
                self.inventory[agent_id][order['sku']] += order['qty']
            self.pipeline[agent_id] = [order for order in self.pipeline[agent_id] if order['arrival_day'] > self.current_day]
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
                        ship_qty = min(owed, avail)
                        self._ship_to_retailer(dc_id, retailer_id, sku, ship_qty, lead_time_sample=True)
                        self.inventory[dc_id][sku] -= ship_qty
                        self.dc_retailer_backlog[dc_id][retailer_id][sku] -= ship_qty

    def _process_customer_demand(self):
        for retailer_id in self.retailer_ids:
            self.demand_fulfilled[retailer_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand_met[retailer_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_demand[retailer_id] = np.zeros(self.n_skus, dtype=np.float32)
            self.step_orders_placed[retailer_id] = {sku: 0 for sku in range(self.n_skus)}
            self.step_orders_from_stock[retailer_id] = {sku: 0 for sku in range(self.n_skus)}
        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            demand = self._get_demand(retailer_idx, self.current_day)
            self.step_demand[retailer_id] = demand.copy()
            for sku in range(self.n_skus):
                if len(self.demand_history[sku]) <= retailer_idx:
                    self.demand_history[sku].append([])
                self.demand_history[sku][retailer_idx].append(demand[sku])
            for sku in range(self.n_skus):
                avail = self.inventory[retailer_id][sku]
                if self.backlog[retailer_id][sku] > 0 and avail > 0:
                    backlog_cleared = min(self.backlog[retailer_id][sku], avail)
                    avail -= backlog_cleared
                    self.backlog[retailer_id][sku] -= backlog_cleared
                    self.demand_fulfilled[retailer_id][sku] += backlog_cleared
                if demand[sku] > 0:
                    self.step_orders_placed[retailer_id][sku] = 1
                    if avail >= demand[sku]:
                        self.step_orders_from_stock[retailer_id][sku] = 1
                    else:
                        self.step_orders_from_stock[retailer_id][sku] = 0
                if avail >= demand[sku]:
                    avail -= demand[sku]
                    self.demand_fulfilled[retailer_id][sku] += demand[sku]
                    self.step_demand_met[retailer_id][sku] += demand[sku]
                else:
                    self.demand_fulfilled[retailer_id][sku] += avail
                    self.step_demand_met[retailer_id][sku] += avail
                    shortage = demand[sku] - avail
                    avail = 0
                    self.backlog[retailer_id][sku] += shortage
                self.inventory[retailer_id][sku] = avail
            step_demanded = float(np.sum(self.step_demand[retailer_id]))
            step_fresh_met = float(np.sum(self.step_demand_met[retailer_id]))
            self.sl_history[retailer_id]['fulfilled'].append(step_fresh_met)
            self.sl_history[retailer_id]['demanded'].append(step_demanded)
            if len(self.sl_history[retailer_id]['fulfilled']) > 7:
                self.sl_history[retailer_id]['fulfilled'].pop(0)
                self.sl_history[retailer_id]['demanded'].pop(0)

    def _compute_retailer_demand_stats(self):
        self.demand_mean = np.array(self.config['demand']['mean'], dtype=np.float32)
        self.demand_std = np.maximum(np.array(self.config['demand']['std'], dtype=np.float32), 0.1)

    def _get_demand(self, retailer_idx: int, day: int) -> np.ndarray:
        demand = np.random.normal(self.demand_mean, self.demand_std).astype(np.float32)
        demand = np.maximum(demand, 0.0)
        return demand

    def _update_market_prices(self):
        for sku in range(self.n_skus):
            change = np.random.normal(0, self.price_volatility * self.base_market_price[sku])
            new_price = self.market_prices[sku] + change
            reversion_force = 0.1 * (self.base_market_price[sku] - new_price)
            new_price += reversion_force
            new_price = np.clip(new_price, self.price_bounds['min'][sku], self.price_bounds['max'][sku])
            self.market_prices[sku] = new_price
            self.price_history[sku].append(new_price)

    def _calculate_rewards(self, actions: Dict[int, np.ndarray]) -> Dict[int, float]:
        rewards = {}
        for dc_id in self.dc_ids:
            total_cost = 0.0
            for sku in range(self.n_skus):
                holding = self.H_dc[dc_id][sku] * self.inventory[dc_id][sku] * self.holding_weight
                total_owed_sku = sum(self.dc_retailer_backlog[dc_id][r_id][sku] for r_id in self.dc_assignments[dc_id])
                backlog = self.B_dc[dc_id][sku] * total_owed_sku * self.backlog_weight
                order_qty = actions[dc_id][sku]
                if order_qty > 0:
                    market_price = self.market_prices[sku]
                    var_cost = market_price * order_qty
                    ordering = (self.C_fixed_dc[dc_id][sku] + var_cost) * self.ordering_weight
                else:
                    ordering = 0.0
                if self.excess_penalty_dc > 0.0 and self._dc_target_stock is not None:
                    target = self._dc_target_stock[sku]
                    excess = max(0.0, self.inventory[dc_id][sku] - target)
                    if excess > 0:
                        tier1 = min(excess, target)
                        tier2 = max(0.0, excess - target)
                        ordering += self.excess_penalty_dc * tier1
                        ordering += self.excess_penalty_dc * 3.0 * tier2
                total_cost += holding + backlog + ordering
            rewards[dc_id] = -total_cost
        for retailer_idx, retailer_id in enumerate(self.retailer_ids):
            total_cost = 0.0
            assigned_dc = self.retailer_to_dc[retailer_id]
            for sku in range(self.n_skus):
                holding = self.H_retailer[retailer_idx][sku] * self.inventory[retailer_id][sku] * self.holding_weight
                backlog = self.B_retailer[retailer_idx][sku] * self.backlog[retailer_id][sku] * self.backlog_weight
                action = actions[retailer_id]
                order_qty = float(action[sku])
                ordering = 0.0
                if order_qty > 0:
                    var_cost = self.C_var_retailer[retailer_idx][assigned_dc][sku]
                    ordering = (self.C_fixed_retailer[retailer_idx][sku] + (var_cost * order_qty)) * self.ordering_weight
                ss_threshold = self.safety_stock_threshold[retailer_idx][sku]
                if ss_threshold > 0:
                    shortfall = max(0.0, ss_threshold - self.inventory[retailer_id][sku])
                    ordering += self.safety_stock_penalty * shortfall
                if self.excess_penalty_retailer > 0.0 and self._retailer_target_stock is not None:
                    excess = max(0.0, self.inventory[retailer_id][sku] - self._retailer_target_stock[sku])
                    ordering += self.excess_penalty_retailer * excess
                total_cost += holding + backlog + ordering
            retailer_sale_revenue = self.sale_revenue_retailer * float(np.sum(self.demand_fulfilled[retailer_id]))
            rewards[retailer_id] = -total_cost + retailer_sale_revenue
        if self.shaping_enabled and self.shaping_weight > 0.0:
            for agent_id in range(self.n_agents):
                shaping_bonus = 0.0
                for sku in range(self.n_skus):
                    rec = self._get_heuristic_recommendation(agent_id, sku)
                    action_sku = float(actions[agent_id][sku])
                    phi_curr = self._compute_potential(rec, action_sku)
                    phi_prev = self.prev_potential[agent_id][sku]
                    shaping_bonus += phi_curr - phi_prev
                rewards[agent_id] += self.shaping_weight * shaping_bonus
        norm = float(self.max_days)
        rewards = {agent_id: r / norm for agent_id, r in rewards.items()}
        if self.reward_alpha > 0.0:
            global_mean = sum(rewards.values()) / len(rewards)
            rewards = {agent_id: (1 - self.reward_alpha) * r + self.reward_alpha * global_mean for agent_id, r in rewards.items()}
        return rewards

    def _get_heuristic_recommendation(self, agent_id: int, sku: int) -> float:
        z_base = 1.65
        if agent_id in self.dc_ids:
            assigned_retailers = self.dc_assignments[agent_id]
            n_assigned = max(len(assigned_retailers), 1)
            mu = float(self.demand_mean[sku]) * n_assigned
            sigma = float(self.demand_std[sku]) * n_assigned
            lead_time = float(self.lt_supplier_to_dc_max)
            on_hand = float(self.inventory[agent_id][sku])
            owed = sum(self.dc_retailer_backlog[agent_id][r_id][sku] for r_id in assigned_retailers)
            pipeline = sum(o['qty'] for o in self.pipeline[agent_id] if o['sku'] == sku)
            ip = on_hand - owed + pipeline
            out_level_dc = mu * lead_time + z_base * sigma * float(np.sqrt(max(lead_time, 1)))
            if ip >= out_level_dc:
                return 0.0
            if owed > 0:
                z_boost = min(0.5 * float(np.log1p(owed / max(mu, 1.0))), 1.5)
                z = min(z_base + z_boost, 3.0)
            else:
                z = z_base
        else:
            retailer_idx = self.retailer_ids.index(agent_id)
            if (retailer_idx < len(self.demand_history[sku]) and len(self.demand_history[sku][retailer_idx]) >= 2):
                hist = self.demand_history[sku][retailer_idx][-self.shaping_window:]
                mu = float(np.mean(hist))
                sigma = float(np.std(hist)) if len(hist) > 1 else float(self.demand_std[sku])
            else:
                mu = float(self.demand_mean[sku])
                sigma = float(self.demand_std[sku])
            lead_time = 7
            on_hand = float(self.inventory[agent_id][sku])
            backlog = float(self.backlog[agent_id][sku])
            pipeline = sum(o['qty'] for o in self.pipeline[agent_id] if o['sku'] == sku)
            ip = on_hand - backlog + pipeline
            if backlog > 0:
                z_boost = min(0.5 * float(np.log1p(backlog / max(mu, 1.0))), 1.5)
                z = min(z_base + z_boost, 3.0)
            else:
                z = z_base
        out_level = mu * lead_time + z * sigma * float(np.sqrt(max(lead_time, 1)))
        recommendation = max(0.0, out_level - ip)
        return float(recommendation)

    def _compute_potential(self, recommendation: float, action: float) -> float:
        return -self.shaping_k * abs(recommendation - action)

    def decay_shaping_weight(self, decay_rate: float = 0.995) -> float:
        self.shaping_weight = max(0.0, self.shaping_weight * decay_rate)
        return self.shaping_weight

    def _get_observations(self) -> Dict[int, np.ndarray]:
        observations = {}
        for dc_id in self.dc_ids:
            observations[dc_id] = self._get_dc_observation(dc_id)
        for retailer_id in self.retailer_ids:
            observations[retailer_id] = self._get_retailer_observation(retailer_id)
        return observations

    def _get_dc_observation(self, dc_id: int) -> np.ndarray:
        obs = []
        for sku in range(self.n_skus):
            obs.append(self.inventory[dc_id][sku] / 1000.0)
            total_owed_sku = sum(self.dc_retailer_backlog[dc_id][r_id][sku] for r_id in self.dc_assignments[dc_id])
            obs.append(total_owed_sku / 100.0)
            pipeline_7_9 = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku and self.current_day + 7 <= o['arrival_day'] <= self.current_day + 9)
            pipeline_10_12 = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku and self.current_day + 10 <= o['arrival_day'] <= self.current_day + 12)
            pipeline_13_14 = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku and self.current_day + 13 <= o['arrival_day'] <= self.current_day + 14)
            obs.append(pipeline_7_9 / 500.0)
            obs.append(pipeline_10_12 / 500.0)
            obs.append(pipeline_13_14 / 500.0)
            total_pipeline = sum(o['qty'] for o in self.pipeline[dc_id] if o['sku'] == sku)
            obs.append(total_pipeline / 7000.0)
            obs.append(self.market_prices[sku] / self.price_bounds['max'][sku])
            avg_retailer_backlog = np.mean([self.backlog[r][sku] for r in self.retailer_ids])
            obs.append(avg_retailer_backlog / 100.0)
            avg_retailer_inventory = np.mean([self.inventory[r][sku] for r in self.retailer_ids])
            obs.append(avg_retailer_inventory / 150.0)
        total_f = sum(sum(self.sl_history[r]['fulfilled']) for r in self.retailer_ids)
        total_d = sum(sum(self.sl_history[r]['demanded']) for r in self.retailer_ids)
        rolling_sl = (total_f / total_d) if total_d > 0 else 1.0
        obs.append(float(np.clip(rolling_sl, 0.0, 1.0)))
        return np.array(obs, dtype=np.float32)

    def _get_retailer_observation(self, retailer_id: int) -> np.ndarray:
        obs = []
        retailer_idx = self.retailer_ids.index(retailer_id)
        assigned_dc = self.retailer_to_dc[retailer_id]
        for sku in range(self.n_skus):
            obs.append(self.inventory[retailer_id][sku] / 150.0)
            obs.append(self.backlog[retailer_id][sku] / 100.0)
            obs.append(self.inventory[assigned_dc][sku] / 1000.0)
            obs.append(self.backlog[assigned_dc][sku] / 100.0)
            pipeline_day1 = sum(o['qty'] for o in self.pipeline[retailer_id] if o['sku'] == sku and o['arrival_day'] == self.current_day + 1)
            obs.append(pipeline_day1 / 70.0)
            total_pipeline = sum(o['qty'] for o in self.pipeline[retailer_id] if o['sku'] == sku)
            obs.append(total_pipeline / 70.0)
            demand_cap = float(self.demand_mean[sku] + 3.0 * self.demand_std[sku])
            if retailer_idx < len(self.demand_history[sku]) and len(self.demand_history[sku][retailer_idx]) > 0:
                recent_demand = np.mean(self.demand_history[sku][retailer_idx][-3:])
            else:
                recent_demand = 0
            obs.append(recent_demand / demand_cap)
        hist = self.sl_history[retailer_id]
        if len(hist['demanded']) > 0:
            total_d = sum(hist['demanded'])
            total_f = sum(hist['fulfilled'])
            own_sl = (total_f / total_d) if total_d > 0 else 1.0
        else:
            own_sl = 1.0
        obs.append(float(np.clip(own_sl, 0.0, 1.0)))
        return np.array(obs, dtype=np.float32)

Env = MultiDCInventoryEnv
