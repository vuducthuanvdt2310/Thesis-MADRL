#!/usr/bin/env python
"""
Baseline Evaluation Script: Base-Stock Policy
==============================================

This script evaluates a classical Base-Stock (Order-Up-To) policy on the
multi-DC inventory environment defined in envs/multi_dc_env.py.

It is designed as a *baseline* for comparison against MADRL models
(e.g. GNN-HAPPO).  The output format, metrics, charts, and CSV/JSON files
mirror those produced by test_trained_model_gnn.py so the results can be
compared directly column-by-column.

Policy overview
---------------
At each time step every agent computes:

    order_qty = max(0,  S  -  IP)

where
    S  = base-stock level (configurable; one value per agent-type)
    IP = inventory position = on-hand + in-pipeline - backlog

For DCs a fixed base-stock level S_dc is used.
For retailers a fixed base-stock level S_retailer is used.
Both are per-SKU scalars applied uniformly across all agents of that type.

Usage (all arguments are optional):
    python Test_baseline_basestock.py
    python Test_baseline_basestock.py --num_episodes 5 --episode_length 365
    python Test_baseline_basestock.py \\
        --basestock_dc 300 --basestock_retailer 8 \\
        --num_episodes 3 --experiment_name "basestock_v1"
"""

import sys
import os
import argparse
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Environment import
# ---------------------------------------------------------------------------
# Add project root to sys.path so the import works from any cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from envs.multi_dc_env import MultiDCInventoryEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types (mirrors test_trained_model_gnn.py)."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate (s,S) Heuristic Policy on Multi-DC Inventory Environment'
    )
    # (s,S) policy levels — separate for DCs and Retailers
    # DC: reorder when IP ≤ s_dc, order up to S_dc (per SKU)
    parser.add_argument('--s_dc', type=float, default=300.0,
                        help='Reorder point for DCs (units per SKU; default 50)')
    parser.add_argument('--S_dc', type=float, default=500.0, 
                        help='Order-up-to level for DCs (units per SKU; default 300)')
    # Retailer: reorder when IP ≤ s_retailer, order up to S_retailer (per SKU)
    parser.add_argument('--s_retailer', type=float, default=3,
                        help='Reorder point for Retailers (units per SKU; default 3)')
    parser.add_argument('--S_retailer', type=float, default=12,
                        help='Order-up-to level for Retailers (units per SKU; default 12)')
    
    # Episode settings
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100 for validation)')
    parser.add_argument('--episode_length', type=int, default=90,
                        help='Length of each episode in days (default: 365)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # Environment config
    parser.add_argument('--config_path', type=str,
                        default='configs/multi_dc_config.yaml',
                        help='Path to environment config YAML')

    # Output
    parser.add_argument('--save_dir', type=str,
                        default='evaluation_results',
                        help='Root directory for saved results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this run (default: auto timestamp)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Base-Stock Policy
# ---------------------------------------------------------------------------

class SsPolicy:
    """
    (s, S) Inventory Policy — reorder only when Inventory Position drops at or
    below the reorder point `s`; then order up to the order-up-to level `S`.

    At each step, for each agent and SKU:
        IP = on-hand + in-pipeline - backlog (retailers) / owed-to-retailers (DCs)
        if IP <= s:
            order_qty = S - IP   (always positive since S > s)
        else:
            order_qty = 0

    Separate (s, S) parameters for DC agents and Retailer agents.

    Parameters
    ----------
    s_dc, S_dc : float
        Reorder point and order-up-to level for DCs (per SKU).
    s_retailer, S_retailer : float
        Reorder point and order-up-to level for Retailers (per SKU).
    n_dcs, n_agents, n_skus : int
    """

    def __init__(self, s_dc: float, S_dc: float,
                 s_retailer: float, S_retailer: float,
                 n_dcs: int, n_agents: int, n_skus: int):
        assert S_dc > s_dc,       f'S_dc ({S_dc}) must be > s_dc ({s_dc})'
        assert S_retailer > s_retailer, f'S_retailer ({S_retailer}) must be > s_retailer ({s_retailer})'
        self.s_dc = s_dc
        self.S_dc = S_dc
        self.s_retailer = s_retailer
        self.S_retailer = S_retailer
        self.n_dcs = n_dcs
        self.n_agents = n_agents
        self.n_skus = n_skus

    def get_actions(self, env: MultiDCInventoryEnv) -> dict:
        """
        Compute (s,S) actions for all agents given the current env state.

        Returns
        -------
        actions : dict {agent_id: np.ndarray of shape (n_skus,)}
        """
        actions = {}

        for agent_id in range(self.n_agents):
            order = np.zeros(self.n_skus, dtype=np.float32)

            for sku in range(self.n_skus):
                on_hand = float(env.inventory[agent_id][sku])

                # In-pipeline inventory for this agent / sku
                pipeline_qty = sum(
                    o['qty'] for o in env.pipeline[agent_id] if o['sku'] == sku
                )

                if agent_id < self.n_dcs:
                    # DC: IP = on-hand - owed_to_retailers + pipeline
                    owed = sum(
                        env.dc_retailer_backlog[agent_id][r_id][sku]
                        for r_id in env.dc_assignments[agent_id]
                    )
                    ip = on_hand - owed + pipeline_qty
                    s, S = self.s_dc, self.S_dc
                else:
                    # Retailer: IP = on-hand - backlog + pipeline
                    backlog = float(env.backlog[agent_id][sku])
                    ip = on_hand - backlog + pipeline_qty
                    s, S = self.s_retailer, self.S_retailer

                # (s,S) trigger: only order when IP reaches or drops below s
                if ip <= s:
                    order[sku] = max(0.0, S)
                # else: order[sku] stays 0

            actions[agent_id] = order

        return actions

# Keep old name as alias for backward compatibility
BaseStockPolicy = SsPolicy


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BaseStockEvaluator:
    """
    Evaluates the base-stock policy on the multi-DC inventory environment.
    Mirrors the structure and output of GNNModelEvaluator in
    test_trained_model_gnn.py.
    """

    def __init__(self, args):
        self.args = args

        # Output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = (args.experiment_name
                    if args.experiment_name
                    else f'basestock_{timestamp}')
        self.save_dir = Path(args.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._print_header()

        # Create environment
        self.env = self._create_env()

        # Convenience attributes (mirror GNNModelEvaluator)
        self.n_agents = self.env.n_agents
        self.n_dcs    = self.env.n_dcs
        self.n_skus   = self.env.n_skus

        # Instantiate the (s,S) policy
        self.policy = SsPolicy(
            s_dc=args.s_dc,
            S_dc=args.S_dc,
            s_retailer=args.s_retailer,
            S_retailer=args.S_retailer,
            n_dcs=self.n_dcs,
            n_agents=self.n_agents,
            n_skus=self.n_skus,
        )

        # Storage (same structure as test_trained_model_gnn.py)
        self.episode_metrics   = []
        self.detailed_trajectory = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _print_header(self):
        print('=' * 70)
        print('(s,S) Heuristic Policy Evaluation')
        print('=' * 70)
        print(f'Config           : {self.args.config_path}')
        print(f'Num episodes     : {self.args.num_episodes}')
        print(f'Episode length   : {self.args.episode_length} days')
        print(f'Seed             : {getattr(self.args, "seed", 42)}')
        print(f'DC  (s, S)       : ({self.args.s_dc}, {self.args.S_dc}) per SKU')
        print(f'Retailer (s, S)  : ({self.args.s_retailer}, {self.args.S_retailer}) per SKU')
        print(f'Results dir      : {str(Path(self.args.save_dir) / (self.args.experiment_name or "ss_heuristic_..."))}')
        print('=' * 70 + '\n')

    def _create_env(self) -> MultiDCInventoryEnv:
        print('Creating evaluation environment...')
        env = MultiDCInventoryEnv(config_path=self.args.config_path)

        # Override max_days so the environment matches --episode_length
        env.max_days = self.args.episode_length

        print(f'[OK] Environment created')
        print(f'     Agents : {env.n_agents} '
              f'({env.n_dcs} DCs + {env.n_retailers} retailers)')
        print(f'     SKUs   : {env.n_skus}\n')
        return env

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self):
        # ── Reproducibility ────────────────────────────────────────────────
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)
        # ───────────────────────────────────────────────────────────────────

        # ── Patch env clip so retailer actions are NOT capped at 10 ────────
        # The default env._clip_actions() enforces clip(0, 10) for retailers,
        # which would truncate the (s,S) order-up-to quantities.  We replace
        # it with a version that only ensures non-negativity for retailers.
        _original_clip = self.env._clip_actions

        def _ss_clip_actions(acts):
            clipped = {}
            for aid, act in acts.items():
                if aid in self.env.dc_ids:
                    clipped[aid] = np.clip(act, 0, 5000)   # DC: unchanged
                else:
                    clipped[aid] = np.maximum(0.0, np.array(act, dtype=np.float32))  # Retailer: only non-neg
            return clipped

        self.env._clip_actions = _ss_clip_actions
        # ────────────────────────────────────────────────────────────────────

        print('=' * 70)
        print(f'Starting (s,S) Heuristic Evaluation: {self.args.num_episodes} episode(s)  [seed={seed}]')
        print('=' * 70 + '\n')

        for ep in range(self.args.num_episodes):
            save_traj = (ep == 0)
            metrics = self._run_episode(ep, save_trajectory=save_traj)
            self.episode_metrics.append(metrics)

            avg_r = np.mean([m['total_reward'] for m in self.episode_metrics])
            print(f'Episode {ep + 1}/{self.args.num_episodes} '
                  f'| Total reward: {metrics["total_reward"]:>14.2f} '
                  f'| Running avg: {avg_r:>14.2f}')

        # Restore original clip so env remains unmodified after evaluation
        self.env._clip_actions = _original_clip

        print('\n[OK] Evaluation complete!\n')

    def _run_episode(self, episode_num: int, save_trajectory: bool = False) -> dict:
        """Run one full episode with the base-stock policy and collect metrics."""
        obs = self.env.reset()

        # Episode accumulators (same keys as test_trained_model_gnn.py)
        ep_data = {
            'total_reward':   0.0,
            'total_cost':     0.0,
            'agent_rewards':  [0.0] * self.n_agents,
            'agent_costs':    [0.0] * self.n_agents,
            'holding_costs':  [0.0] * self.n_agents,
            'backlog_costs':  [0.0] * self.n_agents,
            'ordering_costs': [0.0] * self.n_agents,
            'avg_inventory':  [0.0] * self.n_agents,
            'avg_backlog':    [0.0] * self.n_agents,
            'service_level':  [0.0] * self.n_agents,
            # Order-count fill-rate accumulators (retailers)
            '_orders_placed':     [0] * self.n_agents,
            '_orders_from_stock': [0] * self.n_agents,
            # DC Cycle SL
            'dc_cycle_service_level': {},
            'final_inventory': None,
            'final_backlog':   None,
        }

        # Optional trajectory storage (episode 1 only)
        if save_trajectory:
            traj = {
                'inventory':      [[] for _ in range(self.n_agents)],
                'inventory_skus': [[] for _ in range(self.n_agents)],
                'backlog':        [[] for _ in range(self.n_agents)],
                'actions':        [[] for _ in range(self.n_agents)],
                'rewards':        [[] for _ in range(self.n_agents)],
                'demand':         [[] for _ in range(self.n_agents)],
                # Normalized signals (retailers only; zeros for DCs)
                'norm_demand':    [[] for _ in range(self.n_agents)],
                'norm_inventory': [[] for _ in range(self.n_agents)],
                'norm_order':     [[] for _ in range(self.n_agents)],
                # Order-count fill rate per step
                'orders_placed':      [[] for _ in range(self.n_agents)],
                'orders_from_stock':  [[] for _ in range(self.n_agents)],
                'norm_scales': None,
            }

        # ── main time-step loop ──────────────────────────────────────────────
        for step in range(self.args.episode_length):

            # Snapshot pre-step market prices (for accurate ordering-cost log —
            # mirrors the pre-step price capture in test_trained_model_gnn.py)
            pre_step_prices = self.env.market_prices.copy()

            # Get base-stock actions (dict {agent_id: np.ndarray(n_skus)})
            actions = self.policy.get_actions(self.env)

            try:
                obs, rewards, dones, infos = self.env.step(actions)
            except Exception as exc:
                print(f'[WARNING] env.step() raised an exception at step {step}: {exc}')
                break

            # For (s,S) policy the actions are already non-negative by construction
            # (order_qty = max(0, S - IP) or 0).  We do NOT apply the env's hard
            # clip(0,10) so that retailers can order the full S-IP quantity.
            # We just ensure non-negativity here for consistent cost accounting.
            executed_actions = {
                aid: np.maximum(0.0, np.array(a, dtype=np.float32))
                for aid, a in actions.items()
            }

            # ── per-agent metric accumulation ─────────────────────────────────
            for agent_id in range(self.n_agents):
                reward = float(rewards[agent_id])
                cost   = -reward

                ep_data['agent_rewards'][agent_id] += reward
                ep_data['agent_costs'][agent_id]   += cost
                ep_data['total_reward']             += reward
                ep_data['total_cost']               += cost

                # ── Cost breakdown (holding / backlog / ordering) ──────────
                h_cost = b_cost = o_cost = 0.0
                is_dc = agent_id < self.n_dcs

                if is_dc:
                    dc_idx = agent_id
                    for sku in range(self.n_skus):
                        h_cost += (self.env.inventory[agent_id][sku]
                                   * self.env.H_dc[dc_idx][sku])
                        dc_owed_sku = sum(
                            self.env.dc_retailer_backlog[agent_id][r_id][sku]
                            for r_id in self.env.dc_assignments[agent_id]
                        )
                        b_cost += dc_owed_sku * self.env.B_dc[dc_idx][sku]
                        act_sku = float(executed_actions[agent_id][sku])
                        if act_sku > 0:
                            price = pre_step_prices[sku]
                            o_cost += (self.env.C_fixed_dc[dc_idx][sku]
                                       + price * act_sku)
                else:
                    r_idx      = agent_id - self.n_dcs
                    assigned_dc = self.env.retailer_to_dc[agent_id]
                    for sku in range(self.n_skus):
                        h_cost += (self.env.inventory[agent_id][sku]
                                   * self.env.H_retailer[r_idx][sku])
                        b_cost += (self.env.backlog[agent_id][sku]
                                   * self.env.B_retailer[r_idx][sku])
                        order_qty = float(executed_actions[agent_id][sku])
                        if order_qty > 0:
                            o_cost += (
                                self.env.C_fixed_retailer[r_idx][sku]
                                + self.env.C_var_retailer[r_idx][assigned_dc][sku] * order_qty
                            )

                    # Order-count fill rate (retailers only)
                    for sku in range(self.n_skus):
                        placed = self.env.step_orders_placed.get(agent_id, {}).get(sku, 0)
                        from_stock = self.env.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                        ep_data['_orders_placed'][agent_id]     += placed
                        ep_data['_orders_from_stock'][agent_id] += from_stock

                ep_data['holding_costs'][agent_id]  += h_cost
                ep_data['backlog_costs'][agent_id]   += b_cost
                ep_data['ordering_costs'][agent_id]  += o_cost

                # ── Inventory & backlog snapshots ─────────────────────────────
                inv_vec = self.env.inventory[agent_id]        # np.array (n_skus,)
                inv     = float(inv_vec.sum())

                if is_dc:
                    bl = sum(
                        sum(self.env.dc_retailer_backlog[agent_id][r_id].values())
                        for r_id in self.env.dc_assignments[agent_id]
                    )
                else:
                    bl = float(self.env.backlog[agent_id].sum())

                ep_data['avg_inventory'][agent_id] += inv
                ep_data['avg_backlog'][agent_id]   += bl

                # ── Trajectory recording (episode 1 only) ─────────────────────
                if save_trajectory:
                    traj['inventory'][agent_id].append(inv)
                    traj['inventory_skus'][agent_id].append(
                        np.array(inv_vec, dtype=float).copy()
                    )
                    traj['backlog'][agent_id].append(bl)
                    traj['rewards'][agent_id].append(reward)
                    traj['actions'][agent_id].append(
                        executed_actions[agent_id].copy()
                    )

                    # Demand vector for this agent this step
                    if is_dc:
                        demand_vec = np.zeros(self.n_skus, dtype=float)
                        for r_id in self.env.dc_assignments[agent_id]:
                            op = self.env.step_orders_placed.get(r_id, {})
                            for s in range(self.n_skus):
                                demand_vec[s] += op.get(s, 0.0)
                    else:
                        demand_vec = np.array(
                            self.env.step_demand.get(
                                agent_id, np.zeros(self.n_skus, dtype=float)
                            ),
                            dtype=float,
                        )
                    traj['demand'][agent_id].append(demand_vec.copy())

                    # Normalized signals (retailers only; zeros for DCs)
                    if is_dc:
                        traj['norm_demand'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                        traj['norm_inventory'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                        traj['norm_order'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                    else:
                        dm = getattr(self.env, 'demand_mean', np.ones(self.n_skus) * 1.5)
                        ds = getattr(self.env, 'demand_std',  np.ones(self.n_skus) * 1.0)
                        demand_cap = np.maximum(dm + 3.0 * ds, 1e-6)
                        norm_d   = (demand_vec / demand_cap).astype(float)
                        norm_inv = (np.array(inv_vec, dtype=float) / 150.0)
                        act      = executed_actions[agent_id]
                        norm_ord = np.clip((np.array(act, dtype=float) - 20.0) / 50.0, 0.0, 1.0)
                        traj['norm_demand'][agent_id].append(norm_d)
                        traj['norm_inventory'][agent_id].append(norm_inv)
                        traj['norm_order'][agent_id].append(norm_ord)

                    # Per-step order-count fill-rate vectors
                    op_vec  = [self.env.step_orders_placed.get(agent_id, {}).get(s, 0)
                               for s in range(self.n_skus)]
                    ofs_vec = [self.env.step_orders_from_stock.get(agent_id, {}).get(s, 0)
                               for s in range(self.n_skus)]
                    traj['orders_placed'][agent_id].append(np.array(op_vec, dtype=float))
                    traj['orders_from_stock'][agent_id].append(np.array(ofs_vec, dtype=float))

                    # Store normalisation scales once (same as GNN script)
                    if traj['norm_scales'] is None and hasattr(self.env, 'demand_mean'):
                        dm2 = np.array(self.env.demand_mean, dtype=float).flatten()
                        ds2 = np.array(self.env.demand_std,  dtype=float).flatten()
                        traj['norm_scales'] = {
                            'demand_mean_0': float(dm2[0]) if len(dm2) > 0 else 0,
                            'demand_mean_1': float(dm2[1]) if len(dm2) > 1 else 0,
                            'demand_mean_2': float(dm2[2]) if len(dm2) > 2 else 0,
                            'demand_std_0':  float(ds2[0]) if len(ds2) > 0 else 0,
                            'demand_std_1':  float(ds2[1]) if len(ds2) > 1 else 0,
                            'demand_std_2':  float(ds2[2]) if len(ds2) > 2 else 0,
                            'demand_cap_0':  float(dm2[0] + 3 * ds2[0]) if len(dm2) > 0 else 0,
                            'demand_cap_1':  float(dm2[1] + 3 * ds2[1]) if len(dm2) > 1 else 0,
                            'demand_cap_2':  float(dm2[2] + 3 * ds2[2]) if len(dm2) > 2 else 0,
                            'inv_scale_retailer':  150.0,
                            'backlog_scale_retailer': 100.0,
                            'order_min_retailer':  0,
                            'order_max_retailer':  50.0,
                        }

            # Final-step state snapshot
            if step == self.args.episode_length - 1:
                ep_data['final_inventory'] = [
                    float(self.env.inventory[i].sum()) for i in range(self.n_agents)
                ]
                ep_data['final_backlog'] = [
                    float(self.env.backlog[i].sum()) for i in range(self.n_agents)
                ]

            # Read DC Cycle SL from step infos
            info_dc_sl = infos.get(0, {}).get('dc_cycle_service_level', {}) if isinstance(infos, dict) else {}
            if not info_dc_sl and isinstance(infos, dict):
                # infos might be keyed by agent_id; try agent 0
                info_dc_sl = {}
                for v in infos.values():
                    if isinstance(v, dict) and 'dc_cycle_service_level' in v:
                        info_dc_sl = v['dc_cycle_service_level']
                        break
            ep_data['dc_cycle_service_level'] = {
                int(dc_id): float(sl_val)
                for dc_id, sl_val in info_dc_sl.items()
            }

        # ── Normalize time-averaged metrics ──────────────────────────────────
        T = self.args.episode_length
        for agent_id in range(self.n_agents):
            ep_data['avg_inventory'][agent_id] /= T
            ep_data['avg_backlog'][agent_id]   /= T

            if agent_id >= self.n_dcs:
                # Retailer: order-count fill rate
                placed     = ep_data['_orders_placed'][agent_id]
                from_stock = ep_data['_orders_from_stock'][agent_id]
                ep_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                # DC: cycle SL from env
                dc_sl_map = ep_data.get('dc_cycle_service_level', {})
                ep_data['service_level'][agent_id] = dc_sl_map.get(agent_id, 100.0)

        if save_trajectory:
            self.detailed_trajectory = traj

        return ep_data

    # ------------------------------------------------------------------
    # Report generation  (mirrors GNNModelEvaluator.generate_report)
    # ------------------------------------------------------------------

    def generate_report(self):
        print('Generating evaluation report...')
        stats = self._calculate_statistics()
        self._save_metrics_json(stats)
        self._save_metrics_csv()
        self._create_visualizations(stats)
        self._save_step_trajectory_excel()
        self._print_summary(stats)
        print(f'\n[OK] Report saved to: {self.save_dir}\n')

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _calculate_statistics(self) -> dict:
        def arr(key):
            return [m[key] for m in self.episode_metrics]

        stats = {
            'num_episodes':   len(self.episode_metrics),
            'episode_length': self.args.episode_length,
            'policy': '(s,S) Heuristic',
            's_dc':       self.args.s_dc,
            'S_dc':       self.args.S_dc,
            's_retailer': self.args.s_retailer,
            'S_retailer': self.args.S_retailer,
            'total_reward': {
                'mean': float(np.mean(arr('total_reward'))),
                'std':  float(np.std(arr('total_reward'))),
                'min':  float(np.min(arr('total_reward'))),
                'max':  float(np.max(arr('total_reward'))),
            },
            'total_cost': {
                'mean': float(np.mean(arr('total_cost'))),
                'std':  float(np.std(arr('total_cost'))),
                'min':  float(np.min(arr('total_cost'))),
                'max':  float(np.max(arr('total_cost'))),
            },
            'per_agent': {},
            'dc_cycle_service_level': {
                dc_id: float(np.mean([
                    m['dc_cycle_service_level'].get(dc_id, 100.0)
                    for m in self.episode_metrics
                ]))
                for dc_id in range(self.n_dcs)
            },
        }

        for aid in range(self.n_agents):
            label = f'{"DC" if aid < self.n_dcs else "Retailer"}_{aid}'

            if aid >= self.n_dcs:
                total_placed     = sum(m['_orders_placed'][aid]     for m in self.episode_metrics)
                total_from_stock = sum(m['_orders_from_stock'][aid] for m in self.episode_metrics)
                pooled_sl = (
                    (total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0
                )
            else:
                pooled_sl = float(np.mean([m['service_level'][aid] for m in self.episode_metrics]))

            stats['per_agent'][label] = {
                'avg_reward':        float(np.mean([m['agent_rewards'][aid] for m in self.episode_metrics])),
                'avg_cost':          float(np.mean([m['agent_costs'][aid]   for m in self.episode_metrics])),
                'avg_holding_cost':  float(np.mean([m['holding_costs'][aid] for m in self.episode_metrics])),
                'avg_backlog_cost':  float(np.mean([m['backlog_costs'][aid] for m in self.episode_metrics])),
                'avg_ordering_cost': float(np.mean([m['ordering_costs'][aid]for m in self.episode_metrics])),
                'avg_inventory':     float(np.mean([m['avg_inventory'][aid] for m in self.episode_metrics])),
                'avg_backlog':       float(np.mean([m['avg_backlog'][aid]   for m in self.episode_metrics])),
                'service_level':     pooled_sl,
            }

        # System-wide retailer fill rate (pooled)
        total_placed_all     = sum(
            m['_orders_placed'][aid]    for m in self.episode_metrics
            for aid in range(self.n_dcs, self.n_agents)
        )
        total_from_stock_all = sum(
            m['_orders_from_stock'][aid] for m in self.episode_metrics
            for aid in range(self.n_dcs, self.n_agents)
        )
        stats['system_retailer_fill_rate'] = (
            (total_from_stock_all / total_placed_all * 100.0)
            if total_placed_all > 0 else 100.0
        )

        return stats

    # ------------------------------------------------------------------
    # JSON / CSV output
    # ------------------------------------------------------------------

    def _save_metrics_json(self, stats):
        path = self.save_dir / 'evaluation_metrics.json'
        output = {
            'metadata': {
                'algorithm':      '(s,S) Heuristic (Baseline)',
                'config_path':    str(self.args.config_path),
                'num_episodes':   self.args.num_episodes,
                'episode_length': self.args.episode_length,
                's_dc':           self.args.s_dc,
                'S_dc':           self.args.S_dc,
                's_retailer':     self.args.s_retailer,
                'S_retailer':     self.args.S_retailer,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'statistics':   stats,
            'episode_data': self.episode_metrics,
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f'[OK] Saved metrics JSON : {path.name}')

    def _save_metrics_csv(self):
        # Primary output: standardised validation file
        results_path = self.save_dir / 'results_ss_heuristic.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'
        for path in (results_path, compat_path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                                 'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'])
                for ep_num, m in enumerate(self.episode_metrics):
                    # Fill_Rate: average of per-agent service_level (original method)
                    fill_rate = float(np.mean(m['service_level']))
                    # Lost_Sales: total unfulfilled demand units pooled across all retailer agents
                    total_placed     = sum(m['_orders_placed'][aid]     for aid in range(self.n_dcs, self.n_agents))
                    total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(self.n_dcs, self.n_agents))
                    lost_sales = total_placed - total_from_stock
                    avg_inventory = float(np.mean(m['avg_inventory']))
                    total_holding = float(np.sum(m['holding_costs']))
                    total_backlog = float(np.sum(m['backlog_costs']))
                    total_ordering = float(np.sum(m['ordering_costs']))
                    true_total_cost = total_holding + total_backlog + total_ordering
                    writer.writerow([
                        ep_num + 1,
                        round(true_total_cost, 4),
                        round(fill_rate, 4),
                        round(lost_sales, 4),
                        round(avg_inventory, 4),
                        round(total_holding, 4),
                        round(total_backlog, 4),
                        round(total_ordering, 4),
                    ])
        print(f'[OK] Saved metrics CSV  : {results_path.name}  (also {compat_path.name})')

    # ------------------------------------------------------------------
    # Step-trajectory Excel (mirrors _save_step_trajectory_excel)
    # ------------------------------------------------------------------

    def _save_step_trajectory_excel(self):
        """Save step-level trajectory to Excel (Data + Scales sheets)."""
        if not self.detailed_trajectory:
            return

        try:
            import pandas as pd
        except ImportError:
            print('[WARN] pandas not available — skipping Excel export.')
            return

        traj = self.detailed_trajectory
        num_steps = len(traj['inventory'][0])
        cum_orders_placed     = {aid: 0 for aid in range(self.n_agents)}
        cum_orders_from_stock = {aid: 0 for aid in range(self.n_agents)}

        rows = []
        for step in range(num_steps):
            for aid in range(self.n_agents):
                inv        = traj['inventory'][aid][step]
                bl         = traj['backlog'][aid][step]
                action_vec = np.array(traj['actions'][aid][step], dtype=float).flatten()
                demand_vec = np.array(traj['demand'][aid][step],  dtype=float).flatten()
                norm_d     = np.array(traj['norm_demand'][aid][step],    dtype=float).flatten()
                norm_inv   = np.array(traj['norm_inventory'][aid][step], dtype=float).flatten()
                norm_ord   = np.array(traj['norm_order'][aid][step],     dtype=float).flatten()
                op_vec     = np.array(traj['orders_placed'][aid][step],      dtype=float).flatten()
                ofs_vec    = np.array(traj['orders_from_stock'][aid][step],   dtype=float).flatten()

                step_placed     = int(np.sum(op_vec))
                step_from_stock = int(np.sum(ofs_vec))
                cum_orders_placed[aid]     += step_placed
                cum_orders_from_stock[aid] += step_from_stock
                cum_placed     = cum_orders_placed[aid]
                cum_from_stock = cum_orders_from_stock[aid]

                row = {
                    'step':     step + 1,
                    'agent_id': aid,
                    'agent':    f'{"DC" if aid < self.n_dcs else "R"}_{aid}',
                    'inv':      inv,
                    'backlog':  bl,
                    'reward':   traj['rewards'][aid][step],
                }
                for i in range(self.n_skus):
                    row[f'demand_{i}']  = demand_vec[i] if i < len(demand_vec) else 0
                    row[f'order_{i}']   = action_vec[i] if i < len(action_vec) else 0
                    row[f'norm_d_{i}']  = round(norm_d[i],   4) if i < len(norm_d)   else 0
                    row[f'norm_inv_{i}']= round(norm_inv[i], 4) if i < len(norm_inv) else 0
                    row[f'norm_ord_{i}']= round(norm_ord[i], 4) if i < len(norm_ord) else 0
                row['orders_placed']      = step_placed
                row['orders_from_stock']  = step_from_stock
                row['step_sl_pct']        = round(step_from_stock / step_placed * 100, 1) if step_placed > 0 else 100
                row['cum_placed']         = cum_placed
                row['cum_from_stock']     = cum_from_stock
                row['cum_sl_pct']         = round(cum_from_stock / cum_placed * 100, 1) if cum_placed > 0 else 100
                rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.save_dir / 'step_trajectory_ep1.xlsx'
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            scales = traj.get('norm_scales')
            if scales:
                df_scales = pd.DataFrame([{
                    'demand_cap_0': scales.get('demand_cap_0'),
                    'demand_cap_1': scales.get('demand_cap_1'),
                    'demand_cap_2': scales.get('demand_cap_2'),
                    'inv_scale':    scales.get('inv_scale_retailer'),
                    'backlog_scale':scales.get('backlog_scale_retailer'),
                    'order_min':    scales.get('order_min_retailer'),
                    'order_max':    scales.get('order_max_retailer'),
                }])
                df_scales.to_excel(writer, sheet_name='Scales', index=False)
            else:
                pd.DataFrame([{'note': 'Scales not captured'}]).to_excel(
                    writer, sheet_name='Scales', index=False
                )
        print(f'[OK] Saved step_trajectory_ep1.xlsx (Data + Scales)')

    # ------------------------------------------------------------------
    # Visualizations  (mirrors test_trained_model_gnn.py charts)
    # ------------------------------------------------------------------

    def _create_visualizations(self, stats):
        print('Creating visualizations...')
        self._plot_episode_rewards()
        self._plot_cost_breakdown(stats)
        self._plot_service_levels(stats)
        if self.detailed_trajectory:
            self._plot_trajectory()
            self._plot_normalized_trajectory()
            self._plot_dc_inventory_fluctuation()
            self._plot_retailer_inventory_fluctuation()
        self._plot_performance_distribution()
        print('[OK] All visualizations created')

    def _plot_episode_rewards(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        episodes = range(1, len(self.episode_metrics) + 1)
        rewards  = [m['total_reward'] for m in self.episode_metrics]
        ax.plot(episodes, rewards, linewidth=2, alpha=0.7, label='Episode Reward')
        window = min(10, len(rewards))
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), roll,
                    linewidth=2.5, color='red',
                    label=f'{window}-Ep Moving Avg')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Base-Stock Policy Performance Across Episodes',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'episode_rewards.png', dpi=300)
        plt.close()

    def _plot_cost_breakdown(self, stats):
        fig, ax = plt.subplots(figsize=(14, 7))
        agents = list(stats['per_agent'].keys())
        hc = [stats['per_agent'][a]['avg_holding_cost']  for a in agents]
        bc = [stats['per_agent'][a]['avg_backlog_cost']  for a in agents]
        oc = [stats['per_agent'][a]['avg_ordering_cost'] for a in agents]
        x = np.arange(len(agents))
        w = 0.6
        ax.bar(x, hc, w, label='Holding',  color='#4ECDC4', edgecolor='black')
        ax.bar(x, bc, w, bottom=hc,        label='Backlog',  color='#FF6B6B', edgecolor='black')
        ax.bar(x, oc, w, bottom=np.array(hc) + np.array(bc),
               label='Ordering', color='#FFD700', edgecolor='black')
        for i in range(len(agents)):
            total = hc[i] + bc[i] + oc[i]
            ax.text(i, total, f'{total:.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_ylabel('Avg Cost / Episode', fontsize=12)
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_title('Base-Stock Cost Breakdown by Agent',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'cost_breakdown.png', dpi=300)
        plt.close()

    def _plot_service_levels(self, stats):
        fig, ax = plt.subplots(figsize=(12, 5))
        agents = list(stats['per_agent'].keys())
        svc    = [stats['per_agent'][a]['service_level'] for a in agents]
        colors = ['#2E86AB'] * self.n_dcs + ['#A23B72'] * (len(agents) - self.n_dcs)
        bars   = ax.bar(agents, svc, color=colors, alpha=0.85, edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target 95%')
        ax.set_ylabel('Service Level (%)', fontsize=12)
        ax.set_title('Service Level by Agent — Base-Stock Baseline '
                     '(Fill-Rate for Retailers, Cycle SL for DCs)',
                     fontsize=13, fontweight='bold')
        ax.set_ylim([0, 110])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'service_levels.png', dpi=300)
        plt.close()

    def _plot_trajectory(self):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        days = range(1, self.args.episode_length + 1)
        for aid in range(self.n_agents):
            label = f'{"DC" if aid < self.n_dcs else "R"}_{aid}'
            axes[0].plot(days, self.detailed_trajectory['inventory'][aid],
                         label=label, linewidth=1.2, alpha=0.75)
            axes[1].plot(days, self.detailed_trajectory['backlog'][aid],
                         label=label, linewidth=1.2, alpha=0.75)
            axes[2].plot(days, self.detailed_trajectory['rewards'][aid],
                         label=label, linewidth=1.2, alpha=0.75)
        for ax, title, ylabel in zip(
            axes,
            ['Inventory Trajectory (Ep 1)', 'Backlog Trajectory (Ep 1)', 'Reward Trajectory (Ep 1)'],
            ['Total Inventory', 'Total Backlog', 'Reward']
        ):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(loc='upper right', ncol=3, fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[2].set_xlabel('Day', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'detailed_trajectory.png', dpi=300)
        plt.close()

    def _plot_normalized_trajectory(self):
        traj = self.detailed_trajectory
        if not traj or 'norm_demand' not in traj:
            return
        days        = range(1, self.args.episode_length + 1)
        retailer_ids = [i for i in range(self.n_agents) if i >= self.n_dcs]
        n_show       = min(5, len(retailer_ids))
        fig, axes    = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for aid in retailer_ids[:n_show]:
            label   = f'R_{aid}'
            norm_d   = np.array(traj['norm_demand'][aid],    dtype=float)
            norm_inv = np.array(traj['norm_inventory'][aid], dtype=float)
            norm_ord = np.array(traj['norm_order'][aid],     dtype=float)
            axes[0].plot(days, norm_d.mean(axis=1),   label=label, linewidth=1.2, alpha=0.85)
            axes[1].plot(days, norm_inv.mean(axis=1), label=label, linewidth=1.2, alpha=0.85)
            axes[2].plot(days, norm_ord.mean(axis=1), label=label, linewidth=1.2, alpha=0.85)
        for ax, title, ylabel in zip(
            axes,
            ['Normalized Demand (Ep 1, mean over SKUs)',
             'Normalized Inventory (Ep 1, mean over SKUs)',
             'Normalized Order Qty (Ep 1, mean over SKUs)'],
            ['Demand / (mean+3*std)', 'Inventory / 150', 'Order: (qty-20)/50']
        ):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.15)
        axes[2].set_xlabel('Day', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'normalized_trajectory.png', dpi=300)
        plt.close()
        print('[OK] Saved normalized trajectory plot: normalized_trajectory.png')

    def _plot_dc_inventory_fluctuation(self):
        traj   = self.detailed_trajectory
        days   = np.arange(1, self.args.episode_length + 1)
        dc_ids = list(range(self.n_dcs))
        n_cols = self.n_skus + 1

        fig, axes = plt.subplots(
            self.n_dcs, n_cols,
            figsize=(5 * n_cols, 4 * self.n_dcs),
            squeeze=False,
        )
        fig.suptitle('DC Inventory Fluctuation Over Steps (Episode 1) — Base-Stock',
                     fontsize=15, fontweight='bold', y=1.01)

        sku_colors  = ['#2196F3', '#4CAF50', '#FF9800']
        total_color = '#9C27B0'

        for row, dc_id in enumerate(dc_ids):
            dc_label  = f'DC_{dc_id}'
            inv_skus  = np.array(traj['inventory_skus'][dc_id], dtype=float)
            inv_total = np.array(traj['inventory'][dc_id],      dtype=float)

            for sku in range(self.n_skus):
                ax = axes[row][sku]
                ax.plot(days, inv_skus[:, sku],
                        color=sku_colors[sku % len(sku_colors)],
                        linewidth=1.5, alpha=0.85)
                ax.fill_between(days, inv_skus[:, sku], alpha=0.15,
                                color=sku_colors[sku % len(sku_colors)])
                ax.set_title(f'{dc_label} — SKU {sku}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Inventory', fontsize=10)
                ax.set_xlabel('Step (day)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(days[0], days[-1])

            ax_total = axes[row][n_cols - 1]
            ax_total.plot(days, inv_total, color=total_color, linewidth=2.0,
                          alpha=0.9, label='Total')
            ax_total.fill_between(days, inv_total, alpha=0.12, color=total_color)
            ax_total.axhline(np.mean(inv_total), color='red', linestyle='--',
                             linewidth=1.2, label=f'Mean: {np.mean(inv_total):.0f}')
            ax_total.set_title(f'{dc_label} — Total Inventory', fontsize=11, fontweight='bold')
            ax_total.set_ylabel('Total Inventory', fontsize=10)
            ax_total.set_xlabel('Step (day)', fontsize=10)
            ax_total.legend(fontsize=9)
            ax_total.grid(True, alpha=0.3)
            ax_total.set_xlim(days[0], days[-1])

        plt.tight_layout()
        out = self.save_dir / 'dc_inventory_fluctuation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[OK] Saved DC inventory fluctuation plot: {out.name}')

    def _plot_retailer_inventory_fluctuation(self):
        traj         = self.detailed_trajectory
        days         = np.arange(1, self.args.episode_length + 1)
        retailer_ids = [i for i in range(self.n_agents) if i >= self.n_dcs]
        n_retailers  = len(retailer_ids)

        inv_matrix = np.array(
            [traj['inventory'][rid] for rid in retailer_ids], dtype=float
        )

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.suptitle('Retailer Inventory Fluctuation Over Steps (Episode 1) — Base-Stock',
                     fontsize=15, fontweight='bold')

        cmap     = plt.cm.get_cmap('tab20', n_retailers)
        ax_lines = axes[0]
        for i, rid in enumerate(retailer_ids):
            ax_lines.plot(days, inv_matrix[i],
                          color=cmap(i), linewidth=1.2, alpha=0.75,
                          label=f'R_{rid}')
        ax_lines.set_title('Individual Retailer Inventory', fontsize=12, fontweight='bold')
        ax_lines.set_ylabel('Total Inventory', fontsize=11)
        ax_lines.legend(loc='upper right', ncol=max(1, n_retailers // 5),
                        fontsize=7, framealpha=0.7)
        ax_lines.grid(True, alpha=0.3)

        ax_avg  = axes[1]
        mean_inv = inv_matrix.mean(axis=0)
        std_inv  = inv_matrix.std(axis=0)
        ax_avg.plot(days, mean_inv, color='#1565C0', linewidth=2.0,
                    label='Mean (all retailers)')
        ax_avg.fill_between(
            days,
            np.maximum(mean_inv - std_inv, 0),
            mean_inv + std_inv,
            alpha=0.2, color='#1565C0', label='±1 std'
        )
        ax_avg.axhline(mean_inv.mean(), color='red', linestyle='--', linewidth=1.2,
                       label=f'Time-avg: {mean_inv.mean():.0f}')
        ax_avg.set_title('System-Wide Avg Retailer Inventory', fontsize=12, fontweight='bold')
        ax_avg.set_ylabel('Avg Total Inventory', fontsize=11)
        ax_avg.set_xlabel('Step (day)', fontsize=11)
        ax_avg.legend(fontsize=9)
        ax_avg.grid(True, alpha=0.3)
        ax_avg.set_xlim(days[0], days[-1])

        plt.tight_layout()
        out = self.save_dir / 'retailer_inventory_fluctuation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[OK] Saved retailer inventory fluctuation plot: {out.name}')

    def _plot_performance_distribution(self):
        if len(self.episode_metrics) < 2:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        rewards = [m['total_reward'] for m in self.episode_metrics]
        ax.hist(rewards, bins=min(20, len(rewards)), edgecolor='black',
                color='#3498db', alpha=0.8)
        ax.axvline(np.mean(rewards), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(rewards):.0f}')
        ax.set_xlabel('Total Episode Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Base-Stock Episode Reward Distribution',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'reward_distribution.png', dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Console summary  (mirrors _print_summary in test_trained_model_gnn.py)
    # ------------------------------------------------------------------

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('(s,S) Heuristic Policy Evaluation Summary')
        print('=' * 70)
        print(f"Episodes        : {stats['num_episodes']}")
        print(f"Episode length  : {stats['episode_length']} days")
        print(f"DC  (s, S)      : ({self.args.s_dc}, {self.args.S_dc}) per SKU")
        print(f"Ret (s, S)      : ({self.args.s_retailer}, {self.args.S_retailer}) per SKU")
        print(f"Avg reward      : {stats['total_reward']['mean']:>14.2f} "
              f"(+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost        : {stats['total_cost']['mean']:>14.2f} "
              f"(+/-{stats['total_cost']['std']:.2f})")
        print(f"Best reward     : {stats['total_reward']['max']:>14.2f}")
        print(f"Worst reward    : {stats['total_reward']['min']:>14.2f}")
        sys_fr = stats.get('system_retailer_fill_rate', 0.0)
        print(f"Retailer Fill Rate (system): {sys_fr:>6.1f}%   <- KPI")
        dc_csl = stats.get('dc_cycle_service_level', {})
        for dc_id, csl_val in dc_csl.items():
            print(f"DC_{dc_id} Cycle SL : {csl_val:>6.1f}%")
        print('-' * 70)
        print(f"{'Agent':<14} {'Avg Reward':>12} {'Avg Cost':>12} "
              f"{'Holding':>12} {'Backlog':>12} {'Svc%':>8} {'SL Type':>12}")
        print('-' * 70)
        for agent, data in stats['per_agent'].items():
            is_dc   = agent.startswith('DC')
            sl_type = 'Cycle SL' if is_dc else 'Fill-Rate'
            print(f"{agent:<14} {data['avg_reward']:>12.1f} {data['avg_cost']:>12.1f} "
                  f"{data['avg_holding_cost']:>12.1f} {data['avg_backlog_cost']:>12.1f} "
                  f"{data['service_level']:>7.1f}% {sl_type:>12}")
        print('=' * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Print a quick recap of the (s,S) configuration
    print('\n=== (s,S) Heuristic Configuration ===')
    print(f'  DC       (s, S) per SKU : ({args.s_dc}, {args.S_dc})')
    print(f'  Retailer (s, S) per SKU : ({args.s_retailer}, {args.S_retailer})')
    print('======================================\n')

    evaluator = BaseStockEvaluator(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
