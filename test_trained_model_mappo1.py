#!/usr/bin/env python
"""
Test/Evaluation Script for Trained MAPPO Models (v1)
=====================================================

Thin wrapper around MAPPOModelEvaluator from test_trained_model_mappo.py.

Produces exactly the same outputs as test_trained_model.py (HAPPO baseline):
  - evaluation_metrics.json
  - results_mappo.csv  /  episode_metrics.csv
  - episode_rewards.png, cost_breakdown.png, cost_breakdown_retailer.png,
    service_levels.png, detailed_trajectory.png, normalized_trajectory.png,
    dc_inventory_fluctuation.png, retailer_inventory_fluctuation.png,
    reward_distribution.png
  - step_trajectory_ep1.xlsx

The class fixes the two broken attribute references present in the base
MAPPOModelEvaluator (_run_episode uses self.adj_tensor / self.single_agent_obs_dim
which only exist in GNN evaluators) and overrides _save_metrics_json so that
the metadata block is MAPPO-specific rather than GNN-HAPPO-specific.

Usage:
    python test_trained_model_mappo1.py \\
        --model_dir results/24Apr_MAPPO/run_seed_1/models \\
        --episode_length 90 \\
        --num_episodes 10 \\
        --experiment_name eval_mappo1
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Re-use everything from the base MAPPO evaluator
# ---------------------------------------------------------------------------
from test_trained_model_mappo import (
    MAPPOModelEvaluator,
    NumpyEncoder,
    parse_args,
)


class MAPPOModelEvaluatorV1(MAPPOModelEvaluator):
    """
    Corrected MAPPO evaluator that removes GNN-specific attributes and
    produces output identical to the HAPPO baseline (test_trained_model.py).

    Overrides:
      - _run_episode  : uses per-agent flat obs; no adj_tensor / single_agent_obs_dim
      - _save_metrics_json : MAPPO-specific metadata (no GNN fields)
      - _save_metrics_csv  : output file named results_mappo.csv
      - _print_summary     : header says "MAPPO" instead of "GNN-HAPPO"
    """

    # ------------------------------------------------------------------
    # Evaluation loop — pure MAPPO (no GNN, no adj_tensor)
    # ------------------------------------------------------------------

    def _run_episode(self, episode_num: int, save_trajectory: bool = False) -> dict:
        """Run one evaluation episode using a flat per-agent MAPPO actor."""
        obs, _ = self.env.reset()

        # RNN states: [1, n_agents, recurrent_N, hidden_size]
        rnn_states = np.zeros((1, self.n_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.n_agents, 1), dtype=np.float32)

        ep_data = {
            'total_reward': 0.0,
            'total_cost': 0.0,
            'agent_rewards': [0.0] * self.n_agents,
            'agent_costs': [0.0] * self.n_agents,
            'holding_costs': [0.0] * self.n_agents,
            'backlog_costs': [0.0] * self.n_agents,
            'ordering_costs': [0.0] * self.n_agents,
            'avg_inventory': [0.0] * self.n_agents,
            'avg_backlog': [0.0] * self.n_agents,
            'service_level': [0.0] * self.n_agents,
            '_orders_placed':     [0] * self.n_agents,
            '_orders_from_stock': [0] * self.n_agents,
            'dc_cycle_service_level': {},
            'final_inventory': None,
            'final_backlog': None,
            # Trajectory: step-level retail demand & orders per SKU (aggregated)
            'traj_demand': [np.zeros(self.n_skus) for _ in range(self.args.episode_length)],
            'traj_orders': [np.zeros(self.n_skus) for _ in range(self.args.episode_length)],
        }

        if save_trajectory:
            traj = {
                'inventory':       [[] for _ in range(self.n_agents)],
                'inventory_skus':  [[] for _ in range(self.n_agents)],
                'backlog':         [[] for _ in range(self.n_agents)],
                'actions':         [[] for _ in range(self.n_agents)],
                'rewards':         [[] for _ in range(self.n_agents)],
                'demand':          [[] for _ in range(self.n_agents)],
                'norm_demand':     [[] for _ in range(self.n_agents)],
                'norm_inventory':  [[] for _ in range(self.n_agents)],
                'norm_order':      [[] for _ in range(self.n_agents)],
                'orders_placed':   [[] for _ in range(self.n_agents)],
                'orders_from_stock': [[] for _ in range(self.n_agents)],
                'norm_scales': None,
            }

        for step in range(self.args.episode_length):
            # Capture market prices BEFORE step (matches env's _calculate_rewards)
            _pre_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = _pre_env_list[0].market_prices.copy() if _pre_env_list else None

            actions_env = []
            raw_actions = {}

            for agent_id in range(self.n_agents):
                # Build per-agent flat obs and pad if model was trained with larger dim
                obs_agent = np.stack(obs[:, agent_id])          # [1, obs_dim_i]
                policy_input_dim = self.obs_dims[agent_id]
                current_obs_dim  = obs_agent.shape[1]
                if current_obs_dim < policy_input_dim:
                    padding = np.zeros(
                        (obs_agent.shape[0], policy_input_dim - current_obs_dim),
                        dtype=np.float32,
                    )
                    obs_agent = np.concatenate([obs_agent, padding], axis=1)

                with torch.no_grad():
                    action, rnn_state = self.policies[agent_id].act(
                        obs_agent,
                        rnn_states[:, agent_id],    # [1, recurrent_N, hidden]
                        masks[:, agent_id],          # [1, 1]
                        deterministic=True,
                        agent_id=agent_id,
                    )

                rnn_states[:, agent_id] = (
                    rnn_state.cpu().numpy()
                    if isinstance(rnn_state, torch.Tensor)
                    else rnn_state
                )
                action_np = (
                    action.cpu().numpy()
                    if isinstance(action, torch.Tensor)
                    else action
                )
                raw_action = action_np[0]

                # ── DC IP-sufficiency guard (inference-time) ──────────────
                if agent_id < 2 and _pre_env_list:
                    _env = _pre_env_list[0]
                    _z    = 1.4
                    _lt   = 7
                    _n_ret = len(_env.dc_assignments[agent_id])
                    _zero_action = True
                    for _sku in range(self.n_skus):
                        _mu    = float(_env.demand_mean[_sku]) * _n_ret
                        _sigma = float(_env.demand_std[_sku])  * _n_ret
                        _out   = _mu * _lt + _z * _sigma * float(np.sqrt(_lt))
                        _on_hand  = float(_env.inventory[agent_id][_sku])
                        _owed     = sum(
                            _env.dc_retailer_backlog[agent_id][r_id][_sku]
                            for r_id in _env.dc_assignments[agent_id]
                        )
                        _pipeline = sum(
                            o['qty'] for o in _env.pipeline[agent_id] if o['sku'] == _sku
                        )
                        _ip = _on_hand - _owed + _pipeline
                        if _ip < _out:
                            _zero_action = False
                            break
                    if _zero_action:
                        raw_action = np.zeros_like(raw_action)
                # ─────────────────────────────────────────────────────────

                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            # Step environment
            obs, rewards, dones, infos = self.env.step([actions_env])

            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)

                for agent_id in range(self.n_agents):
                    reward = float(np.array(rewards[0][agent_id]).item())
                    cost = -reward
                    ep_data['agent_rewards'][agent_id] += reward
                    ep_data['agent_costs'][agent_id]   += cost
                    ep_data['total_reward'] += reward
                    ep_data['total_cost']   += cost

                    # ---- Cost breakdown ----
                    h_cost = b_cost = o_cost = 0.0
                    is_dc = agent_id < 2

                    if is_dc:
                        dc_idx = agent_id
                        for sku in range(self.n_skus):
                            h_cost += (env_state.inventory[agent_id][sku]
                                       * env_state.H_dc[dc_idx][sku])
                            dc_owed_sku = sum(
                                env_state.dc_retailer_backlog[agent_id][r_id][sku]
                                for r_id in env_state.dc_assignments[agent_id]
                            )
                            b_cost += dc_owed_sku * env_state.B_dc[dc_idx][sku]
                            if executed_actions[agent_id][sku] > 0:
                                price = (pre_step_prices[sku]
                                         if pre_step_prices is not None
                                         else env_state.market_prices[sku])
                                o_cost += (env_state.C_fixed_dc[dc_idx][sku]
                                           + price * executed_actions[agent_id][sku])
                    else:
                        r_idx = agent_id - 2
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(self.n_skus):
                            h_cost += (env_state.inventory[agent_id][sku]
                                       * env_state.H_retailer[r_idx][sku])
                            b_cost += (env_state.backlog[agent_id][sku]
                                       * env_state.B_retailer[r_idx][sku])
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                o_cost += (env_state.C_fixed_retailer[r_idx][sku]
                                           + env_state.C_var_retailer[r_idx][assigned_dc][sku]
                                           * order_qty)

                        # Retailer order-count fill rate
                        for sku in range(self.n_skus):
                            placed     = env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            from_stock = env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                            ep_data['_orders_placed'][agent_id]     += placed
                            ep_data['_orders_from_stock'][agent_id] += from_stock

                            # Trajectory aggregation (Retail Demand vs Orders)
                            actual_demand = 0.0
                            if (sku < len(env_state.demand_history) and
                                    r_idx < len(env_state.demand_history[sku]) and
                                    len(env_state.demand_history[sku][r_idx]) > 0):
                                actual_demand = float(env_state.demand_history[sku][r_idx][-1])

                            ep_data['traj_demand'][step][sku] += actual_demand
                            ep_data['traj_orders'][step][sku] += float(executed_actions[agent_id][sku])

                    ep_data['holding_costs'][agent_id]  += h_cost
                    ep_data['backlog_costs'][agent_id]  += b_cost
                    ep_data['ordering_costs'][agent_id] += o_cost

                    inv_vec = env_state.inventory[agent_id]
                    inv     = inv_vec.sum()
                    if is_dc:
                        bl = sum(
                            sum(env_state.dc_retailer_backlog[agent_id][r_id].values())
                            for r_id in env_state.dc_assignments[agent_id]
                        )
                    else:
                        bl = env_state.backlog[agent_id].sum()

                    ep_data['avg_inventory'][agent_id] += inv
                    ep_data['avg_backlog'][agent_id]   += bl

                    # ---- Trajectory logging ----
                    if save_trajectory:
                        traj['inventory'][agent_id].append(float(inv))
                        traj['inventory_skus'][agent_id].append(
                            np.array(inv_vec, dtype=float).copy()
                        )
                        traj['backlog'][agent_id].append(float(bl))
                        traj['rewards'][agent_id].append(reward)
                        traj['actions'][agent_id].append(
                            np.array(executed_actions[agent_id], dtype=float).copy()
                        )

                        if agent_id < 2:
                            demand_vec = np.zeros(self.n_skus, dtype=float)
                            for r_id in env_state.dc_assignments[agent_id]:
                                op = env_state.step_orders_placed.get(r_id, {})
                                for s in range(self.n_skus):
                                    demand_vec[s] += op.get(s, 0.0)
                        else:
                            demand_vec = np.array(
                                env_state.step_demand.get(
                                    agent_id, np.zeros(self.n_skus, dtype=float)
                                ),
                                dtype=float,
                            )
                        traj['demand'][agent_id].append(demand_vec.copy())

                        # Normalized values (retailers only)
                        if agent_id < 2:
                            traj['norm_demand'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                            traj['norm_inventory'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                            traj['norm_order'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                        else:
                            dm = getattr(env_state, 'demand_mean', np.ones(self.n_skus) * 1.5)
                            ds = getattr(env_state, 'demand_std',  np.ones(self.n_skus) * 1.0)
                            demand_cap = np.maximum(dm + 3.0 * ds, 1e-6)
                            norm_d   = (demand_vec / demand_cap).astype(float)
                            norm_inv = (np.array(inv_vec, dtype=float) / 150.0)
                            act_clip = np.array(executed_actions[agent_id], dtype=float)
                            norm_ord = np.clip(act_clip / 10.0, 0.0, 1.0)
                            traj['norm_demand'][agent_id].append(norm_d)
                            traj['norm_inventory'][agent_id].append(norm_inv)
                            traj['norm_order'][agent_id].append(norm_ord)

                        op_vec  = np.array([env_state.step_orders_placed.get(agent_id, {}).get(s, 0)
                                            for s in range(self.n_skus)], dtype=float)
                        ofs_vec = np.array([env_state.step_orders_from_stock.get(agent_id, {}).get(s, 0)
                                            for s in range(self.n_skus)], dtype=float)
                        traj['orders_placed'][agent_id].append(op_vec)
                        traj['orders_from_stock'][agent_id].append(ofs_vec)

                        if traj['norm_scales'] is None and hasattr(env_state, 'demand_mean'):
                            dm2 = np.array(env_state.demand_mean, dtype=float).flatten()
                            ds2 = np.array(env_state.demand_std,  dtype=float).flatten()
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
                                'inv_scale_retailer':    150.0,
                                'backlog_scale_retailer': 100.0,
                                'order_min_retailer':    0,
                                'order_max_retailer':    10.0,
                            }

                if step == self.args.episode_length - 1:
                    ep_data['final_inventory'] = [
                        float(env_state.inventory[i].sum()) for i in range(self.n_agents)
                    ]
                    ep_data['final_backlog'] = [
                        float(env_state.backlog[i].sum()) for i in range(self.n_agents)
                    ]

            # DC Cycle SL from infos
            info_dc_sl = infos[0][0].get('dc_cycle_service_level', {}) if infos and infos[0] else {}
            ep_data['dc_cycle_service_level'] = {
                int(dc_id): float(sl_val)
                for dc_id, sl_val in info_dc_sl.items()
            }

        # Normalise accumulators
        T = self.args.episode_length
        for agent_id in range(self.n_agents):
            ep_data['avg_inventory'][agent_id] /= T
            ep_data['avg_backlog'][agent_id]   /= T
            if agent_id >= 2:
                placed     = ep_data['_orders_placed'][agent_id]
                from_stock = ep_data['_orders_from_stock'][agent_id]
                ep_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                dc_sl_map = ep_data.get('dc_cycle_service_level', {})
                ep_data['service_level'][agent_id] = dc_sl_map.get(agent_id, 100.0)

        if save_trajectory:
            self.detailed_trajectory = traj

        return ep_data

    # ------------------------------------------------------------------
    # Override: MAPPO-specific JSON metadata (remove GNN fields)
    # ------------------------------------------------------------------

    def _save_metrics_json(self, stats):
        path = self.save_dir / 'evaluation_metrics.json'
        output = {
            'metadata': {
                'model_dir':      str(self.args.model_dir),
                'algorithm':      'MAPPO',
                'num_episodes':   self.args.num_episodes,
                'episode_length': self.args.episode_length,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'statistics':   stats,
            'episode_data': self.episode_metrics,
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f'[OK] Saved metrics JSON: {path.name}')

    # ------------------------------------------------------------------
    # Override: output CSV named results_mappo.csv (mirrors HAPPO script)
    # ------------------------------------------------------------------

    def _save_metrics_csv(self):
        import csv
        results_path = self.save_dir / 'results_mappo.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'
        for path in (results_path, compat_path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales',
                    'Avg_Inventory', 'Total_Holding_Cost', 'Total_Backlog_Cost',
                    'Total_Ordering_Cost',
                ])
                for ep_num, m in enumerate(self.episode_metrics):
                    fill_rate        = float(np.mean(m['service_level']))
                    total_placed     = sum(m['_orders_placed'][aid]
                                          for aid in range(2, self.n_agents))
                    total_from_stock = sum(m['_orders_from_stock'][aid]
                                          for aid in range(2, self.n_agents))
                    lost_sales     = total_placed - total_from_stock
                    avg_inventory  = float(np.mean(m['avg_inventory']))
                    total_holding  = float(np.sum(m['holding_costs']))
                    total_backlog  = float(np.sum(m['backlog_costs']))
                    total_ordering = float(np.sum(m['ordering_costs']))
                    true_total_cost = total_holding + total_backlog + total_ordering
                    writer.writerow([
                        ep_num + 1,
                        round(true_total_cost, 4),
                        round(fill_rate,        4),
                        round(lost_sales,        4),
                        round(avg_inventory,     4),
                        round(total_holding,     4),
                        round(total_backlog,     4),
                        round(total_ordering,    4),
                    ])
        print(f'[OK] Saved metrics CSV: {results_path.name}  (also {compat_path.name})')

    # ------------------------------------------------------------------
    # Override: print summary header says MAPPO
    # ------------------------------------------------------------------

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('MAPPO Evaluation Summary')
        print('=' * 70)
        print(f"Episodes      : {stats['num_episodes']}")
        print(f"Episode length: {stats['episode_length']} days")
        print(f"Avg reward    : {stats['total_reward']['mean']:>14.2f} "
              f"(+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost      : {stats['total_cost']['mean']:>14.2f} "
              f"(+/-{stats['total_cost']['std']:.2f})")
        print(f"Best reward   : {stats['total_reward']['max']:>14.2f}")
        print(f"Worst reward  : {stats['total_reward']['min']:>14.2f}")
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
    evaluator = MAPPOModelEvaluatorV1(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
