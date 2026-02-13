#!/usr/bin/env python
"""
Test/Evaluation Script for Trained Multi-Agent RL Models
=========================================================

This script evaluates a trained MADRL model on the multi-DC inventory environment.
It generates comprehensive metrics and visualizations to demonstrate that MADRL
can solve the inventory optimization problem.

Usage:
    python test_trained_model.py --model_dir results/experiment_name/run_seed_1/models \
                                  --num_episodes 50 \
                                  --save_dir evaluation_results
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import json
from itertools import chain

from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from algorithms.happo_policy import HAPPO_Policy


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def parse_test_args():
    """Parse command-line arguments for testing."""
    parser = argparse.ArgumentParser(description='Test Trained MADRL Model')
    
    # Model paths
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to saved model directory (e.g., results/full_training/run_seed_1/models)')
    parser.add_argument('--config_path', type=str, default='configs/multi_sku_config.yaml',
                       help='Path to environment config file')
    
    # Testing parameters
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of evaluation episodes to run')
    parser.add_argument('--episode_length', type=int, default=365,
                       help='Length of each episode (days)')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this evaluation run (default: timestamp)')
    
    # Environment settings (must match training config)
    parser.add_argument('--num_agents', type=int, default=5,
                       help='Number of agents (2 DCs + 3 Retailers)')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    return args


class ModelEvaluator:
    """Evaluates trained MADRL models and generates comprehensive metrics."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = args.experiment_name if args.experiment_name else f"eval_{timestamp}"
        self.save_dir = Path(args.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"MADRL Model Evaluation")
        print(f"{'='*70}")
        print(f"Model directory: {args.model_dir}")
        print(f"Num episodes: {args.num_episodes}")
        print(f"Episode length: {args.episode_length} days")
        print(f"Results will be saved to: {self.save_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Create environment
        self.env = self._create_env()
        
        # Load trained models
        self.policies = self._load_models()
        
        # Metrics storage
        self.episode_metrics = []
        self.detailed_trajectory = None
        
    def _create_env(self):
        """Create evaluation environment."""
        print("Creating evaluation environment...")
        
        # Get config
        parser = get_config()
        parser.set_defaults(
            env_name="MultiDC",
            scenario_name="inventory_2echelon",
            num_agents=self.args.num_agents,
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name="happo"
        )
        
        all_args = parser.parse_known_args([])[0]
        env = DummyVecEnvMultiDC(all_args)
        
        # Sync num_agents with environment (config might override args)
        if hasattr(env, 'num_agent'):
            self.args.num_agents = env.num_agent
        
        print(f"✓ Environment created: {env.num_envs} environment(s)")
        print(f"  - Agents: {self.args.num_agents} (2 DCs + {self.args.num_agents - 2} Retailers)")
        print(f"  - Observation spaces: DCs=27D, Retailers=42D")
        print(f"  - Action spaces: All agents=6D continuous\n")
        
        return env
    
    def _load_models(self):
        """Load trained model weights for all agents."""
        print("Loading trained models...")
        
        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        policies = []
        
        # Get environment config
        parser = get_config()
        parser.set_defaults(
            env_name="MultiDC",
            scenario_name="inventory_2echelon",
            num_agents=self.args.num_agents,
            use_centralized_V=True,
            algorithm_name="happo",
            hidden_size=128,
            layer_N=2,
            use_ReLU=True,
            use_orthogonal=True,
            gain=0.01,
            recurrent_N=2,
            use_naive_recurrent_policy=True
        )
        all_args = parser.parse_known_args([])[0]
        
        for agent_id in range(self.args.num_agents):
            # Get observation and action spaces
            obs_space = self.env.observation_space[agent_id]
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space = self.env.action_space[agent_id]
            
            # Create policy
            # Find the best model for this agent
            # Pattern: actor_agent{id}_reward_{reward}.pt or actor_agent{id}.pt
            agent_files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
            
            if not agent_files:
                 raise FileNotFoundError(f"No model found for agent {agent_id} in {model_dir}")
            
            best_file = None
            best_reward = -float('inf')
            
            # First try to find files with reward suffix
            suffixed_files = []
            for f in agent_files:
                 if f.name == f"actor_agent{agent_id}.pt":
                     continue
                 try:
                     # Parse reward from filename: actor_agent0_reward_-123.45.pt
                     parts = f.name.split('_reward_')
                     if len(parts) == 2:
                         reward_str = parts[1].replace('.pt', '')
                         reward = float(reward_str)
                         suffixed_files.append((reward, f))
                 except ValueError:
                     continue
            
            if suffixed_files:
                # Sort by reward (descending) and pick the best
                suffixed_files.sort(key=lambda x: x[0], reverse=True)
                best_reward, best_file = suffixed_files[0]
                print(f"  Agent {agent_id}: Found best model with reward {best_reward:.2f}")
            else:
                # Fallback to standard name
                simple_path = model_dir / f"actor_agent{agent_id}.pt"
                if simple_path.exists():
                    best_file = simple_path
                    print(f"  Agent {agent_id}: Using standard model file")
                else:
                     # If we have files but none matched expected patterns, just take first one
                     # This is a fallback ensuring we try something
                     best_file = agent_files[0]
                     print(f"  Agent {agent_id}: Using first available file (unknown naming pattern)")

            print(f"  Loading: {best_file.name}")
            
            # Load state dict first to check dimensions
            state_dict = torch.load(str(best_file), map_location=self.device)
            
            # Check input dimension from the first layer weights
            # Typical path: base.mlp.fc1.0.weight -> shape [hidden, input_dim]
            saved_input_dim = None
            if 'base.mlp.fc1.0.weight' in state_dict:
                saved_input_dim = state_dict['base.mlp.fc1.0.weight'].shape[1]
            elif 'base.cnn.cnn.0.weight' in state_dict:
                saved_input_dim = state_dict['base.cnn.cnn.0.weight'].shape[1]
                
            # Adjust observation space if mismatch detected
            # This handles cases where training used padding (e.g. GNN trained with max dim)
            current_obs_dim = obs_space.shape[0]
            if saved_input_dim is not None and saved_input_dim != current_obs_dim:
                print(f"  DIMENSION MISMATCH: Model expects {saved_input_dim}, Env provides {current_obs_dim}")
                print(f"  -> Adjusting policy input dimension to {saved_input_dim} (Likely due to padding in training)")
                
                from gymnasium import spaces
                # Create a dummy space with the correct dimension
                # Using lower bound -inf to avoid issues, as we only care about shape here
                obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(saved_input_dim,), dtype=np.float32)
            
            # Create policy with potentially adjusted obs_space
            policy = HAPPO_Policy(
                all_args,
                obs_space,
                share_obs_space,
                act_space,
                device=self.device
            )
            
            # Load weights
            policy.actor.load_state_dict(state_dict)
            policy.actor.eval()
            
            policies.append(policy)
            print(f"✓ Loaded agent {agent_id} successfully")
        
        print(f"\n✓ All {self.args.num_agents} agent models loaded successfully!\n")
        return policies
    
    def evaluate(self):
        """Run evaluation episodes and collect metrics."""
        print(f"{'='*70}")
        print(f"Starting Evaluation: {self.args.num_episodes} episodes")
        print(f"{'='*70}\n")
        
        for episode in range(self.args.num_episodes):
            metrics = self._run_episode(episode, save_trajectory=(episode == 0))
            self.episode_metrics.append(metrics)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([m['total_reward'] for m in self.episode_metrics[-10:]])
                avg_cost = np.mean([m['total_cost'] for m in self.episode_metrics[-10:]])
                print(f"Episode {episode+1}/{self.args.num_episodes} - "
                      f"Avg Reward (last 10): {avg_reward:.2f}, "
                      f"Avg Cost (last 10): {avg_cost:.2f}")
        
        print(f"\n{'='*70}")
        print(f"Evaluation Complete!")
        print(f"{'='*70}\n")
    
    def _run_episode(self, episode_num, save_trajectory=False):
        """Run a single evaluation episode."""
        obs, _ = self.env.reset()
        
        # Initialize RNN states
        rnn_states = np.zeros((1, self.args.num_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.args.num_agents, 1), dtype=np.float32)
        
        # Metrics for this episode
        episode_data = {
            'total_reward': 0,
            'total_cost': 0,
            'agent_rewards': [0] * self.args.num_agents,
            'agent_costs': [0] * self.args.num_agents,
            'holding_costs': [0] * self.args.num_agents,
            'backlog_costs': [0] * self.args.num_agents,
            'ordering_costs': [0] * self.args.num_agents,
            'final_inventory': None,
            'final_backlog': None,
            'avg_inventory': [0] * self.args.num_agents,
            'avg_backlog': [0] * self.args.num_agents,
            'service_level': [0] * self.args.num_agents,  # % of time with inventory > 0
        }
        
        # Trajectory data (only for first episode)
        if save_trajectory:
            trajectory = {
                'inventory': [[] for _ in range(self.args.num_agents)],
                'backlog': [[] for _ in range(self.args.num_agents)],
                'actions': [[] for _ in range(self.args.num_agents)],
                'rewards': [[] for _ in range(self.args.num_agents)],
            }
        
        # Run episode
        for step in range(self.args.episode_length):
            # Get actions from all agents
            actions_env = []
            
            for agent_id in range(self.args.num_agents):
                self.policies[agent_id].actor.eval()
                
                # Convert observation to float array (handle object array from env)
                # Convert observation to float array (handle object array from env)
                obs_agent = np.stack(obs[:, agent_id])
                
                # Check for observation padding requirements
                # The policy might have been initialized with a larger dimension (e.g. 36) than the env provides (e.g. 27)
                policy_input_dim = self.policies[agent_id].obs_space.shape[0]
                current_obs_dim = obs_agent.shape[1]
                
                if current_obs_dim < policy_input_dim:
                    # Pad with zeros
                    diff = policy_input_dim - current_obs_dim
                    # Create zeros of shape (batch_size, diff)
                    padding = np.zeros((obs_agent.shape[0], diff), dtype=np.float32)
                    obs_agent = np.concatenate([obs_agent, padding], axis=1)
                
                with torch.no_grad():
                    action, rnn_state = self.policies[agent_id].act(
                        obs_agent,
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True
                    )
                
                # Update RNN states using the returned rnn_state, NOT action
                rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                actions_env.append(action_np[0])
                
                if save_trajectory:
                    trajectory['actions'][agent_id].append(action_np[0].copy())
            
            # Step environment
            obs, rewards, dones, infos = self.env.step([actions_env])
            
            # Extract environment state for metrics
            # Support both 'env_list' (from our wrapper) and 'envs' (standard vec env)
            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            
            if env_list and len(env_list) > 0:
                env_state = env_list[0]
                
                # Calculate costs from rewards (rewards are negative costs)
                for agent_id in range(self.args.num_agents):
                    # Extract scalar reward (handle (1,) shape from env wrapper)
                    reward = float(rewards[0][agent_id])
                    cost = -reward
                    
                    episode_data['agent_rewards'][agent_id] += reward
                    episode_data['agent_costs'][agent_id] += cost
                    episode_data['total_reward'] += reward
                    episode_data['total_cost'] += cost
                    
                    # Track inventory and backlog
                    inv = env_state.inventory[agent_id].sum()
                    bl = env_state.backlog[agent_id].sum()
                    
                    episode_data['avg_inventory'][agent_id] += inv
                    episode_data['avg_backlog'][agent_id] += bl
                    
                    # Service level: count steps with positive inventory
                    if inv > 0:
                        episode_data['service_level'][agent_id] += 1
                    
                    if save_trajectory:
                        trajectory['inventory'][agent_id].append(inv)
                        trajectory['backlog'][agent_id].append(bl)
                        trajectory['rewards'][agent_id].append(reward)
                
                # Store final state
                if step == self.args.episode_length - 1:
                    episode_data['final_inventory'] = [
                        env_state.inventory[i].sum() for i in range(self.args.num_agents)
                    ]
                    episode_data['final_backlog'] = [
                        env_state.backlog[i].sum() for i in range(self.args.num_agents)
                    ]
        
        # Calculate averages and percentages
        for agent_id in range(self.args.num_agents):
            episode_data['avg_inventory'][agent_id] /= self.args.episode_length
            episode_data['avg_backlog'][agent_id] /= self.args.episode_length
            episode_data['service_level'][agent_id] = (
                episode_data['service_level'][agent_id] / self.args.episode_length * 100
            )
        
        # Save trajectory for first episode
        if save_trajectory:
            self.detailed_trajectory = trajectory
        
        return episode_data
    
    def generate_report(self):
        """Generate comprehensive evaluation report with metrics and visualizations."""
        print("Generating evaluation report...")
        
        # Calculate aggregate statistics
        stats = self._calculate_statistics()
        
        # Save metrics to JSON
        self._save_metrics_json(stats)
        
        # Save metrics to CSV
        self._save_metrics_csv()
        
        # Generate visualizations
        self._create_visualizations(stats)
        
        # Print summary
        self._print_summary(stats)
        
        print(f"\n✓ Evaluation report saved to: {self.save_dir}\n")
    
    def _calculate_statistics(self):
        """Calculate aggregate statistics from all episodes."""
        stats = {
            'num_episodes': len(self.episode_metrics),
            'episode_length': self.args.episode_length,
            'total_reward': {
                'mean': np.mean([m['total_reward'] for m in self.episode_metrics]),
                'std': np.std([m['total_reward'] for m in self.episode_metrics]),
                'min': np.min([m['total_reward'] for m in self.episode_metrics]),
                'max': np.max([m['total_reward'] for m in self.episode_metrics]),
            },
            'total_cost': {
                'mean': np.mean([m['total_cost'] for m in self.episode_metrics]),
                'std': np.std([m['total_cost'] for m in self.episode_metrics]),
                'min': np.min([m['total_cost'] for m in self.episode_metrics]),
                'max': np.max([m['total_cost'] for m in self.episode_metrics]),
            },
            'per_agent': {}
        }
        
        # Per-agent statistics
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            agent_name = f"{agent_type}_{agent_id}"
            
            stats['per_agent'][agent_name] = {
                'avg_reward': np.mean([m['agent_rewards'][agent_id] for m in self.episode_metrics]),
                'avg_cost': np.mean([m['agent_costs'][agent_id] for m in self.episode_metrics]),
                'avg_inventory': np.mean([m['avg_inventory'][agent_id] for m in self.episode_metrics]),
                'avg_backlog': np.mean([m['avg_backlog'][agent_id] for m in self.episode_metrics]),
                'service_level': np.mean([m['service_level'][agent_id] for m in self.episode_metrics]),
            }
        
        return stats
    
    def _save_metrics_json(self, stats):
        """Save metrics to JSON file."""
        json_path = self.save_dir / "evaluation_metrics.json"
        
        # Add metadata
        output = {
            'metadata': {
                'model_dir': str(self.args.model_dir),
                'num_episodes': self.args.num_episodes,
                'episode_length': self.args.episode_length,
                'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            'statistics': stats,
            'episode_data': self.episode_metrics
        }
        
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved metrics JSON: {json_path.name}")
    
    def _save_metrics_csv(self):
        """Save episode metrics to CSV file."""
        import csv
        
        csv_path = self.save_dir / "episode_metrics.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['episode', 'total_reward', 'total_cost']
            for agent_id in range(self.args.num_agents):
                agent_type = "DC" if agent_id < 2 else "Retailer"
                prefix = f"{agent_type}{agent_id}"
                header.extend([
                    f'{prefix}_reward', f'{prefix}_cost',
                    f'{prefix}_avg_inv', f'{prefix}_avg_backlog',
                    f'{prefix}_service_level'
                ])
            writer.writerow(header)
            
            # Data rows
            for ep_num, metrics in enumerate(self.episode_metrics):
                row = [ep_num + 1, metrics['total_reward'], metrics['total_cost']]
                for agent_id in range(self.args.num_agents):
                    row.extend([
                        metrics['agent_rewards'][agent_id],
                        metrics['agent_costs'][agent_id],
                        metrics['avg_inventory'][agent_id],
                        metrics['avg_backlog'][agent_id],
                        metrics['service_level'][agent_id]
                    ])
                writer.writerow(row)
        
        print(f"✓ Saved metrics CSV: {csv_path.name}")
    
    def _create_visualizations(self, stats):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # 1. Episode rewards over time
        self._plot_episode_rewards()
        
        # 2. Cost breakdown by agent
        self._plot_cost_breakdown(stats)
        
        # 3. Service level comparison
        self._plot_service_levels(stats)
        
        # 4. Inventory trajectory (first episode)
        if self.detailed_trajectory:
            self._plot_trajectory()
        
        # 5. Performance distribution
        self._plot_performance_distribution()
        
        print("✓ All visualizations created")
    
    def _plot_episode_rewards(self):
        """Plot total reward across episodes."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(self.episode_metrics) + 1)
        rewards = [m['total_reward'] for m in self.episode_metrics]
        
        ax.plot(episodes, rewards, linewidth=2, alpha=0.7, label='Episode Reward')
        
        # Rolling average
        window = min(10, len(rewards))
        if len(rewards) >= window:
            rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), rolling_avg,
                   linewidth=2.5, color='red', label=f'{window}-Episode Moving Average')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('MADRL Model Performance Across Episodes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'episode_rewards.png', dpi=300)
        plt.close()
    
    def _plot_cost_breakdown(self, stats):
        """Plot cost breakdown by agent."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agents = list(stats['per_agent'].keys())
        costs = [stats['per_agent'][a]['avg_cost'] for a in agents]
        
        colors = ['#2E86AB', '#2E86AB', '#A23B72', '#A23B72', '#A23B72']
        bars = ax.bar(agents, costs, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Average Total Cost per Episode', fontsize=12)
        ax.set_title('Cost Distribution Across Agents', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Distribution Centers'),
            Patch(facecolor='#A23B72', label='Retailers')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'cost_breakdown.png', dpi=300)
        plt.close()
    
    def _plot_service_levels(self, stats):
        """Plot service level comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agents = list(stats['per_agent'].keys())
        service_levels = [stats['per_agent'][a]['service_level'] for a in agents]
        
        colors = ['#2E86AB', '#2E86AB', '#A23B72', '#A23B72', '#A23B72']
        bars = ax.bar(agents, service_levels, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add target line (e.g., 95% service level)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
        
        ax.set_ylabel('Service Level (%)', fontsize=12)
        ax.set_title('Service Level: % of Time with Positive Inventory', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'service_levels.png', dpi=300)
        plt.close()
    
    def _plot_trajectory(self):
        """Plot detailed trajectory from first episode."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        days = range(1, self.args.episode_length + 1)
        
        # Plot 1: Inventory levels
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            label = f"{agent_type}_{agent_id}"
            axes[0].plot(days, self.detailed_trajectory['inventory'][agent_id],
                        label=label, linewidth=1.5, alpha=0.8)
        
        axes[0].set_ylabel('Total Inventory', fontsize=11)
        axes[0].set_title('Inventory Trajectory (First Episode)', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', ncol=2)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Backlog levels
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            label = f"{agent_type}_{agent_id}"
            axes[1].plot(days, self.detailed_trajectory['backlog'][agent_id],
                        label=label, linewidth=1.5, alpha=0.8)
        
        axes[1].set_ylabel('Total Backlog', fontsize=11)
        axes[1].set_title('Backlog Trajectory (First Episode)', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', ncol=2)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Rewards
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            label = f"{agent_type}_{agent_id}"
            axes[2].plot(days, self.detailed_trajectory['rewards'][agent_id],
                        label=label, linewidth=1.5, alpha=0.8)
        
        axes[2].set_xlabel('Day', fontsize=11)
        axes[2].set_ylabel('Reward (Negative Cost)', fontsize=11)
        axes[2].set_title('Reward Trajectory (First Episode)', fontsize=12, fontweight='bold')
        axes[2].legend(loc='lower right', ncol=2)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'detailed_trajectory.png', dpi=300)
        plt.close()
    
    def _plot_performance_distribution(self):
        """Plot distribution of episode performance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reward distribution
        rewards = [m['total_reward'] for m in self.episode_metrics]
        axes[0].hist(rewards, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(rewards):.1f}')
        axes[0].set_xlabel('Total Episode Reward', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Episode Rewards', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cost distribution
        costs = [m['total_cost'] for m in self.episode_metrics]
        axes[1].hist(costs, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(costs), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(costs):.1f}')
        axes[1].set_xlabel('Total Episode Cost', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Episode Costs', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_distribution.png', dpi=300)
        plt.close()
    
    def _print_summary(self, stats):
        """Print evaluation summary to console."""
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {self.args.model_dir}")
        print(f"Episodes evaluated: {stats['num_episodes']}")
        print(f"Episode length: {stats['episode_length']} days")
        print()
        
        print(f"OVERALL PERFORMANCE:")
        print(f"  Average Total Reward: {stats['total_reward']['mean']:.2f} ± {stats['total_reward']['std']:.2f}")
        print(f"  Average Total Cost:   {stats['total_cost']['mean']:.2f} ± {stats['total_cost']['std']:.2f}")
        print(f"  Best Episode Reward:  {stats['total_reward']['max']:.2f}")
        print(f"  Worst Episode Reward: {stats['total_reward']['min']:.2f}")
        print()
        
        print(f"PER-AGENT BREAKDOWN:")
        print(f"{'Agent':<15} {'Avg Cost':<12} {'Avg Inventory':<15} {'Avg Backlog':<15} {'Service Level':<15}")
        print(f"{'-'*70}")
        for agent_name, agent_stats in stats['per_agent'].items():
            print(f"{agent_name:<15} "
                  f"{agent_stats['avg_cost']:<12.2f} "
                  f"{agent_stats['avg_inventory']:<15.2f} "
                  f"{agent_stats['avg_backlog']:<15.2f} "
                  f"{agent_stats['service_level']:<15.1f}%")
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {self.save_dir}")
        print(f"{'='*70}\n")


def main():
    """Main evaluation function."""
    args = parse_test_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args)
    
    # Run evaluation
    evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report()
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
