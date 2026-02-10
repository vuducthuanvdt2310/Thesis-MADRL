"""
GNN-HAPPO Trainer.

Modified HAPPO trainer that handles graph data (adjacency matrix).
Core HAPPO algorithm remains the same - this just adapts data flow.
"""

import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check


class GNN_HAPPO():
    """
    Trainer class for GNN-HAPPO.
    
    Key differences from standard HAPPO:
    1. Passes adjacency matrix to policy forward/evaluate calls
    2. Buffer now includes adjacency matrix
    3. Otherwise identical update logic (PPO clipping, value loss, etc.)
    """
    
    def __init__(self, args, policy, n_agents, device=torch.device("cpu")):
        """
        Args:
            args: Configuration arguments
            policy: GNN_HAPPO_Policy instance
            n_agents: Number of agents
            device: torch device
        """
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.n_agents = n_agents
        
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
    
    def _reconstruct_structured_obs(self, share_obs_batch):
        """
        Reconstruct structured observations [batch, n_agents, max_obs_dim] from concatenated share_obs_batch.
        
        Args:
            share_obs_batch: Concatenated observations [batch, total_obs_dim]
                where total_obs_dim = 2*27 + 15*36 = 594
        
        Returns:
            obs_structured: [batch, n_agents, max_obs_dim] where max_obs_dim=36
        """
        batch_size = share_obs_batch.shape[0]
        n_agents = self.n_agents
        max_obs_dim = 36
        
        # Convert to numpy for slicing
        if isinstance(share_obs_batch, torch.Tensor):
            share_obs_np = share_obs_batch.detach().cpu().numpy()
        else:
            share_obs_np = share_obs_batch
        
        obs_structured = np.zeros((batch_size, n_agents, max_obs_dim), dtype=np.float32)
        
        # Extract observations for each agent from concatenated array
        # First 2 agents (DCs): 27D each
        # Next 15 agents (Retailers): 36D each
        offset = 0
        for agent_id in range(n_agents):
            if agent_id < 2:  # DCs
                obs_dim = 27
            else:  # Retailers
                obs_dim = 36
            
            # Extract this agent's observations and pad if necessary
            obs_structured[:, agent_id, :obs_dim] = share_obs_np[:, offset:offset+obs_dim]
            offset += obs_dim
        
        return obs_structured
    
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """Calculate value function loss (same as standard HAPPO)."""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)
        
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        
        return value_loss
    
    def ppo_update(self, sample, adj, agent_id, update_actor=True):
        """
        PPO update with adjacency matrix.
        
        Args:
            sample: Data batch from buffer
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent to update
            update_actor: Whether to update actor
        
        Returns:
            value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
        """
        (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
         actions_batch, value_preds_batch, return_batch, masks_batch,
         active_masks_batch, old_action_log_probs_batch, adv_targ,
         available_actions_batch, factor_batch) = sample
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        
        # Reconstruct structured observations from concatenated share_obs_batch
        # share_obs_batch is [batch, 594] (concatenated), we need [batch, n_agents, 36] (structured)
        obs_structured = self._reconstruct_structured_obs(share_obs_batch)
        
        # Evaluate actions with GNN (pass adjacency matrix and structured observations)
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            obs_structured, obs_structured, adj, agent_id,  # Use reconstructed structured observations
            rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch,
            available_actions_batch, active_masks_batch
        )
        
        # Actor update (same as standard HAPPO)
        imp_weights = torch.prod(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1, keepdim=True
        )
        
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) 
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        
        policy_loss = policy_action_loss
        
        self.policy.actor_optimizer.zero_grad()
        
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        
        self.policy.actor_optimizer.step()
        
        # Critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        
        self.policy.critic_optimizer.step()
        
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def train(self, buffer, adj, agent_id, update_actor=True):
        """
        Training update using minibatch GD.
        
        Args:
            buffer: Replay buffer with training data
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent to update
            update_actor: Whether to update actor
        
        Returns:
            train_info: Dictionary with training metrics
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch
                )
            
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = \
                    self.ppo_update(sample, adj, agent_id, update_actor=update_actor)
                
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
        
        num_updates = self.ppo_epoch * self.num_mini_batch
        
        for k in train_info.keys():
            train_info[k] /= num_updates
        
        return train_info
    
    def prep_training(self):
        """Set networks to training mode."""
        self.policy.actor.train()
        self.policy.critic.train()
    
    def prep_rollout(self):
        """Set networks to evaluation mode."""
        self.policy.actor.eval()
        self.policy.critic.eval()
