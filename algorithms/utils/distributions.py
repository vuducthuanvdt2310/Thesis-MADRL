import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


# class DiagGaussian(nn.Module):
#     def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
#         super(DiagGaussian, self).__init__()
#
#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
#
#         self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         self.logstd = AddBias(torch.zeros(num_outputs))
#
#     def forward(self, x, available_actions=None):
#         action_mean = self.fc_mean(x)
#
#         #  An ugly hack for my KFAC implementation.
#         zeros = torch.zeros(action_mean.size())
#         if x.is_cuda:
#             zeros = zeros.cuda()
#
#         action_logstd = self.logstd(zeros)
#         return FixedNormal(action_mean, action_logstd.exp())

import torch.nn.functional as F

class DiagGaussian(nn.Module):
    """
    Diagonal-covariance Gaussian policy head.

    Method 2 — Observation-Dependent Mean + Std:
      - fc_mean: 2-layer MLP (Linear → ReLU → Linear) so the mean can capture
        non-linear relationships between the hidden state and the optimal action.
        A single linear layer can only learn affine mappings; this often collapses
        to a constant output when observations are highly correlated.
      - fc_log_std: separate 1-layer network whose output depends on the
        observation at EACH step. Unlike a global scalar log_std parameter that
        the optimizer pushes to -∞ to reduce entropy, this network must learn
        WHEN to be uncertain (high std) and when to be confident (low std).
        Result: both mean and std genuinely vary per timestep.

    Requires retraining from scratch — NOT compatible with old single-layer .pt weights.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None, action_low=None, action_range=None):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        self.action_low   = action_low
        self.action_range = action_range

        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = args.std_y_coef
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        # ── Mean network: 2-layer MLP ──────────────────────────────────────────
        # Hidden dim = num_inputs // 2 keeps parameter count low while giving
        # the network enough capacity to learn non-linear state → action mappings.
        hidden_dim = max(num_inputs // 2, num_outputs)
        self.fc_mean = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, num_outputs)),
        )

        # ── Std network: observation-dependent ────────────────────────────────
        # A separate lightweight linear layer outputs log_std from the hidden
        # state each step. Initialized with small weights so initial std is near
        # softplus(0) + MIN_STD ≈ 0.69 + 0.05 = 0.74 — a reasonable starting point.
        self.fc_log_std = nn.Linear(num_inputs, num_outputs)
        nn.init.constant_(self.fc_log_std.weight, 0.0)
        nn.init.constant_(self.fc_log_std.bias, 0.0)

    def forward(self, x, available_actions=None, reference_demand=None):
        """
        Args:
            x: hidden state [batch, num_inputs]
            available_actions: not used for continuous
            reference_demand: demand at current step [batch, num_outputs] or None.
                              When provided, uses Approach A: Demand + Residual.
                              When None, falls back to standard tanh squashing.
        """
        # ── Action mean ─────────────────────────────────────────────────────────
        # `action_mean` is the raw MLP output — completely unbounded (-inf, +inf).
        raw_mean = self.fc_mean(x)  # shape: [batch, num_skus]

        if reference_demand is not None:
            # ── Approach A: Demand + Residual ──────────────────────────────────
            # Instead of mapping raw_mean to [0, MAX_ORDER] with tanh (which saturates
            # at the boundary and kills gradients), we compute a SMALL ADJUSTMENT
            # relative to the actual observation demand.
            #
            # max_residual = 1/3 of action range:
            #   Retailer range = 3  → max_residual = 1.0  (order demand ± 1 unit)
            #   DC range = 5000     → max_residual = 1666 (order demand ± 1666 units)
            #
            # Even if raw_mean → +∞, tanh saturates at +1.0, so scaled_mean can
            # rise at MOST by max_residual above demand — never touches the ceiling.
            # Gradient of tanh is healthy because raw_mean stays in a small range.
            #
            #   adjustment   = tanh(raw_mean) × max_residual   ∈ (-max_r, +max_r)
            #   target_mean  = reference_demand + adjustment
            #   scaled_mean  = clamp(target_mean, action_low, action_low + action_range)

            action_low   = self.action_low   if self.action_low   is not None else 0.0
            action_range = self.action_range if self.action_range is not None else 3.0
            max_residual = action_range / 3.0

            adjustment        = torch.tanh(raw_mean) * max_residual
            target_mean       = reference_demand + adjustment
            action_mean_scaled = torch.clamp(target_mean,
                                             min=action_low,
                                             max=action_low + action_range)

        elif self.action_low is not None and self.action_range is not None:
            # ── Fallback: standard tanh squashing ──────────────────────────────
            # Used for agents without explicit demand obs (e.g. DCs).
            tanh_out           = torch.tanh(raw_mean)
            action_mean_scaled = self.action_low + ((tanh_out + 1.0) / 2.0) * self.action_range
        else:
            action_mean_scaled = raw_mean

        # ── Action std (observation-dependent) ──────────────────────────────────
        # softplus(x) = log(1 + exp(x)) > 0 always → std never collapses to 0.
        # Clamped to [MIN_STD, MAX_STD] to guarantee healthy exploration every step.
        MIN_STD     = 0.05
        MAX_STD     = self.std_y_coef                   # = 1.5 (set in train script)
        log_std_raw = self.fc_log_std(x)                # [batch, num_skus], varies per step
        action_std  = torch.clamp(F.softplus(log_std_raw), min=MIN_STD, max=MAX_STD)

        return FixedNormal(action_mean_scaled, action_std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
