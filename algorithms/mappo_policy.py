import torch
from algorithms.actor_critic import Actor, Critic
from utils.util import update_linear_schedule


class MAPPO_Policy:
    """
    MAPPO Policy class. Wraps actor and critic networks to compute actions and value function
    predictions.

    Unlike HAPPO, MAPPO treats all agents independently and does NOT use a sequential
    importance-weight factor between agents.  The centralized critic still takes the global
    state (concatenated observations) as input, providing the "centralized training,
    decentralized execution" (CTDE) paradigm.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) local observation space for this agent.
    :param cent_obs_space: (gym.Space) centralized (global) state space fed to the critic.
    :param act_space: (gym.Space) action space.
    :param device: (torch.device) device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates linearly.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False, agent_id=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized global state input to the critic.
        :param obs (np.ndarray): local agent observations for the actor.
        :param rnn_states_actor: (np.ndarray) RNN states for actor.
        :param rnn_states_critic: (np.ndarray) RNN states for critic.
        :param masks: (np.ndarray) reset flags for RNN states.
        :param available_actions: (np.ndarray) mask of legal actions (None = all legal).
        :param deterministic: (bool) if True return the mode of the distribution.
        :param agent_id: (int) index of this agent (passed to actor for heterogeneous handling).

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) sampled (or deterministic) actions.
        :return action_log_probs: (torch.Tensor) log-probabilities of the chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic, agent_id=agent_id
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions from the centralized critic.
        :param cent_obs (np.ndarray): centralized global state.
        :param rnn_states_critic: (np.ndarray) RNN states for critic.
        :param masks: (np.ndarray) reset flags for RNN states.
        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action,
                         masks, available_actions=None, active_masks=None, agent_id=None):
        """
        Get action log-probs / entropy and value predictions for actor update.
        :param cent_obs (np.ndarray): centralized global state input to the critic.
        :param obs (np.ndarray): local agent observations for the actor.
        :param rnn_states_actor: (np.ndarray) RNN states for actor.
        :param rnn_states_critic: (np.ndarray) RNN states for critic.
        :param action: (np.ndarray) actions whose log-probabilities and entropy to compute.
        :param masks: (np.ndarray) reset flags for RNN states.
        :param available_actions: (np.ndarray) mask of legal actions.
        :param active_masks: (torch.Tensor) whether agent is active at this time step.
        :param agent_id: (int) index of this agent.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log-probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks,
            agent_id=agent_id
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None,
            deterministic=False, agent_id=None):
        """
        Compute actions without returning value estimates (used during evaluation).
        :param obs (np.ndarray): local agent observations for the actor.
        :param rnn_states_actor: (np.ndarray) RNN states for actor.
        :param masks: (np.ndarray) reset flags for RNN states.
        :param available_actions: (np.ndarray) mask of legal actions.
        :param deterministic: (bool) if True return the mode of the distribution.
        :param agent_id: (int) index of this agent.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic, agent_id=agent_id
        )
        return actions, rnn_states_actor
