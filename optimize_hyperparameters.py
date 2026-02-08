import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gymnasium as gym
from gymnasium import spaces

# ==============================================================================
# 1. MOCK ENVIRONMENT & AGENT (Placeholder for your actual code)
# ==============================================================================

class MockInventoryEnv(gym.Env):
    """
    A Mock Environment mimicking the Multi-DC Inventory Management problem.
    Used for testing the Optuna pipeline without the full complex simulation.
    """
    def __init__(self, num_agents=17):
        self.num_agents = num_agents
        # Observation space: 2 DCs (27 dims) + 15 Retailers (36 dims)
        self.observation_space = []
        self.action_space = []
        
        for i in range(num_agents):
            obs_dim = 27 if i < 2 else 36
            self.observation_space.append(spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32))
            # Action space: Continuous 6 dims (as per your wrapper)
            self.action_space.append(spaces.Box(low=0, high=50, shape=(6,), dtype=np.float32))
            
        self.steps = 0
        self.max_steps = 365 # One year per episode

    def reset(self):
        self.steps = 0
        obs = [np.random.rand(space.shape[0]).astype(np.float32) for space in self.observation_space]
        return np.array(obs, dtype=object), {}

    def step(self, actions):
        self.steps += 1
        # Generate random next observations
        obs = [np.random.rand(space.shape[0]).astype(np.float32) for space in self.observation_space]
        
        # Mock Reward: Higher is better (minimize cost = maximize negative cost)
        # In a real scenario, this comes from the environment.
        # Here we simulate valid rewards.
        rewards = np.random.uniform(-100, 10, size=(self.num_agents, 1))
        
        dones = [False] * self.num_agents
        truncated = [False] * self.num_agents
        
        if self.steps >= self.max_steps:
             dones = [True] * self.num_agents
             truncated = [True] * self.num_agents
             
        infos = {}
        return np.array(obs, dtype=object), rewards, np.array(dones), infos

class MockHAPPOAgent:
    """
    A simplified HAPPO Agent that initializes networks based on hyperparameters.
    """
    def __init__(self, observation_space, action_space, lr, clip_range, entropy_coef, gae_lambda, batch_size):
        self.lr = lr
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        
        # Mock Actor Network
        self.actor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0])
        )
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def select_action(self, obs):
        # Mock action selection
        with torch.no_grad():
            action_mean = self.actor(torch.FloatTensor(obs))
        return action_mean.numpy()

    def update(self, rollouts):
        # Mock update step: simulates training delay and optimization
        # In real code, this uses PPO loss, calculating advantages with gae_lambda, etc.
        self.optimizer.zero_grad()
        dummy_loss = torch.tensor(0.0, requires_grad=True) # Placeholder
        dummy_loss.backward()
        self.optimizer.step()
        return -np.random.rand() # Return a dummy loss

# ==============================================================================
# 2. OPTUNA OBJECTIVE FUNCTION
# ==============================================================================

def objective(trial):
    """
    Objective function for Optuna optimization.
    Returns the mean reward of the evaluation over the last few episodes.
    """
    # -----------------------------------------------------------
    # A. Define Hyperparameter Search Space
    # -----------------------------------------------------------
    # 1. Learning Rate (Log uniform: 1e-5 to 1e-3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # 2. Clip Range (Uniform: 0.1 to 0.3)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # 3. Entropy Coefficient (Uniform: 0.0 to 0.05)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.05)
    
    # 4. GAE Lambda (Uniform: 0.9 to 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    
    # 5. Batch Size (Categorical: 32, 64, 128)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # -----------------------------------------------------------
    # B. Initialize Environment and Agent
    # -----------------------------------------------------------
    # Use the Mock Environment (Replace with real make_train_env in production)
    env = MockInventoryEnv(num_agents=17) 
    
    # Initialize Agents (one for each agent in the environment)
    agents = []
    for i in range(env.num_agents):
        agent = MockHAPPOAgent(
            env.observation_space[i],
            env.action_space[i],
            lr=learning_rate,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            gae_lambda=gae_lambda, # Typically shared across agents
            batch_size=batch_size
        )
        agents.append(agent)
        
    # -----------------------------------------------------------
    # C. Training Loop with Pruning
    # -----------------------------------------------------------
    # For optimization, we use a shorter loop than full training.
    N_EPISODES = 100 
    EPISODE_LENGTH = 100 # Reduced for speed in this mock; keep 365 for real
    
    print(f"Trial {trial.number}: Starting training with LR={learning_rate:.2e}, Batch={batch_size}...")
    
    episode_rewards = []
    
    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        current_episode_reward = 0
        
        for step in range(EPISODE_LENGTH):
            # Select actions for all agents
            actions = []
            for i, agent in enumerate(agents):
                # Handle heterogeneous observation shapes
                action = agent.select_action(obs[i])
                actions.append(action)
            
            # Step environment
            next_obs, rewards, dones, _ = env.step(actions)
            
            # Aggregate reward (sum of all agents)
            # In your case, you want to MAXIMIZE total reward (minimize cost)
            step_total_reward = np.sum(rewards)
            current_episode_reward += step_total_reward
            
            obs = next_obs
            if all(dones):
                break
                
        # Mock Update (Training Step) happens here usually
        for agent in agents:
            agent.update(None) # Pass rollouts in real code
            
        # Store episode reward
        episode_rewards.append(current_episode_reward)
        mean_reward = np.mean(episode_rewards[-10:]) # Smooth over last 10
        
        # ------------------------------------------------=======
        # D. Report and Prune
        # ------------------------------------------------=======
        # Report intermediate objective value to Optuna
        trial.report(mean_reward, step=episode)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at episode {episode}.")
            raise optuna.TrialPruned()
            
    # Return the final metric to maximize
    final_mean_reward = np.mean(episode_rewards[-20:]) # Average of last 20 episodes
    print(f"Trial {trial.number} finished with Mean Reward: {final_mean_reward:.2f}")
    return final_mean_reward

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Define the study
    # direction="maximize" because we want to Maximize Reward (Minimize Cost)
    study = optuna.create_study(
        study_name="happo_inventory_optimization",
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(),       # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner(         # Prune trials below median of previous trials
            n_startup_trials=5,                     # Trials before pruning starts
            n_warmup_steps=10,                      # Steps in a trial before pruning starts
            interval_steps=1                        # Check pruning every step
        )
    )
    
    # 2. Run Optimization
    print("Starting optimization on CPU...")
    # Increase n_trials for better results (e.g., 50 or 100)
    # Timeout ensures it doesn't run forever (e.g., 1 hour = 3600 seconds)
    study.optimize(objective, n_trials=20, timeout=3600) 
    
    # 3. Print Results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "="*50)
    print("Optimization Results")
    print("="*50)
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    print("\nBest Trial:")
    trial = study.best_trial
    print(f"  Value (Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # 4. Visualization (Saved to files if running locally)
    try:
        print("\nGenerating visualization plots...")
        # Optimization History
        fig1 = plot_optimization_history(study)
        fig1.write_html("optuna_optimization_history.html")
        print("  Saved: optuna_optimization_history.html")
        
        # Hyperparameter Importance
        fig2 = plot_param_importances(study)
        fig2.write_html("optuna_param_importance.html")
        print("  Saved: optuna_param_importance.html")
        
    except ImportError:
        print("  Visualization libraries (plotly/kaleido) not installed. Skipping plots.")
    except Exception as e:
        print(f"  Error generating plots: {e}")

