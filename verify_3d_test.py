"""Quick shape-compatibility test for 3D action space."""
import sys, os
sys.path.insert(0, r'd:\thuan\thesis\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management')
os.chdir(r'd:\thuan\thesis\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management')

import numpy as np
from envs.multi_dc_env import MultiDCInventoryEnv
from envs.env_wrappers import SubprocVecEnvMultiDC, DummyVecEnvMultiDC

CONFIG = r'd:\thuan\thesis\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management\configs\multi_dc_config.yaml'

# ── 1. Raw environment ─────────────────────────────────────────────────────
env = MultiDCInventoryEnv(CONFIG)
print(f"[ENV] action_dim    = {env.action_dim}")         # 3
print(f"[ENV] action_space  = {env.action_space}")       # Box(3,)
print(f"[ENV] obs_dim_dc    = {env.obs_dim_dc}")         # 28
print(f"[ENV] obs_dim_ret   = {env.obs_dim_retailer}")   # 22
assert env.action_dim == 3, f"Expected action_dim=3, got {env.action_dim}"

obs = env.reset()
assert obs[0].shape == (28,), f"DC obs shape wrong: {obs[0].shape}"
assert obs[2].shape == (22,), f"Retailer obs shape wrong: {obs[2].shape}"
print(f"[ENV] obs[0].shape  = {obs[0].shape}")
print(f"[ENV] obs[2].shape  = {obs[2].shape}")

# 3D actions
actions = {i: np.random.uniform(0, 70, 3).astype(np.float32) for i in range(env.n_agents)}
obs2, rew, done, info = env.step(actions)
assert obs2[0].shape == (28,)
assert 'step_service_level' in info[0]
print(f"[ENV] step reward[0] = {rew[0]:.4f}")
print(f"[ENV] service_level  = {info[0]['step_service_level']:.3f}")
print("[ENV] ✓ Raw env OK")

# ── 2. Wrapper compatibility ───────────────────────────────────────────────
class Args:
    n_rollout_threads = 1
    config_path = CONFIG

args = Args()

# SubprocVecEnvMultiDC
wrapper = SubprocVecEnvMultiDC(args)
assert wrapper.action_dim == 3, f"Wrapper action_dim wrong: {wrapper.action_dim}"
assert wrapper.max_obs_dim == 28, f"Wrapper max_obs_dim wrong: {wrapper.max_obs_dim}"
assert wrapper.action_space[0].shape == (3,), f"Wrapper action_space shape wrong"
obs_w, _ = wrapper.reset()
assert obs_w.shape == (1, 17, 28), f"Wrapper reset obs shape wrong: {obs_w.shape}"
print(f"[WRAPPER] Subproc reset obs shape = {obs_w.shape}")

# Step with 3D actions (shape: [n_threads, n_agents, action_dim=3])
act_w = [[np.random.uniform(0, 70, 3).astype(np.float32) for _ in range(17)]]
obs_w2, rew_w, done_w, info_w = wrapper.step(act_w)
assert obs_w2.shape == (1, 17, 28), f"Wrapper step obs shape wrong: {obs_w2.shape}"
assert rew_w.shape == (1, 17, 1), f"Wrapper step rew shape wrong: {rew_w.shape}"
print(f"[WRAPPER] Subproc step obs   = {obs_w2.shape}")
print(f"[WRAPPER] Subproc step rew   = {rew_w.shape}")
print("[WRAPPER] ✓ SubprocVecEnvMultiDC OK")

# DummyVecEnvMultiDC
dummy = DummyVecEnvMultiDC(args)
assert dummy.action_dim == 3
obs_d, _ = dummy.reset()
assert obs_d.shape == (1, 17, 28)
act_d = [[np.random.uniform(0, 70, 3).astype(np.float32) for _ in range(17)]]
obs_d2, rew_d, done_d, info_d = dummy.step(act_d)
assert obs_d2.shape == (1, 17, 28)
print(f"[WRAPPER] DummyVecEnv step obs = {obs_d2.shape}")
print("[WRAPPER] ✓ DummyVecEnvMultiDC OK")

# ── 3. Buffer shape check ──────────────────────────────────────────────────
# Simulate what the runner does when inserting to buffer
from utils.separated_buffer import SeparatedReplayBuffer
from argparse import Namespace
buf_args = Namespace(
    episode_length=5, n_rollout_threads=1, hidden_size=64,
    recurrent_N=1, gamma=0.99, gae_lambda=0.95, use_gae=True,
    use_popart=False, use_valuenorm=True, use_proper_time_limits=False,
)
import gymnasium.spaces as spaces_gym
obs_space = spaces_gym.Box(low=0, high=1, shape=(28,), dtype=np.float32)
act_space = spaces_gym.Box(low=0, high=70, shape=(3,), dtype=np.float32)

buf = SeparatedReplayBuffer(buf_args, obs_space, obs_space, act_space)
print(f"[BUFFER] actions buffer shape  = {buf.actions.shape}")  # (5, 1, 3)
assert buf.actions.shape[-1] == 3, f"Buffer action shape wrong: {buf.actions.shape}"
print("[BUFFER] ✓ SeparatedReplayBuffer action shape OK")

print()
print("=" * 60)
print("ALL SHAPE TESTS PASSED — 3D action space is compatible ✓")
print("=" * 60)
