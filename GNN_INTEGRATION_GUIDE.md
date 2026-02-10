# GNN-HAPPO Integration Guide

## Summary

✅ **ALL GNN-HAPPO COMPONENTS ARE WORKING CORRECTLY!**

The test script (`test_gnn_components.py`) successfully verified that:
- GNN base layers (GAT/GCN) work correctly
- GNN Actor produces valid actions and log probabilities  
- GNN Critic produces valid value estimates
- Policy wrapper integrates actor and critic properly
- Trainer can handle the GNN policy

## What Works Now

The following GNN-HAPPO components are fully functional:

| Component | File | Status |
|-----------|------|--------|
| GAT Layer | `algorithms/gnn/gnn_base.py` | ✅ Tested |
| GCN Layer | `algorithms/gnn/gnn_base.py` | ✅ Tested |
| GNN Actor | `algorithms/gnn/gnn_actor.py` | ✅ Tested |
| GNN Critic | `algorithms/gnn/gnn_critic.py` | ✅ Tested |
| GNN Policy | `algorithms/gnn_happo_policy.py` | ✅ Tested |
| GNN Trainer | `algorithms/gnn_happo_trainer.py` | ✅ Tested |
| Graph Utils | `utils/graph_utils.py` | ✅ Tested |

## What Needs Integration

To run **full training** with `train_multi_dc_gnn.py`, you need to modify the runner to pass the adjacency matrix through the data pipeline.

### Critical Issue

The baseline `runners/separated/runner.py` was designed for standard MLP-based policies, which only need observations. GNN policies additionally need the **adjacency matrix** in three places:

1. **`collect()`** - When getting actions from policy
2. **`compute()`** - When computing value estimates
3. **`train()`** - When training with the trainer

### Option 1: Quick Modification (Recommended for Testing)

Modify the existing `runners/separated/runner.py` directly:

**Step 1**: Add adjacency matrix in `__init__()` (after line 15):

```python
# Add after super(CRunner, self).__init__(config)
if self.algorithm_name == "gnn_happo":
    from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15, self_loops=True)
    adj = normalize_adjacency(adj, method='symmetric')
    self.adj_tensor = torch.FloatTensor(adj).to(self.device)
    print(f"\\n✓ Supply chain graph created: {adj.shape[0]} nodes\\n")
else:
    self.adj_tensor = None
```

**Step 2**: Modify `collect()` method (around line 270):

Replace:
```python
value, action, action_log_prob, rnn_state, rnn_state_critic \
    = self.trainer[agent_id].policy.get_actions(
        self.buffer[agent_id].share_obs[step],
        self.buffer[agent_id].obs[step],
        self.buffer[agent_id].rnn_states[step],
        self.buffer[agent_id].rnn_states_critic[step],
        self.buffer[agent_id].masks[step],
        avail_actions)
```

With:
```python
if self.algorithm_name == "gnn_happo":
    # GNN needs all agent observations + adjacency matrix
    # Reshape share_obs to [batch, n_agents, obs_dim]
    share_obs_batch = self.buffer[agent_id].share_obs[step]
    obs_batch = share_obs_batch  # For GNN, this should be all agent obs
    
    value, action, action_log_prob, rnn_state, rnn_state_critic \
        = self.trainer[agent_id].policy.get_actions(
            share_obs_batch,  # centralized obs
            obs_batch,        # all agent obs  
            self.adj_tensor,  # adjacency matrix
            agent_id,         # which agent
            self.buffer[agent_id].rnn_states[step],
            self.buffer[agent_id].rnn_states_critic[step],
            self.buffer[agent_id].masks[step],
            avail_actions)
else:
    # Standard HAPPO (baseline)
    value, action, action_log_prob, rnn_state, rnn_state_critic \
        = self.trainer[agent_id].policy.get_actions(
            self.buffer[agent_id].share_obs[step],
            self.buffer[agent_id].obs[step],
            self.buffer[agent_id].rnn_states[step],
            self.buffer[agent_id].rnn_states_critic[step],
            self.buffer[agent_id].masks[step],
            avail_actions)
```

**Step 3**: Modify `compute()` method (in base_runner.py around line 120):

Replace:
```python
next_value = self.trainer[agent_id].policy.get_values(
    self.buffer[agent_id].share_obs[-1], 
    self.buffer[agent_id].rnn_states_critic[-1],
    self.buffer[agent_id].masks[-1])
```

With:
```python
if self.algorithm_name == "gnn_happo":
    next_value = self.trainer[agent_id].policy.get_values(
        self.buffer[agent_id].share_obs[-1],
        self.adj_tensor,  # adjacency
        self.buffer[agent_id].rnn_states_critic[-1],
        self.buffer[agent_id].masks[-1])
else:
    next_value = self.trainer[agent_id].policy.get_values(
        self.buffer[agent_id].share_obs[-1], 
        self.buffer[agent_id].rnn_states_critic[-1],
        self.buffer[agent_id].masks[-1])
```

**Step 4**: Modify `train()` method (in base_runner.py around line 145):

Replace:
```python
train_info = self.trainer[agent_id].train(self.buffer[agent_id])
```

With:
```python
if self.algorithm_name == "gnn_happo":
    train_info = self.trainer[agent_id].train(
        self.buffer[agent_id],
        self.adj_tensor,  # adjacency
        agent_id          # agent ID
    )
else:
    train_info = self.trainer[agent_id].train(self.buffer[agent_id])
```

### Option 2: Create Separate GNN Runner (Cleaner)

Create `runners/separated/gnn_runner.py` that inherits from `CRunner` and overrides only the necessary methods. This keeps baseline code untouched.

I've already created the skeleton in `runners/separated/gnn_base_runner.py`, you would need to complete the `collect()`, `warmup()`, and `run()` methods by copying from `runner.py` and adding the adjacency matrix logic shown above.

---

## Data Flow Issue: share_obs Format

**CRITICAL**: The current environment returns `share_obs` as a **concatenated flat vector** (all agent obs flattened into one big array). GNN needs it as **structured [batch, n_agents, obs_dim]**.

You have two options:

### Option A: Reshape in Runner (Simpler)
In the `collect()` method, reshape the centralized obs before passing to GNN:

```python
# share_obs comes as [batch, total_obs_dim] where total_obs_dim = sum of all agent obs
# We need [batch, n_agents, obs_dim] for GNN

# For Multi-DC: 2 DCs (27D each) + 15 Retailers (36D each)
# Total = 2*27 + 15*36 = 54 + 540 = 594

# Reshape logic (pseudo-code):
share_obs_structured = np.zeros((batch_size, n_agents, max_obs_dim))
for i in range(n_agents):
    if i < 2:  # DC
        share_obs_structured[:, i, :27] = share_obs[:, i*27:(i+1)*27]
    else:  # Retailer
        offset = 2*27 + (i-2)*36
        share_obs_structured[:, i, :36] = share_obs[:, offset:offset+36]
```

### Option B: Modify Environment Wrapper (Cleaner)

Modify `envs/env_wrappers.py` to return observations in structured format for GNN.

---

## Testing the Integration

Once you've made the above changes, test with a short run:

```bash
python train_multi_dc_gnn.py \
  --experiment_name gnn_quick_test \
  --num_env_steps 3650 \  # Just 10 episodes
  --n_rollout_threads 1 \
  --gnn_type GAT \
  --seed 0
```

**Expected output:**
- ✓ Graph created: 17 nodes
- ✓ Training starts
- ✓ Models saved when reward improves
- ✓ TensorBoard logs appear in `results/gnn_quick_test/run_seed_1/logs/`

---

## Viewing TensorBoard

After running training (even just 10 episodes):

```bash
tensorboard --logdir results/gnn_quick_test/run_seed_1/logs
```

Then open http://localhost:6006 to see:
- `eval/average_reward` - Evaluation rewards over time
- `agent0/value_loss` - Critic loss for agent 0
- `agent0/policy_loss` - Actor loss for agent 0
- `agent0/dist_entropy` - Exploration entropy

---

## Current Status

| Task | Status | Notes |
|------|--------|-------|
| GNN Components | ✅ Done | All tests passing |
| Graph Utils | ✅ Done | Build / normalize / visualize |
| GNN Policy | ✅ Done | Actor + Critic wrapped |
| GNN Trainer | ✅ Done | Modified HAPPO trainer |
| Runner Integration | ⚠️ TODO | Need to modify `runner.py` |
| Data Format Fix | ⚠️ TODO | Reshape share_obs for GNN |
| Full Training Test | ⚠️ Pending | Waiting for integration |

---

## Recommendation

**For your thesis**, I recommend **Option 1** (Quick Modification) to get training working quickly. The changes are minimal and you can easily switch between baseline and GNN by changing `--algorithm_name`.

Once training works, you can:
1. Run baseline (`train_multi_dc_baseline.py`)
2. Run GNN (`train_multi_dc_gnn.py` with modified runner)
3. Compare results in TensorBoard
4. Generate comparative plots for thesis

---

## Next Steps

1. **Immediate**: Modify `runners/separated/runner.py` with the 4 changes above
2. **Test**: Run `python train_multi_dc_gnn.py` for 10 episodes
3. **Verify**: Check TensorBoard shows training curves
4. **Full Run**: Train both baseline and GNN for 10k episodes
5. **Thesis**: Compare and analyze results

---

## Need Help?

If you encounter errors during integration:
1. Check that `algorithm_name == "gnn_happo"` in your args
2. Verify adjacency matrix shape is `[17, 17]`
3. Ensure share_obs is reshaped to `[batch, n_agents, obs_dim]`
4. Test individual components with `python test_gnn_components.py`

The GNN architecture is solid - integration is just about connecting the data pipeline! 🚀
