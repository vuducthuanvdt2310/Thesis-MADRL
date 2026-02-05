# Testing Guide: Evaluating Trained MADRL Models

This guide explains how to test your trained Multi-Agent Deep Reinforcement Learning (MADRL) models to prove they can solve the inventory optimization problem.

## Overview

After training your model using `train_multi_dc.py`, use `test_trained_model.py` to:

1. **Load** a saved model from the training results
2. **Run** it on the environment for multiple episodes (without exploration)
3. **Collect** detailed performance metrics (costs, inventory, backlog, service levels)
4. **Visualize** the results with comprehensive plots
5. **Export** data to JSON/CSV for your thesis

---

## Quick Start

### 1. Find Your Trained Model

After training, your models are saved in:
```
results/
└── experiment_name/          # Your experiment name from training
    └── run_seed_1/           # Run with seed 1
        └── models/           # Contains trained model files
            ├── actor_agent0.pt
            ├── actor_agent1.pt
            ├── actor_agent2.pt
            ├── actor_agent3.pt
            └── actor_agent4.pt
```

### 2. Run Evaluation

**Basic usage** (50 episodes):
```bash
python test_trained_model.py --model_dir results/full_training/run_seed_1/models
```

**Custom evaluation** (more episodes for better statistics):
```bash
python test_trained_model.py \
    --model_dir results/full_training/run_seed_1/models \
    --num_episodes 100 \
    --episode_length 365 \
    --save_dir my_evaluation_results
```

### 3. View Results

Results are saved in `evaluation_results/eval_TIMESTAMP/`:

```
evaluation_results/
└── eval_20260201_120000/
    ├── evaluation_metrics.json      # All metrics in JSON format
    ├── episode_metrics.csv          # Episode-by-episode data for Excel/analysis
    ├── episode_rewards.png          # Performance across episodes
    ├── cost_breakdown.png           # Cost distribution by agent
    ├── service_levels.png           # Service level achievement
    ├── detailed_trajectory.png      # Inventory/backlog over time (1st episode)
    └── performance_distribution.png # Statistical distribution of results
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_dir` | **Yes** | - | Path to saved model directory |
| `--num_episodes` | No | 50 | Number of evaluation episodes |
| `--episode_length` | No | 365 | Days per episode |
| `--save_dir` | No | `evaluation_results` | Output directory |
| `--experiment_name` | No | `eval_TIMESTAMP` | Name for this evaluation |
| `--num_agents` | No | 5 | Number of agents (2 DCs + 3 Retailers) |
| `--cuda` | No | True | Use GPU if available |

---

## Output Metrics

### Overall Performance
- **Total Reward**: Sum of all agent rewards (negative total cost)
- **Total Cost**: Holding + Backlog + Ordering costs across all agents
- **Best/Worst Episode**: Performance range across episodes

### Per-Agent Metrics
For each agent (2 DCs + 3 Retailers):

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Average Cost** | Mean total cost per episode | Lower is better |
| **Average Inventory** | Mean inventory level | Balance: not too high (holding costs) or low (stockouts) |
| **Average Backlog** | Mean unfulfilled demand | Lower is better (fewer stockouts) |
| **Service Level %** | % of time with inventory > 0 | Higher is better (target: >95%) |

### Cost Components
- **Holding Cost**: Cost of storing inventory
- **Backlog Cost**: Penalty for unfulfilled demand (stockouts)
- **Ordering Cost**: Fixed + variable costs for ordering

---

## Visualizations Explained

### 1. **episode_rewards.png**
- Shows total reward across all evaluation episodes
- Red line = moving average (smoothed trend)
- **Use in thesis**: Demonstrates stable, consistent policy performance

### 2. **cost_breakdown.png**
- Bar chart comparing average costs per agent
- Colors: Blue = DCs, Red = Retailers
- **Use in thesis**: Shows which agents/echelons contribute most to total cost

### 3. **service_levels.png**
- Shows % of time each agent has positive inventory
- Red dashed line = target service level (95%)
- **Use in thesis**: Proves MADRL can maintain high service levels

### 4. **detailed_trajectory.png**
- Three subplots showing inventory, backlog, and rewards over 365 days
- Shows first episode in detail
- **Use in thesis**: Demonstrates how MADRL balances inventory vs. stockouts

### 5. **performance_distribution.png**
- Histograms of reward and cost distributions
- Shows consistency of policy performance
- **Use in thesis**: Statistical evidence of reliable performance

---

## Example: Complete Workflow

### Step 1: Train Model
```bash
python train_multi_dc.py --experiment_name my_experiment --num_env_steps 365000
```

### Step 2: Wait for Training to Complete
- Training saves models automatically when performance improves
- Look for: `"A better model is saved!"`
- Models saved to: `results/my_experiment/run_seed_1/models/`

### Step 3: Test Model
```bash
python test_trained_model.py \
    --model_dir results/my_experiment/run_seed_1/models \
    --num_episodes 100 \
    --save_dir thesis_evaluation
```

### Step 4: Analyze Results
1. Open `evaluation_results/eval_TIMESTAMP/evaluation_metrics.json`
2. Check average total cost and service levels
3. Use PNG images in your thesis presentation
4. Use CSV for additional analysis in Excel/Python

---

## For Your Thesis

### Proving MADRL Solves Inventory Optimization

Use this testing script to demonstrate:

1. **Cost Minimization**: Show that average total cost is maintained at reasonable levels
2. **Service Level Achievement**: Demonstrate >90-95% service levels
3. **Stability**: Show consistent performance across many episodes (low variance)
4. **Multi-Agent Coordination**: Show that DCs and Retailers coordinate effectively (via cost breakdown)

### Typical Results to Report

Example table for your thesis:

| Metric | Value |
|--------|-------|
| Average Total Cost per Episode | -12,543 ± 1,234 |
| Average Service Level (All Agents) | 94.3% |
| Best Episode Total Cost | -10,234 |
| Episodes Evaluated | 100 |

### Figures for Your Thesis
1. **Figure 1**: Episode rewards plot (shows learning stability)
2. **Figure 2**: Service levels bar chart (shows MADRL meets service targets)
3. **Figure 3**: Detailed trajectory (shows inventory management behavior)
4. **Figure 4**: Cost breakdown (shows multi-echelon coordination)

---

## Troubleshooting

### Error: "Model directory not found"
- **Solution**: Check that `--model_dir` path is correct
- Models should be in `results/experiment_name/run_seed_X/models/`

### Error: "Actor model not found: actor_agent0.pt"
- **Solution**: Training didn't save models. Check that training completed successfully and models were saved

### Low service levels (<80%)
- **Possible cause**: Model undertrained or poor hyperparameters
- **Solution**: Train longer or adjust hyperparameters in `train_multi_dc.py`

### High variance in results
- **Solution**: Run more evaluation episodes (e.g., `--num_episodes 200`)

---

## Next Steps

After evaluating your model:

1. **Compare Results**: Run evaluation on models from different training runs
2. **Sensitivity Analysis**: Test with different demand patterns or cost parameters
3. **Baseline Comparison**: Compare MADRL vs. simple policies (e.g., (s,S) policy, fixed order quantity)

For baseline comparisons, you would need to create additional testing scripts or modify `test_trained_model.py`.

---

## Questions?

This testing script provides comprehensive evidence that your MADRL approach can:
- ✓ Minimize inventory costs
- ✓ Maintain high service levels
- ✓ Coordinate multiple agents across echelons
- ✓ Perform consistently across many episodes

Perfect for demonstrating in your thesis that **MADRL solves the multi-echelon inventory optimization problem**! 🎓
