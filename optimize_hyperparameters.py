"""
optimize_hyperparameters.py
===========================
Hyperparameter optimization for HAPPO on the real Multi-DC Inventory Environment.

Uses Optuna (TPE sampler + Median pruner) to search over:
  - learning_rate  → --lr
  - clip_param     → --clip_param
  - entropy_coef   → --entropy_coef
  - gae_lambda     → --gae_lambda
  - gamma          → --gamma
  - hidden_size    → --hidden_size

Each trial runs a SHORT training session on DummyVecEnvMultiDC (your real env)
and returns the actual eval reward from the supply chain simulation.

To run:
    python optimize_hyperparameters.py

After completion, copy the best params into train_multi_dc_baseline.py's
parser.set_defaults() block, mapping them by the arg names above.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import torch
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
from pathlib import Path

# ---- Project imports ----
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from runners.separated.runner import CRunner as Runner

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# ---- episodes per trial: 20 episodes × 30 steps = 600 env steps ----
# Small enough to finish in ~1 min per trial, big enough to rank hyperparams.
EPISODES_PER_TRIAL = 20
EPISODE_LENGTH     = 150
N_OPTUNA_STEPS     = EPISODES_PER_TRIAL * EPISODE_LENGTH  # = 600

# Number of Optuna trials to run
N_TRIALS = 15

# Maximum wall-clock seconds for the whole study (safety timeout)
TIMEOUT_SECONDS = 2 * 3600  # 2 hours

# ==============================================================================
# HELPERS
# ==============================================================================

def build_args(trial):
    """
    Build an all_args namespace for one Optuna trial.
    IMPORTANT: We assign speed settings DIRECTLY to all_args AFTER parse_args().
    This is necessary because argparse's action='store_true' ignores set_defaults()
    for boolean flags — so use_naive_recurrent_policy would stay True otherwise.
    """
    parser = get_config()
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=17,
        algorithm_name="happo",
        experiment_name=f"optuna_trial_{trial.number}",
        seed=[0],
    )
    all_args = parser.parse_args([])

    # -------------------------------------------------------
    # SPEED SETTINGS — assigned directly to bypass argparse bug
    # -------------------------------------------------------
    all_args.episode_length             = EPISODE_LENGTH  # 30 days (not 365)
    all_args.num_env_steps              = N_OPTUNA_STEPS  # 600 total steps
    all_args.n_rollout_threads          = 1
    all_args.n_eval_rollout_threads     = 1
    all_args.n_training_threads         = 1
    # Eval fires ONCE at the last episode so runner gets a real reward (not -inf)
    all_args.use_eval                   = True
    all_args.eval_interval              = EPISODES_PER_TRIAL - 1  # = 19 → fires at episode 19
    all_args.use_naive_recurrent_policy = False   # DISABLE LSTM → use fast MLP
    all_args.use_recurrent_policy       = False   # DISABLE LSTM
    all_args.ppo_epoch                  = 5       # Default 15 → 3× fewer update passes
    all_args.hidden_size                = 64      # Small network for fast forward/backward
    all_args.log_interval               = 9999    # Suppress console noise
    all_args.n_warmup_evaluations       = 0       # No warmup — let best_reward update in first eval
    all_args.n_no_improvement_thres     = 99999   # Never early-stop (only 1 eval run anyway)

    # -------------------------------------------------------
    # HYPERPARAMETERS to search — the ones that matter most
    # -------------------------------------------------------
    all_args.lr           = trial.suggest_float("lr",           1e-5, 5e-4, log=True)
    all_args.critic_lr    = trial.suggest_float("critic_lr",    1e-5, 5e-4, log=True)
    all_args.clip_param   = trial.suggest_float("clip_param",   0.1,  0.3)
    all_args.entropy_coef = trial.suggest_float("entropy_coef", 0.0,  0.05)
    all_args.gae_lambda   = trial.suggest_float("gae_lambda",   0.9,  0.99)
    all_args.gamma        = trial.suggest_float("gamma",        0.90, 0.99)

    return all_args


# ==============================================================================
# OPTUNA OBJECTIVE
# ==============================================================================

def objective(trial):
    """
    Real objective: runs a short HAPPO training session on the actual
    Multi-DC Inventory environment and returns the eval reward.
    """
    all_args = build_args(trial)

    # Use a temporary directory per trial so logs/models don't accumulate
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_"))
    run_dir = tmp_dir / "run"
    os.makedirs(str(run_dir / "models"), exist_ok=True)

    device = torch.device("cpu")  # CPU for parallelism safety; change to cuda:0 if single-GPU

    print(f"\n[Trial {trial.number}] lr={all_args.lr:.2e}, clip={all_args.clip_param:.3f}, "
          f"ent={all_args.entropy_coef:.4f}, gae_λ={all_args.gae_lambda:.3f}, "
          f"γ={all_args.gamma:.3f}, hidden={all_args.hidden_size}, ppo_epoch={all_args.ppo_epoch}")

    try:
        # Create environments (single thread for isolation)
        envs      = DummyVecEnvMultiDC(all_args)
        eval_envs = DummyVecEnvMultiDC(all_args)

        config = {
            "all_args":  all_args,
            "envs":      envs,
            "eval_envs": eval_envs,
            "num_agents":all_args.num_agents,
            "device":    device,
            "run_dir":   run_dir,
        }

        runner = Runner(config)

        # -------------------------------------------------------
        # Run training — the runner returns (best_reward, best_bw)
        # -------------------------------------------------------
        best_reward, _ = runner.run()

        # Report to Optuna (used by pruner and for result inspection)
        trial.report(best_reward, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        print(f"[Trial {trial.number}] best eval reward = {best_reward:.2f}")
        return best_reward

    except optuna.TrialPruned:
        raise

    except Exception as e:
        print(f"[Trial {trial.number}] FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        # Return a very bad value so Optuna deprioritises this region
        return float("-inf")

    finally:
        try:
            envs.close()
        except Exception:
            pass
        try:
            eval_envs.close()
        except Exception:
            pass
        # Clean up temporary files (logs, models) to save disk space
        try:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        except Exception:
            pass


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HAPPO Hyperparameter Optimization (Real Environment)")
    print("=" * 70)
    print(f"  Steps per trial : {N_OPTUNA_STEPS:,}")
    print(f"  Number of trials: {N_TRIALS}")
    print(f"  Timeout         : {TIMEOUT_SECONDS // 3600}h")
    print("=" * 70 + "\n")

    # Create Optuna study
    study = optuna.create_study(
        study_name="happo_inventory_real",
        direction="maximize",           # Maximize eval reward (= minimize cost)
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,         # Run 5 full trials before pruning
            n_warmup_steps=1,
            interval_steps=1,
        ),
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print(f"  Finished trials : {len(study.trials)}")
    print(f"  Pruned trials   : {len(pruned_trials)}")
    print(f"  Complete trials : {len(complete_trials)}")

    best = study.best_trial
    print(f"\nBest Trial #{best.number}")
    print(f"  Eval Reward: {best.value:.2f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    # ------------------------------------------------------------------
    # Print ready-to-paste block for train_multi_dc_baseline.py
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Copy-paste into train_multi_dc_baseline.py → parser.set_defaults(...):")
    print("=" * 70)
    p = best.params
    print(f"    lr           = {p.get('lr', 1e-4)},")
    print(f"    critic_lr    = {p.get('critic_lr', 1e-4)},")
    print(f"    clip_param   = {p.get('clip_param', 0.2)},")
    print(f"    entropy_coef = {p.get('entropy_coef', 0.01)},")
    print(f"    gae_lambda   = {p.get('gae_lambda', 0.95)},")
    print(f"    gamma        = {p.get('gamma', 0.95)},")
    print(f"    hidden_size  = {p.get('hidden_size', 128)},")
    print(f"    ppo_epoch    = {p.get('ppo_epoch', 15)},")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    try:
        print("\nGenerating visualization plots...")
        fig1 = plot_optimization_history(study)
        fig1.write_html("optuna_optimization_history.html")
        print("  Saved: optuna_optimization_history.html")

        fig2 = plot_param_importances(study)
        fig2.write_html("optuna_param_importance.html")
        print("  Saved: optuna_param_importance.html")
    except ImportError:
        print("  Visualization libraries not installed. Skipping plots.")
    except Exception as e:
        print(f"  Error generating plots: {e}")
