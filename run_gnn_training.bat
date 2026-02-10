@echo off
REM Simple script to run GNN-HAPPO training without complex command-line arguments
echo Starting GNN-HAPPO Training...
echo ================================
python train_multi_dc_gnn.py --experiment_name gnn_happo_thesis --num_env_steps 36500000 --n_rollout_threads 10 --seed 0
echo.
echo ================================
echo Training complete!
pause
