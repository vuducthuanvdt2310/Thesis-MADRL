@echo off
REM Quick Test Example Script for Windows
REM This shows how to test a trained model

echo ==================================
echo MADRL Model Testing Example
echo ==================================
echo.

REM Example 1: Test with default settings (50 episodes)
echo Example 1: Basic testing (50 episodes)
echo Command:
echo   python test_trained_model.py --model_dir results/full_training/run_seed_1/models
echo.

REM Example 2: More comprehensive testing
echo Example 2: Comprehensive testing (100 episodes)
echo Command:
echo   python test_trained_model.py ^
echo       --model_dir results/full_training/run_seed_1/models ^
echo       --num_episodes 100 ^
echo       --save_dir thesis_results
echo.

REM Example 3: Quick test (10 episodes for debugging)
echo Example 3: Quick test (10 episodes)
echo Command:
echo   python test_trained_model.py ^
echo       --model_dir results/full_training/run_seed_1/models ^
echo       --num_episodes 10 ^
echo       --experiment_name quick_test
echo.

echo ==================================
echo After running, check the evaluation_results/ folder for:
echo   - evaluation_metrics.json (all metrics)
echo   - episode_metrics.csv (for Excel analysis)
echo   - PNG charts (for thesis figures)
echo ==================================
echo.
pause
