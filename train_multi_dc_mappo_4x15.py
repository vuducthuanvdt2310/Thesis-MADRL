#!/usr/bin/env python
"""
MAPPO Training Script for 4x15 Inventory Environment
Topology: 1 Supplier → 4 DCs → 15 Retailers (19 agents total)
"""

import sys
import os
import numpy as np
from pathlib import Path
import shutil
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnvMultiDC, DummyVecEnvMultiDC
from runners.separated.runner import CRunner as Runner

def is_running_in_colab():
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False

def mount_google_drive():
    try:
        if os.path.exists('/content/drive/MyDrive'): return True
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False); return True
    except Exception: return False

USE_GOOGLE_DRIVE = False
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/thesis_models"

N_DCS = 4
N_RETAILERS = 15
N_AGENTS = N_DCS + N_RETAILERS  # 19
CONFIG_PATH = 'configs/multi_dc_4x15_config.yaml'

def make_train_env(all_args): return SubprocVecEnvMultiDC(all_args)
def make_eval_env(all_args): return DummyVecEnvMultiDC(all_args)
def parse_args(args, parser): return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    in_colab = is_running_in_colab()
    use_gdrive = USE_GOOGLE_DRIVE or in_colab
    if use_gdrive:
        print("=" * 70); print("Google Colab Environment Detected"); print("=" * 70)
        if in_colab and not mount_google_drive(): use_gdrive = False
        print()

    BASE_SAVE_DIR = Path(GOOGLE_DRIVE_PATH) if use_gdrive else Path(os.path.dirname(os.path.abspath(__file__)))

    parser = get_config()
    parser.set_defaults(
        env_name="MultiDC", scenario_name="inventory_2echelon_4x15",
        num_agents=N_AGENTS, episode_length=365, num_env_steps=36500000,
        n_rollout_threads=4, n_training_threads=1,
        algorithm_name="mappo", experiment_name="mappo_4x15",
        use_eval=True, n_eval_rollout_threads=1, eval_interval=1,
        eval_episodes=5, log_interval=1, n_warmup_evaluations=3,
        n_no_improvement_thres=1000,
        entropy_coef=0.08, std_x_coef=2.0, std_y_coef=1.5,
    )

    all_args = parse_args(sys.argv[1:], parser)
    all_args.env_config_path = CONFIG_PATH

    RESUME_MODEL_DIR = None
    if RESUME_MODEL_DIR: all_args.model_dir = RESUME_MODEL_DIR

    seeds = all_args.seed
    if isinstance(seeds, int): seeds = [seeds]

    print("=" * 70)
    print(f"MAPPO Training — 4x15 ({N_DCS} DCs + {N_RETAILERS} Retailers)")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH} | Algorithm: MAPPO | Agents: {N_AGENTS}")
    print(f"Parallel envs: {all_args.n_rollout_threads} | Steps: {all_args.num_env_steps:,}")
    print("=" * 70)

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0"); torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu"); torch.set_num_threads(all_args.n_training_threads)

    for seed in seeds:
        print(f"\n{'='*70}\nTraining starts for seed: {seed}\n{'='*70}\n")
        run_dir = BASE_SAVE_DIR / "results" / all_args.experiment_name
        if not run_dir.exists(): os.makedirs(str(run_dir))
        seed_res_record_file = run_dir / "seed_results.txt"
        run_dir = run_dir / ('run_seed_%i' % (seed + 1))
        if not run_dir.exists(): os.makedirs(str(run_dir))
        models_dir = run_dir / "models"
        if not models_dir.exists(): os.makedirs(str(models_dir))
        if not os.path.exists(seed_res_record_file): open(seed_res_record_file, 'a+')

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None

        config = {"all_args": all_args, "envs": envs, "eval_envs": eval_envs,
                  "num_agents": all_args.num_agents, "device": device, "run_dir": run_dir}
        try:
            runner = Runner(config); reward, bw = runner.run()
            with open(seed_res_record_file, 'a+') as f:
                f.write(str(seed) + ' ' + str(reward) + ' ')
                for fluc in bw: f.write(str(fluc) + ' ')
                f.write('\n')
            print(f"\nCompleted seed {seed} | Reward: {reward}")
        except KeyboardInterrupt: print("\nInterrupted."); break
        except Exception as e:
            print(f"\nFailed: {e}"); import traceback; traceback.print_exc(); break
        finally:
            envs.close()
            if all_args.use_eval and eval_envs is not envs: eval_envs.close()
        try:
            shutil.make_archive(os.path.join(os.getcwd(), all_args.experiment_name), 'zip', run_dir)
        except Exception: pass

    print(f"\n{'='*70}\nAll MAPPO training runs completed!\n{'='*70}")
