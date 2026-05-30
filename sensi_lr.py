import argparse
import subprocess
import sys
import time


LEARNING_RATES = [0.001, 0.0005, 0.0001]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis for GNN-HAPPO learning rate"
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of training episodes per learning-rate value (default: 50)",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=365,
        help="Length of each episode in days (default: 365)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=4,
        help="Number of parallel rollout threads (default: 4)",
    )

    return parser.parse_args()


def format_elapsed(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs}h {mins}m {secs}s"


def main(args):
    num_env_steps = args.num_episodes * args.episode_length

    print("=" * 80)
    print("Starting Sensitivity Analysis for lr")
    print(f"Values to test: {LEARNING_RATES}")
    print(f"Training for target ~{args.num_episodes} episodes per configuration")
    print(f"Episode length: {args.episode_length} days")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    sensitivity_results = []
    start_total_time = time.time()

    for lr in LEARNING_RATES:
        print(f"\n{'#' * 80}")
        print(f"Starting Run: lr = {lr} (seed = {args.seed})")
        print(f"{'#' * 80}\n")

        experiment_name = f"gnn_happo_sensi_lr_{lr}"

        cmd = [
            sys.executable,
            "train_multi_dc_gnn.py",
            "--lr",
            str(lr),
            "--num_env_steps",
            str(num_env_steps),
            "--episode_length",
            str(args.episode_length),
            "--experiment_name",
            experiment_name,
            "--n_rollout_threads",
            str(args.n_rollout_threads),
            "--seed",
            str(args.seed),
        ]

        print(f"Executing: {' '.join(cmd)}\n")
        start_time = time.time()

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error during training with lr={lr}: {exc}")
            print("Continuing to the next value...")
            continue

        elapsed_seconds = time.time() - start_time
        time_str = format_elapsed(elapsed_seconds)

        sensitivity_results.append(
            {
                "lr": lr,
                "time_taken_sec": elapsed_seconds,
                "time_str": time_str,
                "experiment_name": experiment_name,
            }
        )

        print(f"\n--> Training for lr={lr} completed in {time_str}.")

    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY REPORT")
    print("=" * 80)
    print(f"Total target episodes per configuration: {args.num_episodes}")
    print(f"Seed used: {args.seed}")
    print(
        f"{'Learning Rate':<15} | {'Experiment':<26} | "
        f"{'Time Taken (Seconds)':<22} | {'Time Formatted':<15}"
    )
    print("-" * 91)
    for res in sensitivity_results:
        print(
            f"{res['lr']:<15.6f} | {res['experiment_name']:<26} | "
            f"{res['time_taken_sec']:<22.2f} | {res['time_str']:<15}"
        )
    print("=" * 80)

    print(f"Total time for all runs: {format_elapsed(time.time() - start_total_time)}")
    print("\nYou can now view the results by running:")
    print("tensorboard --logdir=results")
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
