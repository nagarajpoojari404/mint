import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from mint.utils.logger import logger
from scripts.utils.async_download_climbmix import (
    DownloadCoordinator,
    download_process,
    wait_for_training_ready,
)


def training_process(
    train_script: str,
    train_args: list,
    coordinator: DownloadCoordinator,
    min_shards: int,
    use_torchrun: bool = False,
    torchrun_args: list | None = None,
) -> None:
    logger.info(f"Waiting for minimum {min_shards} shards...")
    wait_for_training_ready(coordinator, check_interval=2.0)

    logger.info("STARTING TRAINING")
    logger.info("Download continues in background")

    # Build command
    if use_torchrun:
        cmd = ["torchrun"] + (torchrun_args or []) + [train_script] + train_args
    else:
        cmd = [sys.executable, train_script] + train_args

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    logger.info("TRAINING COMPLETED")


def main():
    parser = argparse.ArgumentParser(
        description="Train with streaming dataset download",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Download arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/climbmix",
        help="Directory to save/load dataset",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=10,
        help="Total number of training shards to download",
    )
    parser.add_argument(
        "--min-shards",
        type=int,
        default=3,
        help="Minimum shards before starting training",
    )
    parser.add_argument(
        "--max-concurrent-downloads",
        type=int,
        default=3,
        help="Maximum concurrent downloads",
    )
    parser.add_argument(
        "--max-shards-to-wait",
        type=int,
        default=-1,
        help="Maximum shards to wait for in dataloader (-1 = wait indefinitely)",
    )

    # Training arguments
    parser.add_argument("--train-script", type=str, required=True, help="Path to training script")
    parser.add_argument(
        "--use-torchrun", action="store_true", help="Use torchrun instead of python"
    )
    parser.add_argument(
        "--torchrun-args",
        type=str,
        default="",
        help="Arguments for torchrun (e.g., '--nproc_per_node=2')",
    )
    parser.add_argument("--train-args", type=str, default="", help="Arguments for training script")

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, assume shards exist",
    )

    args = parser.parse_args()
    torchrun_args = args.torchrun_args.split() if args.torchrun_args else []
    train_args = args.train_args.split() if args.train_args else []

    # Add min-shards argument to training script
    train_args.extend(["--min-shards", str(args.min_shards)])
    if args.max_shards_to_wait > 0:
        train_args.extend(["--max-shards-to-wait", str(args.max_shards_to_wait)])

    coordinator = DownloadCoordinator(args.min_shards)
    processes = []

    if not args.skip_download:
        logger.info("Starting download process...")
        download_proc = mp.Process(
            target=download_process,
            args=(
                args.data_dir,
                args.num_train_shards,
                "karpathy/climbmix-400b-shuffle",
                "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
                "shard_{:05d}.parquet",
                coordinator,
                args.max_concurrent_downloads,
            ),
        )
        download_proc.start()
        processes.append(download_proc)
        logger.info(f"✓ Download started (PID: {download_proc.pid})")
    else:
        coordinator.set_total_shards(args.num_train_shards + 1)
        for _ in range(args.min_shards):
            coordinator.increment_downloaded()

    logger.info("Starting training process...")
    train_proc = mp.Process(
        target=training_process,
        args=(
            args.train_script,
            train_args,
            coordinator,
            args.min_shards,
            args.use_torchrun,
            torchrun_args,
        ),
    )
    train_proc.start()
    processes.append(train_proc)
    logger.info(f"Training started (PID: {train_proc.pid})")

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        logger.info("Stopping all processes...")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
        sys.exit(1)

    logger.info("\n✓ ALL PROCESSES COMPLETED")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
