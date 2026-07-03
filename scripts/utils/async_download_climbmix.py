import argparse
import asyncio
import multiprocessing as mp
import ssl
import sys
import time
from multiprocessing import Lock, Value
from pathlib import Path

import aiofiles
import aiohttp

from mint.utils.logger import logger


try:
    from huggingface_hub import (
        hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
    )

    USE_HF_HUB = True
except ImportError:
    USE_HF_HUB = False  # type: ignore
    hf_hub_download = None  # type: ignore
    logger.info("Warning: huggingface_hub not installed. Using aiohttp.")
    logger.info("For better reliability, install with: pip install huggingface-hub")


class DownloadCoordinator:
    """Shared state between download and training processes."""

    def __init__(self, min_shards: int) -> None:
        self.min_shards = min_shards
        self.downloaded_count = Value("i", 0)
        self.total_shards = Value("i", 0)
        self.download_complete = Value("i", 0)
        self.lock = Lock()

    def increment_downloaded(self):  # noqa: ANN201
        with self.lock:
            self.downloaded_count.value += 1
            return self.downloaded_count.value

    def set_total_shards(self, total: int) -> None:
        with self.lock:
            self.total_shards.value = total

    def mark_complete(self) -> None:
        with self.lock:
            self.download_complete.value = 1

    def is_ready_for_training(self) -> bool:
        with self.lock:
            return self.downloaded_count.value >= self.min_shards

    def is_complete(self) -> bool:
        with self.lock:
            return self.download_complete.value == 1

    def get_status(self) -> dict[str, str]:
        with self.lock:
            return {
                "downloaded": self.downloaded_count.value,
                "total": self.total_shards.value,
                "complete": self.download_complete.value == 1,
            }


async def download_file_async(
    session: aiohttp.ClientSession,
    url: str,
    destination: Path,
    shard_idx: int,
    total_shards: int,
    *,
    show_progress: bool = True,
) -> bool:
    try:
        async with session.get(url) as response:
            if response.status != 200:
                logger.info(f"  ✗ Failed to download shard {shard_idx}: HTTP {response.status}")
                return False

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            async with aiofiles.open(destination, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total_size > 0:
                        percent = min(downloaded * 100.0 / total_size, 100.0)
                        sys.stdout.write(
                            f"\r[{shard_idx + 1}/{total_shards}] Progress: {percent:.1f}% "
                            f"({downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB)"
                        )
                        sys.stdout.flush()

            if show_progress:
                print()  # New line after progress
            return True

    except Exception as e:
        logger.info(f"  ✗ Error downloading shard {shard_idx}: {e}")
        return False


def download_file_hf_sync(repo_id: str, filename: str, destination: Path) -> bool:
    if hf_hub_download is None:
        return False

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(destination.parent),
            local_dir_use_symlinks=False,
        )  # type: ignore

        if Path(downloaded_path) != destination:
            Path(downloaded_path).rename(destination)
        return True  # noqa: TRY300
    except Exception as e:
        logger.info(f"  ✗ Failed to download {filename}: {e}")
        return False


async def download_shard(
    session: aiohttp.ClientSession,
    shard_idx: int,
    output_dir: Path,
    repo_id: str,
    base_url: str,
    shard_pattern: str,
    coordinator: DownloadCoordinator,
    total_shards: int,
    *,
    use_hf: bool = False,
) -> None:
    shard_name = shard_pattern.format(shard_idx)

    if shard_idx < total_shards - 1:
        destination = output_dir / "train" / shard_name
    else:
        destination = output_dir / "val" / shard_name

    if destination.exists():
        logger.info(f"[{shard_idx + 1}/{total_shards}] Skipping {shard_name} (already exists)")
        coordinator.increment_downloaded()
        return

    logger.info(f"[{shard_idx + 1}/{total_shards}] Downloading {shard_name}...")

    success = False
    if use_hf:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, download_file_hf_sync, repo_id, shard_name, destination
        )
    else:
        url = f"{base_url}/{shard_name}"
        success = await download_file_async(session, url, destination, shard_idx, total_shards)

    if success:
        logger.info(f"  ✓ Saved to {destination}")
        count = coordinator.increment_downloaded()

        if count == coordinator.min_shards:
            logger.info(
                f"✓ Minimum {coordinator.min_shards} shards downloaded - READY FOR TRAINING"
            )


async def download_all_shards(
    output_dir: str,
    num_train_shards: int,
    repo_id: str,
    base_url: str,
    shard_pattern: str,
    coordinator: DownloadCoordinator,
    max_concurrent: int = 3,
) -> None:

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_shards = num_train_shards + 1  # +1 for validation shard
    coordinator.set_total_shards(total_shards)

    logger.info(f"Downloading ClimbMix dataset to: {output_path}")
    logger.info(f"Training shards: {num_train_shards}")
    logger.info("Validation shards: 1")
    logger.info(f"Max concurrent downloads: {max_concurrent}")
    logger.info(f"Minimum shards before training: {coordinator.min_shards}")
    logger.info(f"Using: {'huggingface_hub' if USE_HF_HUB else 'aiohttp'}")

    # Create SSL context for aiohttp
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context, limit=max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(shard_idx: int) -> None:
            async with semaphore:
                await download_shard(
                    session,
                    shard_idx,
                    output_path,
                    repo_id,
                    base_url,
                    shard_pattern,
                    coordinator,
                    total_shards,
                    use_hf=USE_HF_HUB,
                )

        # Download all shards concurrently (with limit)
        tasks = [download_with_semaphore(i) for i in range(total_shards)]
        await asyncio.gather(*tasks)

    coordinator.mark_complete()

    logger.info("Download complete!")
    logger.info(f"Training data: {train_dir}")
    logger.info(f"Validation data: {val_dir}")


def download_process(
    output_dir: str,
    num_train_shards: int,
    repo_id: str,
    base_url: str,
    shard_pattern: str,
    coordinator: DownloadCoordinator,
    max_concurrent: int,
) -> None:
    """Entry point for download process."""
    asyncio.run(
        download_all_shards(
            output_dir,
            num_train_shards,
            repo_id,
            base_url,
            shard_pattern,
            coordinator,
            max_concurrent,
        )
    )


def wait_for_training_ready(coordinator: DownloadCoordinator, check_interval: float = 2.0) -> None:
    """Wait until minimum shards are available for training."""
    logger.info(f"Waiting for minimum {coordinator.min_shards} shards to be downloaded...")

    while not coordinator.is_ready_for_training():
        status = coordinator.get_status()
        logger.info(f"  Downloaded: {status['downloaded']}/{coordinator.min_shards} shards")
        time.sleep(check_interval)

    logger.info(f"✓ Training ready! {coordinator.min_shards} shards available.")


def start_download_and_training(
    output_dir: str = "data/climbmix",
    num_train_shards: int = 10,
    min_shards_before_training: int = 3,
    repo_id: str = "karpathy/climbmix-400b-shuffle",
    base_url: str = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
    shard_pattern: str = "shard_{:05d}.parquet",
    max_concurrent: int = 3,
    *,
    start_training: bool = False,
) -> tuple[DownloadCoordinator, mp.Process]:

    coordinator = DownloadCoordinator(min_shards_before_training)

    download_proc = mp.Process(
        target=download_process,
        args=(
            output_dir,
            num_train_shards,
            repo_id,
            base_url,
            shard_pattern,
            coordinator,
            max_concurrent,
        ),
    )
    download_proc.start()

    if start_training:
        wait_for_training_ready(coordinator)
        logger.info("You can now start training!")
        logger.info("Download will continue in the background.")

    return coordinator, download_proc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async download ClimbMix dataset with training coordination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/climbmix",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=10,
        help="Number of shards to download for training",
    )
    parser.add_argument(
        "--min-shards-before-training",
        type=int,
        default=3,
        help="Minimum shards before training can start",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="karpathy/climbmix-400b-shuffle",
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        help="Base URL for the dataset",
    )
    parser.add_argument(
        "--shard-pattern",
        type=str,
        default="shard_{:05d}.parquet",
        help="Pattern for shard filenames",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Maximum concurrent downloads"
    )
    parser.add_argument(
        "--wait-for-training",
        action="store_true",
        help="Wait until minimum shards are ready before exiting",
    )

    args = parser.parse_args()

    _coordinator, download_proc = start_download_and_training(
        output_dir=args.output_dir,
        num_train_shards=args.num_train_shards,
        min_shards_before_training=args.min_shards_before_training,
        repo_id=args.repo_id,
        base_url=args.base_url,
        shard_pattern=args.shard_pattern,
        max_concurrent=args.max_concurrent,
        start_training=args.wait_for_training,
    )

    download_proc.join()


if __name__ == "__main__":
    main()
