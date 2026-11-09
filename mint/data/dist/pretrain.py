import queue
import random
import threading
import time
from collections.abc import Generator, Iterator

import pyarrow.parquet as pq
import torch

from mint.data.dataloader import DataloaderConfig, DistributedDataloader
from mint.tokenizer import Tokenizer
from mint.utils.device import Device
from mint.utils.logger import logger


class DistributedBOSBestfitPretrainDataloader(DistributedDataloader):
    def __init__(
        self,
        device: Device,
        config: DataloaderConfig,
        tokenizer: Tokenizer,
        min_shards_required: int = 2,
        max_shards_to_wait: int = -1,
        shard_check_interval: float = 5.0,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:

        super().__init__(
            device,
            config.data_dir,
            config.batch_size,
            config.seq_length,
            tokenizer,
            *args,
            **kwargs,
        )

        self.buffer_size = config.buffer_size
        self.tokenizer_batch_size = config.tokenizer_batch_size

        proc_info = device.process_info()
        self.rank = proc_info["dp_rank"]
        self.world_size = proc_info["dp_size"]
        self.is_main = proc_info["is_main"]

        # Dynamic shard discovery
        self.min_shards_required = min_shards_required
        self.max_shards_to_wait = max_shards_to_wait  # -1 means wait indefinitely
        self.shard_check_interval = shard_check_interval
        self.train_shards = []
        self.val_shards = []
        self._refresh_shard_lists()

        self.row_capacity = self.T + 1
        self.prefetch_depth = 2

        # GPU Destination tensors
        self.inputs = torch.empty((self.B, self.T), dtype=torch.long, device=device.device)
        self.targets = torch.empty((self.B, self.T), dtype=torch.long, device=device.device)
        self.loss_mask = torch.empty((self.B, self.T), dtype=torch.float32, device=device.device)
        self.doc_ids = torch.empty((self.B, self.T), dtype=torch.long, device=device.device)

        self.current_shard_idx = 0
        self.current_rg_idx = self.rank
        self.batches_consumed = 0
        self.last_shard_refresh = time.time()

    def _refresh_shard_lists(self) -> None:
        self.train_shards = sorted((self.data_dir / "train").glob("*.parquet"))
        self.val_shards = sorted((self.data_dir / "val").glob("*.parquet"))

    def _wait_for_shards(self, split: str, required_count: int) -> None:

        shards = self.train_shards if split == "train" else self.val_shards

        while len(shards) < required_count:
            # Check if we've hit the upper limit
            if self.max_shards_to_wait > 0 and len(shards) >= self.max_shards_to_wait:
                if self.is_main:
                    logger.info(
                        f"✓ Reached max shard limit ({self.max_shards_to_wait}). "
                        f"Proceeding with {len(shards)} shards."
                    )
                return

            if self.is_main:
                wait_msg = f"Waiting for shards... ({len(shards)}/{required_count} available"
                if self.max_shards_to_wait > 0:
                    wait_msg += f", max: {self.max_shards_to_wait}"
                wait_msg += ")"
                logger.info(wait_msg)

            time.sleep(self.shard_check_interval)
            self._refresh_shard_lists()
            shards = self.train_shards if split == "train" else self.val_shards

        if self.is_main:
            logger.info(f"✓ Sufficient shards available: {len(shards)}")

    def _should_refresh_shards(self) -> bool:
        current_time = time.time()
        if current_time - self.last_shard_refresh > self.shard_check_interval:
            self.last_shard_refresh = current_time
            return True
        return False

    def _document_batches(
        self, split: str, start_shard_idx: int = 0, start_rg_idx: int = -1
    ) -> Iterator[list]:
        self._wait_for_shards(split, self.min_shards_required)

        shards = self.train_shards if split == "train" else self.val_shards
        shard_cycle_start = start_shard_idx % len(shards) if shards else 0

        while True:
            # Periodically refresh shard list to discover new downloads
            if self._should_refresh_shards():
                old_count = len(shards)
                self._refresh_shard_lists()
                shards = self.train_shards if split == "train" else self.val_shards
                if len(shards) > old_count and self.is_main:
                    logger.info(
                        f"Discovered {len(shards) - old_count} new shard(s). Total: {len(shards)}"
                    )

            # If no shards available, check if we should wait or stop
            if not shards:
                # If max_shards_to_wait is set and we've reached it, stop waiting
                if self.max_shards_to_wait > 0:
                    if self.is_main:
                        logger.info(
                            f"No shards available and max limit ({self.max_shards_to_wait}) reached."
                        )
                    break

                if self.is_main:
                    logger.info("No shards available, waiting...")
                time.sleep(self.shard_check_interval)
                self._refresh_shard_lists()
                shards = self.train_shards if split == "train" else self.val_shards
                continue

            for shard_offset in range(len(shards)):
                shard_idx = (shard_cycle_start + shard_offset) % len(shards)
                shard_path = shards[shard_idx]

                # Check if shard still exists (in case of cleanup)
                if not shard_path.exists():
                    continue

                pf = pq.ParquetFile(shard_path)

                # Determine starting row group index
                rg_idx = start_rg_idx if shard_offset == 0 and start_rg_idx >= 0 else self.rank

                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()

                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        yield batch[i : i + self.tokenizer_batch_size]
                        self.batches_consumed += 1

                    rg_idx += self.world_size
                    self.current_rg_idx = rg_idx

                self.current_shard_idx = (shard_idx + 1) % len(shards)

    def _refill_buffer(self, doc_buffer: list, batches: Iterator) -> None:
        doc_batch = next(batches)
        token_lists = self.tokenizer.encode_to_list(
            doc_batch, add_bos=True, add_eos=False, padding=False
        )
        doc_buffer.extend(token_lists)

    def _cpu_batches(  # noqa: PLR0912, PLR0915
        self, split: str, resume_state: dict | None
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        use_cuda = self.device.is_cuda
        # Local to this generator so concurrent train/val loaders never share scratch space.
        row_buffer = torch.empty((self.B, self.row_capacity), dtype=torch.long)
        row_doc_ids = torch.empty((self.B, self.row_capacity), dtype=torch.long)

        # Restore state if resuming
        if resume_state and split == "train":
            start_shard_idx = resume_state.get("shard_idx", 0)
            start_rg_idx = resume_state.get("rg_idx", self.rank)
            batches_to_skip = resume_state.get("batches_consumed", 0)
            self.batches_consumed = batches_to_skip
        else:
            start_shard_idx = 0
            start_rg_idx = self.rank
            batches_to_skip = 0
            self.batches_consumed = 0

        batches = self._document_batches(split, start_shard_idx, start_rg_idx)
        doc_buffer = []

        # Skip already-consumed batches when resuming from checkpoint
        if batches_to_skip > 0:
            if self.is_main:
                logger.info(f"Skipping {batches_to_skip} already-consumed document batches...")
            for _ in range(batches_to_skip):
                try:
                    self._refill_buffer(doc_buffer, batches)
                    doc_buffer.clear()  # Discard these batches
                except StopIteration:
                    if self.is_main:
                        logger.warning(
                            f"Reached end of data while skipping batches. "
                            f"Skipped {_} of {batches_to_skip}"
                        )
                    break
            if self.is_main:
                logger.info(f"✓ Skipped {batches_to_skip} batches, resuming training...")

        while True:
            for row_idx in range(self.B):
                pos = 0
                global_doc_id = 0

                while pos < self.row_capacity:
                    while len(doc_buffer) < self.buffer_size:
                        self._refill_buffer(doc_buffer, batches)

                    remaining = self.row_capacity - pos

                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len

                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        doc_len = len(doc)
                        row_buffer[row_idx, pos : pos + doc_len] = torch.tensor(
                            doc, dtype=torch.long
                        )

                        row_doc_ids[row_idx, pos : pos + doc_len] = global_doc_id
                        global_doc_id += 1

                        pos += doc_len
                    else:
                        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(shortest_idx)
                        row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                            doc[:remaining], dtype=torch.long
                        )

                        row_doc_ids[row_idx, pos : pos + remaining] = global_doc_id
                        global_doc_id += 1

                        pos += remaining

            # Fresh pinned tensors per batch: PyTorch's pinned allocator will not reuse a block
            # until the async H2D copy that read it has finished, so queued batches are safe.
            cpu_inputs = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=use_cuda)
            cpu_targets = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=use_cuda)
            cpu_loss_mask = torch.empty((self.B, self.T), dtype=torch.float32, pin_memory=use_cuda)
            cpu_doc_ids = torch.empty((self.B, self.T), dtype=torch.long, pin_memory=use_cuda)

            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])

            input_doc_ids = row_doc_ids[:, :-1]
            target_doc_ids = row_doc_ids[:, 1:]

            cpu_loss_mask.copy_(input_doc_ids == target_doc_ids)
            cpu_doc_ids.copy_(input_doc_ids)

            yield cpu_inputs, cpu_targets, cpu_loss_mask, cpu_doc_ids

    def _prefetch(self, source: Iterator, depth: int) -> Generator:
        """Run `source` in a background thread so CPU batch building overlaps GPU compute."""
        buf: queue.Queue = queue.Queue(maxsize=depth)
        stop = threading.Event()
        done = object()

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    buf.put(item, timeout=0.1)
                except queue.Full:
                    continue
                return True
            return False

        def worker() -> None:
            try:
                for item in source:
                    if not put(item):
                        return
                put(done)
            except BaseException as e:  # noqa: BLE001
                put(e)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = buf.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()

    def batch_loader(
        self, split: str = "train", resume_state: dict | None = None
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        non_blocking = self.device.is_cuda
        for cpu_inputs, cpu_targets, cpu_loss_mask, cpu_doc_ids in self._prefetch(
            self._cpu_batches(split, resume_state), self.prefetch_depth
        ):
            self.inputs.copy_(cpu_inputs, non_blocking=non_blocking)
            self.targets.copy_(cpu_targets, non_blocking=non_blocking)
            self.doc_ids.copy_(cpu_doc_ids, non_blocking=non_blocking)
            self.loss_mask.copy_(cpu_loss_mask, non_blocking=non_blocking)

            yield self.inputs, self.targets, self.loss_mask, self.doc_ids

    def get_state(self) -> dict:
        return {
            "shard_idx": self.current_shard_idx,
            "rg_idx": self.current_rg_idx,
            "batches_consumed": self.batches_consumed,
        }

    def set_state(self, state: dict) -> None:
        self.current_shard_idx = state.get("shard_idx", 0)
        self.current_rg_idx = state.get("rg_idx", self.rank)
        self.batches_consumed = state.get("batches_consumed", 0)

    def sample(self, num_samples: int = 1):  # noqa: ANN201

        shards = self.train_shards if self.train_shards else self.val_shards
        if not shards:
            return []

        samples = []
        for _ in range(num_samples):
            shard_path = random.choice(shards)  # noqa: S311
            pf = pq.ParquetFile(shard_path)
            rg_idx = random.randint(0, pf.num_row_groups - 1)  # noqa: S311
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            text = random.choice(texts)  # noqa: S311

            tokens = self.tokenizer.encode([text], add_bos=True, add_eos=False, padding=False)[0]
            if len(tokens) > self.T:
                tokens = tokens[: self.T]

            input_tokens = tokens[:-1] if len(tokens) > 1 else tokens
            target_tokens = tokens[1:] if len(tokens) > 1 else tokens

            input_str = self.tokenizer.decode(input_tokens.unsqueeze(0))[0]
            target_str = self.tokenizer.decode(target_tokens.unsqueeze(0))[0]

            samples.append(
                {
                    "input_tokens": input_tokens,
                    "target_tokens": target_tokens,
                    "input_str": input_str,
                    "target_str": target_str,
                }
            )

        return samples


class DistributedVarlenPretrainDataloader(DistributedDataloader): ...
