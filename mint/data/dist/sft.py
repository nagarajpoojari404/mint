import random
from collections.abc import Generator

import torch

from mint.data.dataloader import DataloaderConfig, DistributedDataloader
from mint.data.datasets.base import SFTDataset
from mint.tokenizer import Tokenizer
from mint.utils.device import Device


class DistributedSFTDataloader(DistributedDataloader):
    def __init__(
        self,
        device: Device,
        config: DataloaderConfig,
        tokenizer: Tokenizer,
        datasets: list[SFTDataset],
        shuffle: bool = True,  # noqa: FBT001, FBT002
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

        self.datasets = datasets
        self.max_tokens = config.seq_length
        self.shuffle = shuffle

        proc_info = device.process_info()
        self.rank = proc_info["rank"]
        self.world_size = proc_info["world_size"]

        self.dataset_indices = self._build_dataset_indices()
        self.rank_indices = self._partition_for_rank()

        use_cuda = device.is_cuda
        self.cpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, pin_memory=use_cuda)
        self.gpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, device=device.device)

        self.cpu_inputs = self.cpu_buffer[: self.B * self.T].view(self.B, self.T)
        self.cpu_targets = self.cpu_buffer[self.B * self.T :].view(self.B, self.T)
        self.cpu_mask = torch.zeros(self.B, self.T, dtype=torch.long, pin_memory=use_cuda)

        self.inputs = self.gpu_buffer[: self.B * self.T].view(self.B, self.T)
        self.targets = self.gpu_buffer[self.B * self.T :].view(self.B, self.T)
        self.mask = torch.zeros(self.B, self.T, dtype=torch.long, device=device.device)

    def _build_dataset_indices(self) -> list:

        indices = []
        for ds_idx, ds in enumerate(self.datasets):
            indices.extend([(ds_idx, i) for i in range(len(ds))])
        if self.shuffle:
            random.Random(42).shuffle(indices)  # noqa: S311
        return indices

    def _partition_for_rank(self) -> list:
        return [
            idx for i, idx in enumerate(self.dataset_indices) if i % self.world_size == self.rank
        ]

    def batch_loader(
        self,
        split: str = "train",  # noqa: ARG002
        resume_state: dict | None = None,  # noqa: ARG002
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        idx = 0
        total = len(self.rank_indices)

        while True:
            for batch_idx in range(self.B):
                ds_idx, example_idx = self.rank_indices[idx % total]
                idx += 1

                conversation = self.datasets[ds_idx][example_idx]
                ids, mask = self.tokenizer.render_conversation(conversation, self.max_tokens)

                seq_len = len(ids)
                if seq_len <= self.T:
                    ids = ids + [self.tokenizer.pad_token] * (self.T + 1 - seq_len)
                    mask = mask + [0] * (self.T + 1 - seq_len)
                else:
                    ids = ids[: self.T + 1]
                    mask = mask[: self.T + 1]

                self.cpu_inputs[batch_idx] = torch.tensor(ids[:-1], dtype=torch.long)
                self.cpu_targets[batch_idx] = torch.tensor(ids[1:], dtype=torch.long)
                self.cpu_mask[batch_idx] = torch.tensor(mask[1:], dtype=torch.long)

            self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=self.device.is_cuda)
            self.mask.copy_(self.cpu_mask, non_blocking=self.device.is_cuda)

            # doc_ids irrelevent here
            yield self.inputs, self.targets, self.mask, None

    def get_state(self) -> dict:
        return NotImplementedError()  # TODO: implement resume state for SFT

    def set_state(self, state: dict) -> None:  # noqa: ARG002
        return NotImplementedError()

    def sample(self, num_samples: int = 1) -> list[dict]:
        samples = []

        for _ in range(num_samples):
            ds_idx = random.randint(0, len(self.datasets) - 1)  # noqa: S311
            example_idx = random.randint(0, len(self.datasets[ds_idx]) - 1)  # noqa: S311

            conversation = self.datasets[ds_idx][example_idx]
            ids, mask = self.tokenizer.render_conversation(conversation, self.max_tokens)

            if len(ids) > self.T:
                ids = ids[: self.T]
                mask = mask[: self.T]

            samples.append(
                {
                    "tokens": torch.tensor(ids, dtype=torch.long),
                    "mask": torch.tensor(mask, dtype=torch.long),
                    "conversation": conversation,
                }
            )

        return samples
