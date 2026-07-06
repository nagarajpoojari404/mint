from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mint.config.base import Config
from mint.data.sampler import Sampler
from mint.tokenizer import Tokenizer
from mint.utils.device import Device


@dataclass
class DataloaderConfig(Config):
    data_dir: str = "data/climbmix"
    batch_size: int = 4
    buffer_size: int = 1000
    tokenizer_batch_size: int = 128
    seq_length: int = 512


class DistributedDataloader(Sampler, ABC):
    def __init__(
        self,
        device: Device,
        data_dir: str,
        batch_size: int,
        seq_len: int,
        tokenizer: Tokenizer,
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        super().__init__()
        self.device = device
        self.data_dir = Path(data_dir)
        self.B = batch_size
        self.T = seq_len
        self.tokenizer = tokenizer

    @abstractmethod
    def batch_loader(
        self, split: str = "train", resume_state: dict | None = None
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict: ...

    @abstractmethod
    def set_state(self, state: dict) -> None: ...
