from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import nn

from mint.config.base import Config
from mint.utils.device import Device


class Task(Enum):
    MC = "multiple_choice"
    SCHEMA = "schema"
    LM = "language_modeling"


@dataclass
class EvalConfig(Config):
    seq_length: int = 512
    max_new_tokens: int = 256


class Evaluator(ABC):
    def __init__(self, model: nn.Module, config: EvalConfig, device: Device) -> None:
        self.model: Any = model
        self.config = config
        self.device = device
        self.process_info = device.process_info()

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, Any]:  # noqa: ANN002, ANN003
        pass
