from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class Adapter(ABC):
    @classmethod
    @abstractmethod
    def apply(cls, model: nn.Module, **kwargs: Any) -> None:  # noqa: ANN401
        pass
