from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mint.config.base import Config


@dataclass
class ModelConfig(Config): ...


class LogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model: Any = model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        output = self._model(*args, **kwargs)
        return output.logits if hasattr(output, "logits") else output

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return self._model.generate(*args, **kwargs)
