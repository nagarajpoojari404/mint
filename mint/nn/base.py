from dataclasses import dataclass

import torch
from torch import nn

from mint.config.base import Config


@dataclass
class ModelConfig(Config): ...


class LogitsWrapper(nn.Module):
    """Wraps any model so that forward() always returns a plain logits tensor.

    Native mint models return a raw tensor; HuggingFace models return a
    ModelOutput whose ``.logits`` attribute holds the tensor. This wrapper
    normalises both cases so callers never need an inline ``hasattr`` check.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        output = self._model(*args, **kwargs)
        return output.logits if hasattr(output, "logits") else output

    def generate(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Delegate to the underlying model's generate for KV-cached inference."""
        return self._model.generate(*args, **kwargs)
