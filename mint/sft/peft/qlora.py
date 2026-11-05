
import torch
from torch import nn

from .base import Adapter


class QLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class QLoRA(Adapter):
    @classmethod
    def apply(
        cls,
        model: nn.Module,
        target_modules: list[str] | None = None,
        r: int = 8,
        alpha: int = 16,
        **kwargs,  # noqa: ANN003
    ) -> None: ...
