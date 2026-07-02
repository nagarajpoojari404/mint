import math

import torch
from torch import nn

from .base import Adapter


class LoRALinear(nn.Module):
    def __init__(
        self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / r

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):  # noqa: ANN201
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


class LoRA(Adapter):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def apply(
        cls,
        model: nn.Module,
        target_modules: list[str] | None = None,
        r: int = 8,
        alpha: int = 16,
    ) -> None:
        if target_modules is None:
            target_modules = ["wq", "wv"]

        for name, module in model.named_modules():
            # Target specific linear projections
            if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
                # Parse the tree to safely replace the module in-place
                parent_path = name.rsplit(".", 1)
                parent = model.get_submodule(parent_path[0]) if len(parent_path) > 1 else model
                setattr(parent, parent_path[-1], LoRALinear(module, r, alpha))

        # Guarantee only LoRA parameters are tracking gradients
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
