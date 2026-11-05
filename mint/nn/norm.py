import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, *, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.scale)
        if self.shift is not None:
            nn.init.zeros_(self.shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(x.dtype)
