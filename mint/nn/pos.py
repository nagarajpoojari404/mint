import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class PosEmbeddings(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...


class SinusoidalPosEmbeddings(PosEmbeddings):
    def __init__(self, seq_len: int, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim

        pe = torch.zeros(seq_len, head_dim)

        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, head_dim, 2, dtype=torch.float32) * -(math.log(base) / head_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe[None, :, None, :], persistent=False)
        self.pe: torch.Tensor

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pe_slice = self.pe[:, start_pos : start_pos + q.shape[1]].to(q.dtype)
        return q + pe_slice, k + pe_slice, None


class RoPE(PosEmbeddings):
    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        rope_base_theta: int = 100000,
        dtype: torch.dtype = torch.float32,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.base = rope_base_theta
        self.dtype = dtype
        self.scale_factor = scale_factor

        cos, sin = self._precompute_rotary_embeddings()
        # buffers follow the model across devices but are not trainable
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cos: torch.Tensor
        self.sin: torch.Tensor

    def reset_parameters(self) -> None:
        cos, sin = self._precompute_rotary_embeddings()
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        channel_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (channel_range / self.head_dim))

        t = torch.arange(self.seq_len, dtype=torch.float32) / self.scale_factor
        freqs = torch.outer(t, inv_freq)

        cos, sin = freqs.cos().to(self.dtype), freqs.sin().to(self.dtype)
        return cos[None, :, None, :], sin[None, :, None, :]

    @staticmethod
    def _apply_rotary_embs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        end_pos = start_pos + q.shape[1]
        cos = self.cos[:, start_pos:end_pos].to(q.dtype)
        sin = self.sin[:, start_pos:end_pos].to(q.dtype)
        return self._apply_rotary_embs(q, cos, sin), self._apply_rotary_embs(k, cos, sin), None


class LinearScaledRoPE(RoPE):
    """RoPE with positions interpolated by ``1 / scale_factor`` to extend the context window."""

    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        scale_factor: float = 4.0,
        rope_base_theta: int = 100000,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(seq_len, head_dim, rope_base_theta, dtype, scale_factor=scale_factor)


class NoPE(PosEmbeddings):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return q, k, None


class ALiBiPositionalBias(PosEmbeddings):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        slopes = torch.tensor(self._get_slopes(n_heads))
        self.register_buffer("slopes", slopes[None, :, None, None], persistent=False)
        self.slopes: torch.Tensor

    def _get_slopes(self, n_heads: int) -> list[float]:
        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self._get_slopes(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        seq_len_q = q.shape[1]
        total_kv_len = start_pos + seq_len_q

        q_idx = torch.arange(start_pos, total_kv_len, device=q.device)[:, None]
        k_idx = torch.arange(total_kv_len, device=k.device)[None, :]

        # keys at or before the query give a zero/negative distance, i.e. a penalty
        relative_position = k_idx - q_idx
        alibi_bias = self.slopes * relative_position[None, None, :, :]
        return q, k, alibi_bias.to(q.dtype)
