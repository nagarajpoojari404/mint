import os
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from kernels_ import (
    FlashAttentionMQAKernel,  # pyright: ignore[reportMissingImports]
)
from mint.kvcache.base import KVCache


class FlashAttention(nn.Module):
    def __init__(self, *, use_custom_fa: bool = True) -> None:
        super().__init__()
        self.use_custom_fa = use_custom_fa
        self._custom_fa: Any | None = self._load_custom_flash_attention() if use_custom_fa else None
        self._fa3: Any | None = self._load_flash_attention_3()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        kv_cache: KVCache | None = None,
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache is None:
            return self._attention(
                q,
                k,
                v,
                causal=causal,
                window_size=window_size,
                base_attn_mask=base_attn_mask,
                doc_ids=doc_ids,
            )

        return self._attention_with_kvcache(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            causal=causal,
            window_size=window_size,
            base_attn_mask=base_attn_mask,
            doc_ids=doc_ids,
        )

    def _load_custom_flash_attention(self) -> Any:  # noqa: ANN401
        if not torch.cuda.is_available():
            return None
        try:
            return FlashAttentionMQAKernel
        except Exception:
            return None

    def _load_flash_attention_3(self):  # noqa: ANN202
        if not torch.cuda.is_available():
            return None
        try:
            major, _ = torch.cuda.get_device_capability()
            if major != 9:
                return None
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            from kernels import get_kernel  # type: ignore[import-not-found] # noqa: PLC0415

            return get_kernel("varunneal/flash-attention-3").flash_attn_interface
        except Exception:
            return None

    def _use_custom_fa(self, q: torch.Tensor) -> bool:
        return (
            self._custom_fa is not None and q.is_cuda and q.dtype in [torch.float16, torch.bfloat16]
        )

    def _use_fa3(self, q: torch.Tensor) -> bool:
        return self._fa3 is not None and q.is_cuda and q.dtype == torch.bfloat16

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        window_size: tuple[int, int],
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if doc_ids is not None:
            try:
                from flash_attn import flash_attn_varlen_func  # type: ignore # noqa: PLC0415

                B, T, H, D = q.shape
                _, _, H_k, _ = k.shape

                # FlashAttention varlen expects flattened sequences: (total_tokens, H, D)
                q_flat = q.reshape(-1, H, D)
                k_flat = k.reshape(-1, H_k, D)
                v_flat = v.reshape(-1, H_k, D)

                # Calculate consecutive document lengths across the flattened batch
                flat_doc_ids = doc_ids.reshape(-1)
                _, counts = torch.unique_consecutive(flat_doc_ids, return_counts=True)

                # Compute cumulative sequence lengths (must be int32 for FA kernels)
                cu_seqlens = torch.cat(
                    [
                        torch.tensor([0], dtype=torch.int32, device=q.device),
                        torch.cumsum(counts, dim=0, dtype=torch.int32),
                    ]
                )
                max_seqlen = int(counts.max().item())

                out_flat = flash_attn_varlen_func(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=causal,
                    window_size=window_size,
                )
                return out_flat.reshape(B, T, H, D)
            except ImportError:
                # Fall through to SDPA if flash_attn is not installed
                pass

        if self._use_custom_fa(q) and window_size == (-1, -1) and q.shape[2] == k.shape[2]:
            custom_fa = self._custom_fa
            assert custom_fa is not None
            head_dim = q.shape[-1]
            tau = 1.0 / (head_dim**0.5)
            # kernel expects [B, H, T, D]; make contiguous so strides are standard
            q_ = q.transpose(1, 2).contiguous()
            k_ = k.transpose(1, 2).contiguous()
            v_ = v.transpose(1, 2).contiguous()
            out = custom_fa.apply(q_, k_, v_, causal, tau)  # [B, H, T, D]
            return out.transpose(1, 2)  # [B, T, H, D]

        if doc_ids is None and self._use_fa3(q):
            fa3 = self._fa3
            assert fa3 is not None
            return fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

        # 4. Fall back to standard SDPA layout (supports base_attn_mask and doc_ids fallbacks)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

        out = self._sdpa_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            causal=causal,
            window_size=window_size,
            enable_gqa=enable_gqa,
            base_attn_mask=base_attn_mask,
            doc_ids=doc_ids,
        )
        return out.transpose(1, 2)

    def _attention_with_kvcache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache,
        *,
        causal: bool,
        window_size: tuple[int, int],
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k_cache, v_cache, cache_seqlens = kv_cache.get_cache_tensors()

        if doc_ids is None and self._use_fa3(q):
            fa3 = self._fa3
            assert fa3 is not None
            return fa3.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                causal=causal,
                window_size=window_size,
            )

        k_full, v_full = kv_cache.update(k, v, start_pos=int(cache_seqlens[0].item()))

        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

        out = self._sdpa_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            causal=causal,
            window_size=window_size,
            enable_gqa=enable_gqa,
            base_attn_mask=base_attn_mask,
            doc_ids=doc_ids,
        )
        return out.transpose(1, 2)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        window_size: tuple[int, int],
        enable_gqa: bool,
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = q.size(2)
        k_len = k.size(2)

        # Build document-boundary attention mask from doc_ids if available
        mask = None
        if doc_ids is not None:
            # Shape: (B, 1, T, 1) == (B, 1, 1, T) -> Broadcasts to (B, 1, T, T)
            mask = doc_ids.unsqueeze(1).unsqueeze(-1) == doc_ids.unsqueeze(1).unsqueeze(-2)

        if base_attn_mask is not None:
            mask = base_attn_mask if mask is None else (mask & base_attn_mask)

        if causal:
            device = q.device
            row_idx = (k_len - q_len) + torch.arange(q_len, device=device).unsqueeze(1)
            col_idx = torch.arange(k_len, device=device).unsqueeze(0)
            causal_mask = col_idx <= row_idx

            left_window = window_size[0]
            if 0 <= left_window < k_len:
                causal_mask = causal_mask & ((row_idx - col_idx) <= left_window)

            mask = causal_mask if mask is None else (mask & causal_mask)

        # Handle trivial query sequence layouts optimization
        if q_len == 1 and not causal:
            left_window = window_size[0]
            if 0 <= left_window < k_len:
                start = max(0, k_len - (left_window + 1))
                k = k[:, :, start:, :]
                v = v[:, :, start:, :]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=(causal and mask is None),
            enable_gqa=enable_gqa,
        )
