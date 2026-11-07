import torch
from torch import nn

from mint.kvcache.base import KVCache
from mint.nn.attn import GroupedQueryAttention
from mint.nn.ffn import GLU
from mint.nn.norm import RMSNorm
from mint.nn.pos import RoPE


class GQAGluBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        hidden_dim: int,
        seq_len: int,
        rope_base_theta: int,
        dtype: torch.dtype,
        attention_type: str = "L",
        sliding_window: int | None = None,
        use_post_norms: bool = True,
        use_qk_norm: bool = False,
        attn_logit_softcapping: float | None = None,
        query_pre_attn_scalar: float | None = None,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type  # 'S' for short, 'L' for long
        self.use_post_norms = use_post_norms

        self.pre_attn_norm = RMSNorm(embed_dim)
        self.post_attn_norm = RMSNorm(embed_dim) if use_post_norms else None
        self.pre_ffn_norm = RMSNorm(embed_dim)
        self.post_ffn_norm = RMSNorm(embed_dim) if use_post_norms else None

        pos_embeddings = RoPE(
            seq_len=seq_len, head_dim=head_dim, rope_base_theta=rope_base_theta, dtype=dtype
        )

        if attention_type == "S":
            window_size = sliding_window if sliding_window is not None else seq_len // 4
        else:
            window_size = -1

        self.gqa = GroupedQueryAttention(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            embed_dim=embed_dim,
            dtype=dtype,
            pos_embeddings=pos_embeddings,
            window_size=window_size,
            attn_logit_softcapping=attn_logit_softcapping,
            query_pre_attn_scalar=query_pre_attn_scalar,
            use_qk_norm=use_qk_norm,
        )
        self.ffn = GLU(embed_dim=embed_dim, hidden_dim=hidden_dim, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
        # optional: we will apply causal on top of a base attn_mask, passed in special cases
        # like [PAD] tokens in SFT or blocking cross sentence attn in best fit pretrain
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.pre_attn_norm(x)
        attn_out = self.gqa(
            h,
            kv_cache=kv_cache,
            start_pos=start_pos,
            base_attn_mask=base_attn_mask,
            doc_ids=doc_ids,
        )
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)
        h = x + attn_out

        h_ffn = self.pre_ffn_norm(h)
        ffn_out = self.ffn(h_ffn)
        if self.post_ffn_norm is not None:
            ffn_out = self.post_ffn_norm(ffn_out)
        return h + ffn_out


class GemmaBlock(GQAGluBlock):
    """Alias for GQAGluBlock representing standard Gemma family transformer block."""


class Gemma1Block(GQAGluBlock):
    """Gemma 1 block with pre-attention and pre-FFN RMSNorm (no post-norms)."""

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        hidden_dim: int,
        seq_len: int,
        rope_base_theta: int = 10000,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            rope_base_theta=rope_base_theta,
            dtype=dtype,
            use_post_norms=False,
            use_qk_norm=False,
            **kwargs,
        )


class Gemma2Block(GQAGluBlock):
    """Gemma 2 block with dual RMSNorm (pre and post norms) and softcapping."""

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        hidden_dim: int,
        seq_len: int,
        rope_base_theta: int = 10000,
        dtype: torch.dtype = torch.float32,
        sliding_window: int | None = 4096,
        attn_logit_softcapping: float | None = 50.0,
        **kwargs,
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            rope_base_theta=rope_base_theta,
            dtype=dtype,
            sliding_window=sliding_window,
            use_post_norms=True,
            use_qk_norm=False,
            attn_logit_softcapping=attn_logit_softcapping,
            **kwargs,
        )


class Gemma3Block(GQAGluBlock):
    """Gemma 3 block with dual RMSNorm and Q/K RMSNorm."""

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        hidden_dim: int,
        seq_len: int,
        rope_base_theta: int = 1000000,
        dtype: torch.dtype = torch.float32,
        sliding_window: int | None = 4096,
        **kwargs,
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            rope_base_theta=rope_base_theta,
            dtype=dtype,
            sliding_window=sliding_window,
            use_post_norms=True,
            use_qk_norm=True,
            **kwargs,
        )
