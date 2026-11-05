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
        # optional: attn type indicating type of winodw to attend
        attention_type: str = "L",
    ) -> None:
        super().__init__()
        self.attention_type = attention_type  # 'S' for short, 'L' for long

        self.pre_attn_norm = RMSNorm(embed_dim)
        self.post_attn_norm = RMSNorm(embed_dim)
        self.pre_ffn_norm = RMSNorm(embed_dim)
        self.post_ffn_norm = RMSNorm(embed_dim)

        pos_embeddings = RoPE(
            seq_len=seq_len, head_dim=head_dim, rope_base_theta=rope_base_theta, dtype=dtype
        )

        window_size = seq_len // 4 if attention_type == "S" else -1
        self.gqa = GroupedQueryAttention(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            embed_dim=embed_dim,
            dtype=dtype,
            pos_embeddings=pos_embeddings,
            window_size=window_size,
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
        h = x + self.post_attn_norm(
            self.gqa(
                h,
                kv_cache=kv_cache,
                start_pos=start_pos,
                base_attn_mask=base_attn_mask,
                doc_ids=doc_ids,
            )
        )
        h_ffn = self.pre_ffn_norm(h)
        return h + self.post_ffn_norm(self.ffn(h_ffn))
