import torch
from torch import nn

from mint.kvcache.base import KVCache
from mint.nn.flashattn import FlashAttention
from mint.nn.norm import RMSNorm
from mint.nn.pos import PosEmbeddings


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        pos_embeddings: PosEmbeddings,
        dtype: torch.dtype,
        window_size: int = -1,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.pos_embeddings = pos_embeddings

        self.attention = FlashAttention(use_custom_fa=True)

        self.wq = nn.Linear(embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.wv = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.wo = nn.Linear(self.n_heads * self.head_dim, embed_dim, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(  # noqa: ANN201
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
        # optional: we will apply causal on top of a base attn_mask, passed in special cases
        # like [PAD] tokens in SFT or blocking cross sentence attn in best fit pretrain
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ):
        q = self.wq(x)
        bsz, seqlen = q.shape[0], q.shape[1]
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, attn_bias = self.pos_embeddings(q, k, start_pos=start_pos)

        assert attn_bias is None, "attention bias is not supported yet"

        window_size = (self.window_size, -1) if self.window_size > 0 else (-1, -1)
        out = self.attention(
            q,
            k,
            v,
            causal=True,
            window_size=window_size,
            kv_cache=kv_cache,
            base_attn_mask=base_attn_mask,
            doc_ids=doc_ids,
        )

        out = out.reshape(bsz, seqlen, -1)
        return self.wo(out)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        embed_dim: int,
        pos_embeddings: PosEmbeddings,
        dtype: torch.dtype,
        window_size: int = -1,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.pos_embeddings = pos_embeddings

        self.attention = FlashAttention(use_custom_fa=True)

        self.wq = nn.Linear(embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.wv = nn.Linear(embed_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.wo = nn.Linear(self.n_heads * self.head_dim, embed_dim, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(  # noqa: ANN201
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
        # optional: we will apply causal on top of a base attn_mask, passed in special cases
        # like [PAD] tokens in SFT or blocking cross sentence attn in best fit pretrain
        base_attn_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ):
        q = self.wq(x)
        bsz, seqlen = q.shape[0], q.shape[1]
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, attn_bias = self.pos_embeddings(q, k, start_pos=start_pos)

        assert attn_bias is None, "attention bias is not supported yet"

        window_size = (self.window_size, -1) if self.window_size > 0 else (-1, -1)
        out = self.attention(
            q,
            k,
            v,
            causal=True,
            window_size=window_size,
            kv_cache=kv_cache,
            base_attn_mask=base_attn_mask,
            doc_ids=doc_ids,
        )

        out = out.reshape(bsz, seqlen, -1)
        return self.wo(out)
