from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mint.kvcache.base import KVCache
from mint.nn.base import ModelConfig
from mint.nn.blocks import GQAGluBlock
from mint.nn.norm import RMSNorm
from mint.optim.muon_adamw import MuonAdamW, MuonAdamWConfig
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.logger import logger


@dataclass
class GemmaConfig(ModelConfig):
    embed_dim: int
    hidden_dim: int
    seq_length: int
    vocab_size: int
    n_layers: int
    dtype: torch.dtype = torch.float32
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 128
    rope_base_theta: int = 100000
    tie_weights: bool = True


class Gemma(nn.Module):
    def __init__(self, config: GemmaConfig, *, gradient_checkpointing: bool) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim, dtype=config.dtype)

        # alternate short-range ('S') and long-range ('L') attention
        self.blocks = nn.ModuleList(
            GQAGluBlock(
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                seq_len=config.seq_length,
                rope_base_theta=config.rope_base_theta,
                dtype=config.dtype,
                attention_type="S" if i % 2 == 0 else "L",
            )
            for i in range(config.n_layers)
        )

        self.norm = RMSNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False, dtype=config.dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # `Module.to_empty` breaks weight tying, so it is re-established after every re-init
        if self.config.tie_weights:
            self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        doc_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)

        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            kwargs = {
                "kv_cache": kv_cache,
                "start_pos": start_pos,
                "base_attn_mask": attn_mask,
                "doc_ids": doc_ids,
            }
            if self.gradient_checkpointing and self.training and kv_cache is None:
                h = checkpoint(block, h, use_reentrant=False, **kwargs)
            else:
                h = block(h, **kwargs)

        return self.output(self.norm(h))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        kv_caches: list[KVCache] | None = None,
    ) -> torch.Tensor:
        """Sample ``max_new_tokens`` tokens; ``temperature=0`` decodes greedily."""
        self.eval()

        for step in range(max_new_tokens):
            if kv_caches is None:
                logits = self(input_ids[:, -self.config.seq_length :])
            elif step == 0:
                logits = self(input_ids, kv_caches=kv_caches)
            else:
                logits = self(
                    input_ids[:, -1:], kv_caches=kv_caches, start_pos=input_ids.shape[1] - 1
                )
            logits = logits[:, -1, :]

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_to_remove = cumulative_probs > top_p
                    sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
                    sorted_to_remove[..., 0] = False

                    to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
                    logits[to_remove] = -float("inf")

                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params


def configure_optimizer(
    model: nn.Module,
    model_cfg: GemmaConfig,
    sched_cfg: SchedulerConfig,
    optim_cfg: MuonAdamWConfig,
) -> MuonAdamW:
    params: dict[str, list[nn.Parameter]] = {
        "lm_head": [],
        "embedding": [],
        "scalar": [],
        "matrix": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tok_embeddings" in name:
            params["embedding"].append(param)
        elif "output" in name:
            params["lm_head"].append(param)
        elif param.ndim == 2:
            params["matrix"].append(param)
        else:
            params["scalar"].append(param)

    # AdamW LRs are scaled by 1/sqrt(dmodel), tuned for a 768 dim model
    lr_scale = (model_cfg.embed_dim / 768) ** -0.5
    logger.info(f"Scaling AdamW LRs by {lr_scale:.6f}")

    # kind -> (lr, beta2, weight_decay); matrix params are routed to AdamW for stability
    hyperparams = {
        "lm_head": (
            sched_cfg.unembedding_lr * lr_scale,
            optim_cfg.adamw_beta2_lm_head,
            sched_cfg.weight_decay_lm_head,
        ),
        "embedding": (
            sched_cfg.embedding_lr * lr_scale,
            optim_cfg.adamw_beta2_embedding,
            sched_cfg.weight_decay_embedding,
        ),
        "scalar": (
            sched_cfg.scalar_lr * sched_cfg.scalar_lr_multiplier,
            optim_cfg.adamw_beta2_scalar,
            sched_cfg.weight_decay_scalar,
        ),
        "matrix": (
            sched_cfg.matrix_lr,
            optim_cfg.adamw_beta2_scalar,
            sched_cfg.weight_decay,
        ),
    }

    param_groups = []
    for kind, group_params in params.items():
        if not group_params:
            continue
        lr, beta2, weight_decay = hyperparams[kind]
        logger.info(f"{kind}: {len(group_params)} tensors, lr={lr:.6f}")
        param_groups.append(
            {
                "kind": "adamw",
                "params": group_params,
                "lr": lr,
                "initial_lr": lr,
                "betas": (optim_cfg.adamw_beta1, beta2),
                "eps": optim_cfg.adamw_eps,
                "weight_decay": weight_decay,
            }
        )

    return MuonAdamW(param_groups)
