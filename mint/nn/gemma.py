import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mint.kvcache.base import KVCache
from mint.nn.base import ModelConfig
from mint.nn.blocks import GQAGluBlock
from mint.nn.norm import RMSNorm
from mint.optim.dist.muon_adamw import DistMuonAdamW, MuonAdamWConfig
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
    # Round embedding / lm_head rows up to a multiple of this for tensor-core alignment.
    pad_vocab_multiple: int = 1
    # Gemma architecture-specific flags & hyperparameters
    scale_embeddings: bool = True
    use_post_norms: bool = True
    use_qk_norm: bool = False
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    sliding_window: int | None = None
    sliding_window_pattern: str | None = "alternate"
    query_pre_attn_scalar: float | None = None


@dataclass
class Gemma1Config(GemmaConfig):
    rope_base_theta: int = 10000
    use_post_norms: bool = False
    use_qk_norm: bool = False
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    sliding_window: int | None = None
    sliding_window_pattern: str | None = None


@dataclass
class Gemma2Config(GemmaConfig):
    rope_base_theta: int = 10000
    use_post_norms: bool = True
    use_qk_norm: bool = False
    attn_logit_softcapping: float | None = 50.0
    final_logit_softcapping: float | None = 30.0
    sliding_window: int | None = 4096
    sliding_window_pattern: str | None = "alternate"


@dataclass
class Gemma3Config(GemmaConfig):
    rope_base_theta: int = 1000000
    use_post_norms: bool = True
    use_qk_norm: bool = True
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    sliding_window: int | None = 4096
    sliding_window_pattern: str | None = "alternate"


# =============================================================================
# Gemma Model Family Preset Factory Functions
# =============================================================================


def gemma_1_2b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma1Config:
    """Gemma 1 2B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 2048,
        "hidden_dim": 16384,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 18,
        "n_heads": 8,
        "n_kv_heads": 1,
        "head_dim": 256,
        "rope_base_theta": 10000,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma1Config(**params)


def gemma_1_7b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma1Config:
    """Gemma 1 7B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 3072,
        "hidden_dim": 24576,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 28,
        "n_heads": 16,
        "n_kv_heads": 16,
        "head_dim": 256,
        "rope_base_theta": 10000,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma1Config(**params)


def codegemma_2b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma1Config:
    """CodeGemma 2B configuration."""
    return gemma_1_2b_config(seq_length=seq_length, vocab_size=vocab_size, dtype=dtype, **kwargs)


def codegemma_7b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma1Config:
    """CodeGemma 7B configuration."""
    return gemma_1_7b_config(seq_length=seq_length, vocab_size=vocab_size, dtype=dtype, **kwargs)


def gemma_2_2b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma2Config:
    """Gemma 2 2B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 2304,
        "hidden_dim": 9216,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 26,
        "n_heads": 8,
        "n_kv_heads": 4,
        "head_dim": 256,
        "rope_base_theta": 10000,
        "sliding_window": 4096,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 256**-0.5,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma2Config(**params)


def gemma_2_9b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma2Config:
    """Gemma 2 9B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 3584,
        "hidden_dim": 14336,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 42,
        "n_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 256,
        "rope_base_theta": 10000,
        "sliding_window": 4096,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 256**-0.5,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma2Config(**params)


def gemma_2_27b_config(
    seq_length: int = 8192,
    vocab_size: int = 256000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma2Config:
    """Gemma 2 27B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 4608,
        "hidden_dim": 36864,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 46,
        "n_heads": 32,
        "n_kv_heads": 16,
        "head_dim": 128,
        "rope_base_theta": 10000,
        "sliding_window": 4096,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 128**-0.5,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma2Config(**params)


def gemma_3_1b_config(
    seq_length: int = 32768,
    vocab_size: int = 262144,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma3Config:
    """Gemma 3 1B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 1152,
        "hidden_dim": 4608,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 24,
        "n_heads": 4,
        "n_kv_heads": 1,
        "head_dim": 256,
        "rope_base_theta": 1000000,
        "sliding_window": 4096,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma3Config(**params)


def gemma_3_4b_config(
    seq_length: int = 32768,
    vocab_size: int = 262144,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma3Config:
    """Gemma 3 4B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 2560,
        "hidden_dim": 10240,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 34,
        "n_heads": 10,
        "n_kv_heads": 2,
        "head_dim": 256,
        "rope_base_theta": 1000000,
        "sliding_window": 4096,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma3Config(**params)


def gemma_3_12b_config(
    seq_length: int = 32768,
    vocab_size: int = 262144,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma3Config:
    """Gemma 3 12B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 3840,
        "hidden_dim": 15360,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 40,
        "n_heads": 16,
        "n_kv_heads": 4,
        "head_dim": 256,
        "rope_base_theta": 1000000,
        "sliding_window": 4096,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma3Config(**params)


def gemma_3_27b_config(
    seq_length: int = 32768,
    vocab_size: int = 262144,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> Gemma3Config:
    """Gemma 3 27B configuration."""
    params: dict[str, Any] = {
        "embed_dim": 4608,
        "hidden_dim": 36864,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 46,
        "n_heads": 32,
        "n_kv_heads": 16,
        "head_dim": 128,
        "rope_base_theta": 1000000,
        "sliding_window": 4096,
        "dtype": dtype,
    }
    params.update(kwargs)
    return Gemma3Config(**params)


def gemma_56m_config(
    seq_length: int = 128,
    vocab_size: int = 50267,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> GemmaConfig:
    """Gemma 56M configuration for rapid dev, benchmarking, and training on single GPUs."""
    params: dict[str, Any] = {
        "embed_dim": 512,
        "hidden_dim": 2048,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 2,
        "head_dim": 64,
        "rope_base_theta": 100000,
        "tie_weights": True,
        "pad_vocab_multiple": 64,
        "dtype": dtype,
    }
    params.update(kwargs)
    return GemmaConfig(**params)


def gemma_tiny_config(
    seq_length: int = 128,
    vocab_size: int = 1000,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> GemmaConfig:
    """Tiny Gemma configuration for fast unit testing."""
    params: dict[str, Any] = {
        "embed_dim": 128,
        "hidden_dim": 512,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 32,
        "rope_base_theta": 10000,
        "tie_weights": True,
        "pad_vocab_multiple": 1,
        "dtype": dtype,
    }
    params.update(kwargs)
    return GemmaConfig(**params)


GEMMA_CONFIG_REGISTRY: dict[str, Callable[..., GemmaConfig]] = {
    # Gemma 1
    "gemma-2b": gemma_1_2b_config,
    "gemma-1-2b": gemma_1_2b_config,
    "gemma-7b": gemma_1_7b_config,
    "gemma-1-7b": gemma_1_7b_config,
    "codegemma-2b": codegemma_2b_config,
    "codegemma-7b": codegemma_7b_config,
    # Gemma 2
    "gemma-2-2b": gemma_2_2b_config,
    "gemma-2-9b": gemma_2_9b_config,
    "gemma-2-27b": gemma_2_27b_config,
    # Gemma 3
    "gemma-3-1b": gemma_3_1b_config,
    "gemma-3-4b": gemma_3_4b_config,
    "gemma-3-12b": gemma_3_12b_config,
    "gemma-3-27b": gemma_3_27b_config,
    # Small / Benchmarking
    "gemma-56m": gemma_56m_config,
    "56m": gemma_56m_config,
    "gemma-tiny": gemma_tiny_config,
    "tiny": gemma_tiny_config,
}


def get_gemma_config(name_or_variant: str, **overrides: Any) -> GemmaConfig:
    """Look up a Gemma model preset configuration by name with optional overrides."""
    normalized_key = name_or_variant.lower().strip().replace("_", "-")
    if normalized_key in GEMMA_CONFIG_REGISTRY:
        return GEMMA_CONFIG_REGISTRY[normalized_key](**overrides)
    raise ValueError(
        f"Unknown Gemma configuration: '{name_or_variant}'. "
        f"Available presets: {list(GEMMA_CONFIG_REGISTRY.keys())}"
    )


# =============================================================================
# Gemma Model Implementation
# =============================================================================


class Gemma(nn.Module):
    """Full-family Gemma model implementing Gemma 1, Gemma 2, and Gemma 3 architectures."""

    def __init__(self, config: GemmaConfig, *, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        m = config.pad_vocab_multiple
        self.padded_vocab_size = (config.vocab_size + m - 1) // m * m
        self.tok_embeddings = nn.Embedding(
            self.padded_vocab_size, config.embed_dim, dtype=config.dtype
        )
        self.embed_scaler = math.sqrt(config.embed_dim) if config.scale_embeddings else 1.0

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
                attention_type=self._get_layer_attention_type(i, config),
                sliding_window=config.sliding_window,
                use_post_norms=config.use_post_norms,
                use_qk_norm=config.use_qk_norm,
                attn_logit_softcapping=config.attn_logit_softcapping,
                query_pre_attn_scalar=config.query_pre_attn_scalar,
            )
            for i in range(config.n_layers)
        )

        self.norm = RMSNorm(config.embed_dim)
        self.output = nn.Linear(
            config.embed_dim, self.padded_vocab_size, bias=False, dtype=config.dtype
        )
        self.reset_parameters()

    @staticmethod
    def _get_layer_attention_type(layer_idx: int, config: GemmaConfig) -> str:
        pattern = config.sliding_window_pattern
        if config.sliding_window is None or pattern is None:
            return "L"
        if pattern == "alternate":
            return "S" if layer_idx % 2 == 0 else "L"
        if pattern == "all_s":
            return "S"
        if pattern == "all_l":
            return "L"
        if pattern == "gemma3_5_1":
            # 5 local layers followed by 1 global layer
            return "S" if (layer_idx + 1) % 6 != 0 else "L"
        return "L"

    def reset_parameters(self) -> None:
        # Module.to_empty breaks weight tying, so it is re-established after every re-init
        if self.config.tie_weights:
            self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        doc_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        unpad_logits: bool = True,
    ) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        if self.config.scale_embeddings:
            h = h * self.embed_scaler

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

        h = self.norm(h)
        logits = self.output(h)

        if (
            self.config.final_logit_softcapping is not None
            and self.config.final_logit_softcapping > 0
        ):
            logits = (
                torch.tanh(logits / self.config.final_logit_softcapping)
                * self.config.final_logit_softcapping
            )

        # Training passes unpad_logits=False and slices inside compiled loss instead.
        if unpad_logits and self.padded_vocab_size != self.config.vocab_size:
            logits = logits[..., : self.config.vocab_size]
        return logits

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
        """Sample max_new_tokens tokens; temperature=0 decodes greedily."""
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


class Gemma1(Gemma):
    """Gemma 1 model specialization."""

    def __init__(self, config: Gemma1Config, *, gradient_checkpointing: bool = False) -> None:
        super().__init__(config, gradient_checkpointing=gradient_checkpointing)


class Gemma2(Gemma):
    """Gemma 2 model specialization with dual RMSNorm and softcapping."""

    def __init__(self, config: Gemma2Config, *, gradient_checkpointing: bool = False) -> None:
        super().__init__(config, gradient_checkpointing=gradient_checkpointing)


class Gemma3(Gemma):
    """Gemma 3 model specialization with dual RMSNorm and Q/K RMSNorm."""

    def __init__(self, config: Gemma3Config, *, gradient_checkpointing: bool = False) -> None:
        super().__init__(config, gradient_checkpointing=gradient_checkpointing)


class CodeGemma(Gemma1):
    """CodeGemma model specialization."""


# =============================================================================
# Optimizer Configuration
# =============================================================================


def configure_optimizer(
    model: nn.Module,
    model_cfg: GemmaConfig,
    sched_cfg: SchedulerConfig,
    optim_cfg: MuonAdamWConfig,
) -> DistMuonAdamW:
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

    return DistMuonAdamW(param_groups)
