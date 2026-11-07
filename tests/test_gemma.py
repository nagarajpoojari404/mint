import pytest
import torch

from mint.kvcache.base import KVCache
from mint.nn import (
    CodeGemma,
    Gemma,
    Gemma1,
    Gemma2,
    Gemma3,
    GemmaConfig,
    codegemma_2b_config,
    codegemma_7b_config,
    configure_optimizer,
    gemma_1_2b_config,
    gemma_1_7b_config,
    gemma_2_2b_config,
    gemma_2_9b_config,
    gemma_2_27b_config,
    gemma_3_1b_config,
    gemma_3_4b_config,
    gemma_3_12b_config,
    gemma_3_27b_config,
    gemma_56m_config,
    gemma_tiny_config,
    get_gemma_config,
)
from mint.optim.dist.muon_adamw import MuonAdamWConfig
from mint.trainer.scheduler import SchedulerConfig


@pytest.mark.parametrize(
    ("name", "expected_layers", "expected_embed_dim"),
    [
        ("gemma-1-2b", 18, 2048),
        ("gemma-1-7b", 28, 3072),
        ("codegemma-2b", 18, 2048),
        ("codegemma-7b", 28, 3072),
        ("gemma-2-2b", 26, 2304),
        ("gemma-2-9b", 42, 3584),
        ("gemma-2-27b", 46, 4608),
        ("gemma-3-1b", 24, 1152),
        ("gemma-3-4b", 34, 2560),
        ("gemma-3-12b", 40, 3840),
        ("gemma-3-27b", 46, 4608),
        ("gemma-56m", 8, 512),
        ("gemma-tiny", 2, 128),
    ],
)
def test_preset_configs(name: str, expected_layers: int, expected_embed_dim: int) -> None:
    cfg = get_gemma_config(name)
    assert cfg.n_layers == expected_layers
    assert cfg.embed_dim == expected_embed_dim


def test_gemma_forward_and_generate() -> None:
    cfg = gemma_tiny_config(seq_length=32, vocab_size=200)
    model = Gemma(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, cfg.vocab_size)

    gen = model.generate(x, max_new_tokens=4, temperature=0.0)
    assert gen.shape == (2, 12)


def test_gemma1_architecture() -> None:
    cfg = gemma_1_2b_config(seq_length=16, vocab_size=100, n_layers=2)
    assert not cfg.use_post_norms
    assert not cfg.use_qk_norm
    model = Gemma1(cfg)
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 100)


def test_gemma2_architecture_and_softcapping() -> None:
    cfg = gemma_2_2b_config(seq_length=16, vocab_size=100, n_layers=2)
    assert cfg.use_post_norms
    assert cfg.attn_logit_softcapping == 50.0
    assert cfg.final_logit_softcapping == 30.0
    model = Gemma2(cfg)
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 100)


def test_gemma3_architecture_and_qk_norm() -> None:
    cfg = gemma_3_1b_config(seq_length=16, vocab_size=100, n_layers=2)
    assert cfg.use_post_norms
    assert cfg.use_qk_norm
    model = Gemma3(cfg)
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 100)


def test_kv_cache_generation() -> None:
    cfg = gemma_2_2b_config(seq_length=32, vocab_size=100, n_layers=2)
    model = Gemma2(cfg)
    x = torch.randint(0, 100, (2, 8))
    kv_caches = [
        KVCache(
            max_batch_size=2, max_seq_len=32, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim
        )
        for _ in range(cfg.n_layers)
    ]
    gen = model.generate(x, max_new_tokens=3, kv_caches=kv_caches)
    assert gen.shape == (2, 11)


def test_optimizer_configuration() -> None:
    cfg = gemma_tiny_config()
    model = Gemma(cfg)
    opt = configure_optimizer(model, cfg, SchedulerConfig(), MuonAdamWConfig())
    assert len(opt.param_groups) > 0
