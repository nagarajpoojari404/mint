import torch
from torch import nn

from mint.models.gemma import Gemma, GemmaConfig, configure_optimizer
from mint.optim.dist.muon_adamw import MuonAdamWConfig
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.parallel import ParallelContext, _ZeROParam, setup_parallel


def test_zero_param_basic() -> None:
    p = nn.Parameter(torch.randn(10, 10))
    zp = _ZeROParam.from_param(p, dp_rank=0, dp_size=2, ddp_level=3)
    assert isinstance(zp, _ZeROParam)
    assert isinstance(zp, nn.Parameter)
    assert zp._is_sharded is True
    assert zp.data.shape == (0,)
    assert zp._sliced_param.shape == (50,)

    zp.all_gather_param()
    assert zp._is_sharded is False
    assert zp.data.shape == (10, 10)

    zp.free_param()
    assert zp._is_sharded is True
    assert zp.data.shape == (0,)


def test_zero3_forward_backward() -> None:
    cfg = GemmaConfig(
        embed_dim=64,
        hidden_dim=128,
        seq_length=32,
        vocab_size=100,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
    )
    model = Gemma(cfg, gradient_checkpointing=False)
    pctx = ParallelContext(
        rank=0,
        world_size=1,
        dp_rank=0,
        dp_size=1,
        tp_rank=0,
        tp_size=1,
        ddp_level=3,
        sp=False,
    )
    setup_parallel(model, pctx)

    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 100)

    loss = out.sum()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert isinstance(p, _ZeROParam)
            assert p._sliced_grad is not None


def test_zero3_gradient_checkpointing() -> None:
    cfg = GemmaConfig(
        embed_dim=64,
        hidden_dim=128,
        seq_length=32,
        vocab_size=100,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
    )
    model = Gemma(cfg, gradient_checkpointing=True)
    pctx = ParallelContext(
        rank=0,
        world_size=1,
        dp_rank=0,
        dp_size=1,
        tp_rank=0,
        tp_size=1,
        ddp_level=3,
        sp=False,
    )
    setup_parallel(model, pctx)

    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 100)

    loss = out.sum()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert isinstance(p, _ZeROParam)
            assert p._sliced_grad is not None


def test_optimizer_step_zero_levels() -> None:
    for ddp_level in (0, 1, 2, 3):
        cfg = GemmaConfig(
            embed_dim=64,
            hidden_dim=128,
            seq_length=32,
            vocab_size=100,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
        )
        model = Gemma(cfg, gradient_checkpointing=False)
        pctx = ParallelContext(
            rank=0,
            world_size=1,
            dp_rank=0,
            dp_size=1,
            tp_rank=0,
            tp_size=1,
            ddp_level=ddp_level,
            sp=False,
        )
        setup_parallel(model, pctx)

        optim = configure_optimizer(
            model,
            model_cfg=cfg,
            sched_cfg=SchedulerConfig(),
            optim_cfg=MuonAdamWConfig(),
        )

        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        loss = out.sum()
        loss.backward()

        optim.step()
