import os
import socket
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from mint.models.gemma import Gemma, GemmaConfig, configure_optimizer
from mint.optim.dist.muon_adamw import MuonAdamWConfig
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.parallel import ParallelContext, setup_parallel


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _dist_worker(
    rank: int,
    world_size: int,
    ddp_level: int,
    tp_size: int,
    sp: bool,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    pctx = ParallelContext.build(
        tp_size=tp_size,
        ddp_level=ddp_level,
        sp=sp,
        device_type="cpu",
    )

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
    torch.manual_seed(1337)
    model = Gemma(cfg, gradient_checkpointing=False)
    setup_parallel(model, pctx)

    optim = configure_optimizer(
        model,
        model_cfg=cfg,
        sched_cfg=SchedulerConfig(),
        optim_cfg=MuonAdamWConfig(),
        dp_group=pctx.dp_group,
    )

    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    loss = out.sum()
    loss.backward()
    optim.step()

    dist.destroy_process_group()


@pytest.mark.parametrize("ddp_level", [0, 1, 2, 3])
def test_distributed_ddp(ddp_level: int) -> None:
    port = _get_free_port()
    spawn(_dist_worker, args=(2, ddp_level, 1, False, port), nprocs=2, join=True)


@pytest.mark.parametrize("sp", [False, True])
def test_distributed_tp(sp: bool) -> None:
    port = _get_free_port()
    spawn(_dist_worker, args=(2, 0, 2, sp, port), nprocs=2, join=True)


def test_distributed_2d_mesh() -> None:
    port = _get_free_port()
    spawn(_dist_worker, args=(4, 3, 2, True, port), nprocs=4, join=True)
