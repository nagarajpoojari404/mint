"""Benchmark model shapes on the current GPU using the real training step.

Mirrors the trainer path: torch.compile'd model, autocast + GradScaler, compiled masked loss
(padded-vocab masking), grad clipping and the MuonAdamW optimizer. Reports tokens/s, achieved
TFLOPS (6 FLOPs per matmul param + attention, padded vocab excluded) and peak memory.

    python scripts/bench_shapes.py
    python scripts/bench_shapes.py --shapes 256:1024:8:4:2 512:2048:8:8:2 --micro-batches 64 128
"""

import argparse
import gc
import time

import torch
from torch import nn

from mint.nn.models import Gemma, GemmaConfig, configure_optimizer
from mint.optim.muon_adamw import MuonAdamWConfig
from mint.trainer.base import _masked_cross_entropy
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.device import Device, DeviceConfig

VOCAB = 50267
HEAD_DIM = 64
T4_PEAK_TFLOPS = 65.0  # rated FP16 tensor-core peak; sustained is lower (70 W cap)

# embed_dim:hidden_dim:n_layers:n_heads:n_kv_heads
DEFAULT_SHAPES = [
    "128:128:8:8:2",  # current
    "256:1024:8:4:2",
    "384:1536:8:6:2",
    "512:2048:8:8:2",
    "512:2048:12:8:2",
    "768:3072:12:12:4",
]


def flops_per_token(model: Gemma, seq_len: int, n_heads: int, n_layers: int) -> float:
    blocks = sum(
        layer.weight.numel()
        for b in model.blocks
        for layer in b.modules()
        if isinstance(layer, nn.Linear)
    )
    head = VOCAB * model.config.embed_dim  # real vocab: padding is not useful work
    attn = 4 * seq_len * n_heads * HEAD_DIM * n_layers  # full TxT convention
    return 3 * (2 * (blocks + head) + attn)


def run_one(
    shape: str, micro_batch: int, seq_len: int, steps: int, pad_vocab: int, *, with_optim: bool
) -> dict:
    d, hd, n_layers, n_heads, n_kv = (int(x) for x in shape.split(":"))
    device = Device(DeviceConfig(mixed_precision=True))
    cfg = GemmaConfig(
        embed_dim=d,
        hidden_dim=hd,
        seq_length=seq_len,
        vocab_size=VOCAB,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=HEAD_DIM,
        pad_vocab_multiple=pad_vocab,
    )
    model = Gemma(cfg, gradient_checkpointing=False)
    model = device.move_to_device(model, from_meta=False)
    n_params = model.get_num_params(non_embedding=False)
    f_tok = flops_per_token(model, seq_len, n_heads, n_layers)
    model = device.compile_model(model, enabled=True)
    optimizer = (
        configure_optimizer(model, cfg, SchedulerConfig(), MuonAdamWConfig()) if with_optim else None
    )
    loss_fn = (
        torch.compile(_masked_cross_entropy) if device.is_cuda else _masked_cross_entropy
    )
    vocab_arg = VOCAB if pad_vocab > 1 else None
    extra = {"unpad_logits": False} if pad_vocab > 1 else {}

    x = torch.randint(0, VOCAB, (micro_batch, seq_len), device=device.device)
    y = torch.randint(0, VOCAB, (micro_batch, seq_len), device=device.device)
    mask = torch.ones(micro_batch, seq_len, dtype=torch.long, device=device.device)
    doc = (torch.arange(seq_len, device=device.device) // (seq_len // 2)).expand(micro_batch, -1)

    def step() -> None:
        with device.autocast():
            logits = model(x, doc_ids=doc, **extra)
            loss = loss_fn(logits, y, mask, vocab_arg)
        device.backward(loss)
        if optimizer is not None:
            device.optimizer_step(optimizer, grad_clip=1.0, params=model.parameters())
            optimizer.zero_grad(set_to_none=True)

    if device.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):  # warmup + compile
        step()
    device.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    device.synchronize()
    dt = (time.perf_counter() - start) / steps

    tokens = micro_batch * seq_len
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.is_cuda else float("nan")
    return {
        "params_m": n_params / 1e6,
        "ms": dt * 1e3,
        "tok_s": tokens / dt,
        "tflops": f_tok * tokens / dt / 1e12,
        "peak_gb": peak_gb,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="d:hidden:layers:heads:kv")
    p.add_argument("--micro-batches", nargs="+", type=int, default=[64, 128])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--pad-vocab", type=int, default=64)
    p.add_argument("--no-optim", action="store_true", help="skip optimizer (forward+backward only)")
    a = p.parse_args()

    print(
        f"{'shape':20s}{'mb':>5s}{'params M':>10s}{'ms/mb':>9s}{'tok/s':>10s}"
        f"{'TFLOPS':>8s}{'%peak':>7s}{'peak GB':>9s}"
    )
    for shape in a.shapes:
        for mb in a.micro_batches:
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                r = run_one(
                    shape, mb, a.seq_len, a.steps, a.pad_vocab, with_optim=not a.no_optim
                )
                print(
                    f"{shape:20s}{mb:5d}{r['params_m']:10.1f}{r['ms']:9.0f}{r['tok_s']:10,.0f}"
                    f"{r['tflops']:8.2f}{r['tflops'] / T4_PEAK_TFLOPS:7.1%}{r['peak_gb']:9.2f}"
                )
            except torch.OutOfMemoryError:
                print(f"{shape:20s}{mb:5d}  OOM")
            except Exception as e:  # keep sweeping if one config fails
                print(f"{shape:20s}{mb:5d}  FAILED: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
