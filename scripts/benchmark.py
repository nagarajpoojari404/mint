"""Benchmark model shapes across parallelism strategies using the real training step.

Mirrors the trainer path: torch.compile'd model (where supported), autocast + GradScaler, compiled
masked loss (padded-vocab masking), grad clipping and the DistMuonAdamW optimizer.

Sweep (TP and DDP/ZeRO are never combined; TP runs use ddp_level=0 and dp_size=1):

    single   1 GPU baseline
    ddp0..3  data parallel over all GPUs, ZeRO level 0 / 1 / 2 / 3
    tp       tensor parallel, no sequence parallel
    tp_sp    tensor parallel + sequence parallel

Each (mode, shape, micro-batch) runs in its own ``torchrun`` subprocess so an OOM, hang or compile
failure only affects that cell. Per run we record step time, tokens/s, achieved TFLOPS / MFU, memory,
nvidia-smi samples (GPU/memory-controller util, power, SM clock, temperature) and a torch.profiler
breakdown (kernel busy %, matmul / attention / NCCL / other share) for every GPU.

    python scripts/benchmark.py
    python scripts/benchmark.py --shapes 512:2048:8:8:2 --micro-batches 64 --modes single ddp2 tp_sp
    python scripts/benchmark.py --report-from benchmarks/<run-dir>   # re-render the report only

Outputs go to ``benchmarks/<timestamp>/``: report.md, results.json, results.csv, per_gpu.csv, logs/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

VOCAB = 50267
HEAD_DIM = 64

# embed_dim:hidden_dim:n_layers:n_heads:n_kv_heads
DEFAULT_SHAPES = [
    "128:128:8:8:2",  # current
    "256:1024:8:4:2",
    "384:1536:8:6:2",
    "512:2048:8:8:2",
    "512:2048:12:8:2",
    "768:3072:12:12:4",
]

# Rated dense FP16/BF16 tensor-core peak per GPU (TFLOPS). Sustained is lower; override with --peak-tflops.
GPU_PEAK_TFLOPS = [("T4", 65.0), ("V100", 125.0), ("A100", 312.0), ("H100", 989.0), ("L4", 121.0), ("A10", 125.0)]

MODE_KINDS = ["single", "ddp0", "ddp1", "ddp2", "ddp3", "tp", "tp_sp"]
ZERO_DESC = {0: "DDP (ZeRO-0)", 1: "ZeRO-1 (opt state)", 2: "ZeRO-2 (+grads)", 3: "ZeRO-3 (+params)"}


# --------------------------------------------------------------------------------------------------
# Worker: runs under torchrun, one process per GPU
# --------------------------------------------------------------------------------------------------


def flops_per_token(model: Any, seq_len: int, n_heads: int, n_layers: int, embed_dim: int) -> float:
    """Training FLOPs/token: 6 per matmul param (fwd+bwd) + attention; real (unpadded) vocab head."""
    from torch import nn

    blocks = sum(
        layer.weight.numel()
        for b in model.blocks
        for layer in b.modules()
        if isinstance(layer, nn.Linear)
    )
    head = VOCAB * embed_dim
    attn = 4 * seq_len * n_heads * HEAD_DIM * n_layers  # full TxT convention
    return 3 * (2 * (blocks + head) + attn)


def _param_bytes(model: Any) -> int:
    """Persistent per-rank parameter bytes (a ZeRO-3 parameter only keeps its shard)."""
    total = 0
    for p in model.parameters():
        shard = getattr(p, "_sliced_param", None)
        total += (shard.numel() if shard is not None else p.numel()) * p.element_size()
    return total


def _optimizer_state_bytes(optimizer: Any) -> int:
    import torch

    return sum(
        t.numel() * t.element_size()
        for state in optimizer.state.values()
        for t in state.values()
        if torch.is_tensor(t)
    )


def _start_gpu_sampler(gpu_id: str, path: str) -> tuple[subprocess.Popen | None, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, None
    fh = open(path, "w")  # noqa: SIM115
    query = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu"
    proc = subprocess.Popen(
        [exe, f"--id={gpu_id}", f"--query-gpu={query}", "--format=csv,noheader,nounits", "--loop-ms=100"],
        stdout=fh,
        stderr=subprocess.DEVNULL,
    )
    return proc, fh


def _stop_gpu_sampler(proc: subprocess.Popen | None, fh: Any, path: str) -> dict | None:
    if proc is None:
        return None
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    fh.close()
    cols: list[list[float]] = [[] for _ in range(6)]
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:  # "[N/A]"
            continue
        for c, v in zip(cols, vals, strict=True):
            c.append(v)
    if not cols[0]:
        return None
    avg = [sum(c) / len(c) for c in cols]
    return {
        "util_pct": avg[0],
        "mem_util_pct": avg[1],
        "mem_used_gb": avg[2] / 1024,
        "power_w": avg[3],
        "power_max_w": max(cols[3]),
        "sm_clock_mhz": avg[4],
        "temp_c": avg[5],
        "samples": len(cols[0]),
    }


def _categorize(kernel: str) -> str:
    n = kernel.lower()
    if "nccl" in n:
        return "comm"
    if "memcpy" in n or "memset" in n:
        return "memcpy"
    if any(k in n for k in ("flash", "fmha", "attention", "sdpa")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "cublas", "xmma", "triton_tem", "hmma", "tensorop", "s1688", "s16816", "s884")):
        return "matmul"
    return "other"


def _device_time_us(e: Any) -> float:
    for attr in ("device_time_total", "cuda_time_total", "self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v:
            return float(v)
    return 0.0


def _profile_steps(step: Any, n_steps: int, device: Any) -> dict | None:
    """Kernel-time breakdown from torch.profiler (names are classified heuristically)."""
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(n_steps):
                step()
            device.synchronize()
            wall_us = (time.perf_counter() - t0) * 1e6
        buckets = {"matmul": 0.0, "attention": 0.0, "comm": 0.0, "memcpy": 0.0, "other": 0.0}
        for e in prof.events():
            if "CUDA" not in str(e.device_type):
                continue
            buckets[_categorize(e.name)] += _device_time_us(e)
        total = sum(buckets.values())
        if total <= 0:
            return None
        out = {"kernel_busy_pct": min(100.0, total / wall_us * 100)}
        out.update({f"{k}_pct": v / total * 100 for k, v in buckets.items()})
        return out
    except Exception as e:  # profiling is best-effort
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}


def worker(a: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    from mint.models.gemma import Gemma, GemmaConfig, configure_optimizer
    from mint.optim.muon_adamw import MuonAdamWConfig
    from mint.trainer.base import _masked_cross_entropy
    from mint.trainer.scheduler import SchedulerConfig
    from mint.utils.device import Device, DeviceConfig

    d, hd, n_layers, n_heads, n_kv = (int(x) for x in a.shape.split(":"))
    device = Device(
        DeviceConfig(mixed_precision=True, tp_size=a.tp, ddp_level=a.ddp, sp=a.sp)
    )
    par = device.parallel
    cfg = GemmaConfig(
        embed_dim=d,
        hidden_dim=hd,
        seq_length=a.seq_len,
        vocab_size=VOCAB,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=HEAD_DIM,
        pad_vocab_multiple=a.pad_vocab,
    )
    torch.manual_seed(1234)
    gemma = Gemma(cfg, gradient_checkpointing=False)
    gemma = device.move_to_device(gemma, from_meta=False)  # type: ignore[assignment]

    # size / FLOPs of the *full* model, before TP / ZeRO shard it
    n_params = gemma.get_num_params(non_embedding=False)
    n_params_ne = gemma.get_num_params()
    f_tok = flops_per_token(gemma, a.seq_len, n_heads, n_layers, d)

    model = device.setup_parallel(gemma, tp_plan=cfg.tp_plan)
    param_bytes = _param_bytes(model)
    compiled_model = device.compile_model(model, enabled=not a.no_compile)
    compiled = compiled_model is not model
    model = compiled_model

    optimizer = (
        configure_optimizer(model, cfg, SchedulerConfig(), MuonAdamWConfig(), dp_group=par.dp_group)
        if not a.no_optim
        else None
    )
    loss_fn = (
        torch.compile(_masked_cross_entropy)
        if device.is_cuda and not a.no_compile
        else _masked_cross_entropy
    )
    vocab_arg = VOCAB if a.pad_vocab > 1 else None
    extra = {"unpad_logits": False} if a.pad_vocab > 1 else {}

    torch.manual_seed(4321 + par.dp_rank)  # TP ranks (same dp_rank) see identical data
    mb, T = a.mb, a.seq_len
    x = torch.randint(0, VOCAB, (mb, T), device=device.device)
    y = torch.randint(0, VOCAB, (mb, T), device=device.device)
    mask = torch.ones(mb, T, dtype=torch.long, device=device.device)
    doc = (torch.arange(T, device=device.device) // (T // 2)).expand(mb, -1)

    def step() -> None:
        with device.autocast():
            logits = model(x, doc_ids=doc, **extra)
            loss = loss_fn(logits, y, mask, vocab_arg)
        device.backward(loss)
        if optimizer is not None:
            device.optimizer_step(optimizer, grad_clip=1.0, params=model.parameters())
            optimizer.zero_grad(set_to_none=True)

    for _ in range(a.warmup):  # includes compile
        step()
    device.synchronize()
    static_gb = torch.cuda.memory_allocated() / 1e9
    opt_gb = _optimizer_state_bytes(optimizer) / 1e9 if optimizer is not None else 0.0

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    local = torch.cuda.current_device()
    gpu_id = visible.split(",")[local].strip() if visible else str(local)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_path = tmp.name
    sampler, sampler_fh = _start_gpu_sampler(gpu_id, sample_path)

    start = time.perf_counter()
    for _ in range(a.steps):
        step()
    device.synchronize()
    dt_step = (time.perf_counter() - start) / a.steps

    gpu_stats = _stop_gpu_sampler(sampler, sampler_fh, sample_path)
    Path(sample_path).unlink(missing_ok=True)
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    prof = None if a.no_profile else _profile_steps(step, 3, device)

    rank_res = {
        "rank": par.rank,
        "gpu": torch.cuda.get_device_name(local),
        "ms_step": dt_step * 1e3,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "static_gb": static_gb,
        "param_gb": param_bytes / 1e9,
        "optstate_gb": opt_gb,
        "smi": gpu_stats,
        "prof": prof,
    }
    ranks = [rank_res]
    if distributed:
        ranks = [None] * dist.get_world_size()  # type: ignore[list-item]
        dist.all_gather_object(ranks, rank_res)
        dist.barrier()

    if par.rank == 0:
        world = par.world_size
        ms = max(r["ms_step"] for r in ranks)
        tokens_global = mb * T * par.dp_size
        tok_s = tokens_global / (ms / 1e3)
        tflops = f_tok * tok_s / 1e12
        per_gpu = tflops / world
        result = {
            "status": "ok",
            "world": world,
            "dp_size": par.dp_size,
            "tp": a.tp,
            "ddp": a.ddp,
            "sp": a.sp,
            "compiled": compiled,
            "params_total": n_params,
            "params_non_embed": n_params_ne,
            "flops_per_token": f_tok,
            "tokens_per_step": tokens_global,
            "ms_step": ms,
            "tok_s": tok_s,
            "tok_s_per_gpu": tok_s / world,
            "tflops": tflops,
            "tflops_per_gpu": per_gpu,
            "mfu": per_gpu / a.peak_tflops if a.peak_tflops > 0 else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(local))),
            "amp_dtype": str(device.amp_dtype).replace("torch.", ""),
            "ranks": ranks,
        }
        Path(a.out).write_text(json.dumps(result))

    if distributed:
        dist.destroy_process_group()


# --------------------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------------------


class Mode:
    def __init__(self, kind: str, label: str, name: str, tp: int, ddp: int, sp: bool, world: int) -> None:
        self.kind, self.label, self.name = kind, label, name
        self.tp, self.ddp, self.sp, self.world = tp, ddp, sp, world

    def info(self) -> dict:
        return {
            "mode": self.name,
            "kind": self.kind,
            "label": self.label,
            "world": self.world,
            "tp": self.tp,
            "ddp": self.ddp,
            "sp": self.sp,
        }


def build_modes(n_gpus: int, kinds: list[str], tp_sizes: list[int]) -> list[Mode]:
    modes: list[Mode] = []
    if "single" in kinds:
        modes.append(Mode("single", "1 GPU (baseline)", "single", 1, 0, False, 1))
    if n_gpus >= 2:
        for lvl in range(4):
            if f"ddp{lvl}" in kinds:
                modes.append(Mode(f"ddp{lvl}", ZERO_DESC[lvl], f"ddp{lvl}", 1, lvl, False, n_gpus))
        for t in tp_sizes:
            if t < 2 or t > n_gpus:
                continue
            if "tp" in kinds:
                modes.append(Mode("tp", f"TP={t}", f"tp{t}", t, 0, False, t))
            if "tp_sp" in kinds:
                modes.append(Mode("tp_sp", f"TP={t}+SP", f"tp{t}_sp", t, 0, True, t))
    return modes


def incompatible(mode: Mode, shape: str, seq_len: int) -> str | None:
    if mode.tp <= 1:
        return None
    _, hd, _, n_heads, n_kv = (int(x) for x in shape.split(":"))
    for what, val in (("n_heads", n_heads), ("n_kv_heads", n_kv), ("hidden_dim", hd)):
        if val % mode.tp:
            return f"{what}={val} not divisible by tp={mode.tp}"
    if mode.sp and seq_len % mode.tp:
        return f"seq_len={seq_len} not divisible by tp={mode.tp}"
    return None


def query_gpus() -> tuple[list[dict], str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return [], ""
    gpus: list[dict] = []
    for fields in ("index,name,memory.total,driver_version,compute_cap", "index,name,memory.total,driver_version"):
        r = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            keys = fields.split(",")
            gpus = [dict(zip(keys, (p.strip() for p in line.split(",")), strict=False)) for line in r.stdout.splitlines() if line.strip()]
            break
    topo = subprocess.run([exe, "topo", "-m"], capture_output=True, text=True, check=False)
    return gpus, topo.stdout.strip() if topo.returncode == 0 else ""


def resolve_peak(gpus: list[dict], override: float | None) -> float:
    if override:
        return override
    if gpus:
        name = gpus[0]["name"].upper()
        for key, peak in GPU_PEAK_TFLOPS:
            if key in name:
                return peak
    return 0.0


def _tail_error(log: Path) -> tuple[str, str]:
    text = log.read_text(errors="replace") if log.exists() else ""
    low = text.lower()
    if "outofmemoryerror" in low or "out of memory" in low:
        return "OOM", "CUDA out of memory"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    errs = [ln for ln in lines if "Error" in ln or "error:" in ln.lower()]
    msg = (errs[-1] if errs else (lines[-1] if lines else "no output"))[:160]
    return "FAILED", msg


def run_config(a: argparse.Namespace, mode: Mode, shape: str, mb: int, out_dir: Path, peak: float, *, no_compile: bool) -> dict:
    tag = f"{mode.name}_{shape.replace(':', 'x')}_mb{mb}" + ("_eager" if no_compile else "")
    res_path = out_dir / "raw" / f"{tag}.json"
    log_path = out_dir / "logs" / f"{tag}.log"
    res_path.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={mode.world}",
        "-m", "scripts.benchmark", "--worker",
        "--shape", shape, "--mb", str(mb), "--seq-len", str(a.seq_len), "--steps", str(a.steps),
        "--warmup", str(a.warmup), "--pad-vocab", str(a.pad_vocab), "--peak-tflops", str(peak),
        "--tp", str(mode.tp), "--ddp", str(mode.ddp), "--out", str(res_path),
    ]  # fmt: skip
    if mode.sp:
        cmd.append("--sp")
    if a.no_optim:
        cmd.append("--no-optim")
    if a.no_profile:
        cmd.append("--no-profile")
    if no_compile:
        cmd.append("--no-compile")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return {"status": "TIMEOUT", "note": f"> {a.timeout}s"}
    try:  # reap stragglers, e.g. an nvidia-smi sampler orphaned by a crashed worker
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if proc.returncode == 0 and res_path.exists():
        return json.loads(res_path.read_text())
    status, note = _tail_error(log_path)
    return {"status": status, "note": note}


def summarize_line(r: dict) -> str:
    if r["status"] != "ok":
        return f"{r['status']}: {r.get('note', '')}"
    mfu = f"{r['mfu']:.1%}" if r.get("mfu") is not None else "n/a"
    peak = max(x["peak_alloc_gb"] for x in r["ranks"])
    tag = "" if r["compiled"] else " [eager]"
    return f"{r['ms_step']:.0f} ms | {r['tok_s']:,.0f} tok/s | {r['tflops_per_gpu']:.2f} TFLOPS/GPU ({mfu}) | {peak:.2f} GB{tag}"


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def fmt(v: Any, spec: str = "{:.1f}", na: str = "–") -> str:
    return spec.format(v) if _num(v) else na


def _mean(vals: list[Any]) -> float | None:
    vals = [v for v in vals if _num(v)]
    return sum(vals) / len(vals) if vals else None


def mode_order(r: dict) -> tuple[int, int]:
    order = {"single": 0, "ddp0": 1, "ddp1": 2, "ddp2": 3, "ddp3": 4, "tp": 5, "tp_sp": 6}
    return order.get(r["kind"], 9), r.get("tp", 0)


def rank_avg(r: dict, section: str, key: str) -> float | None:
    return _mean([(x.get(section) or {}).get(key) for x in r["ranks"]])


def enrich(results: list[dict]) -> None:
    base: dict[tuple, dict] = {(r["shape"], r["mb"]): r for r in results if r["kind"] == "single" and r["status"] == "ok"}
    for r in results:
        if r["status"] != "ok":
            continue
        b = base.get((r["shape"], r["mb"]))
        r["speedup"] = r["tok_s"] / b["tok_s"] if b else None
        r["scaling_eff"] = r["speedup"] / r["world"] if b else None
        r["peak_alloc_gb"] = max(x["peak_alloc_gb"] for x in r["ranks"])
        r["peak_reserved_gb"] = max(x["peak_reserved_gb"] for x in r["ranks"])


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--"] + ["--:"] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def observations(groups: dict[tuple, list[dict]]) -> list[str]:
    obs: list[str] = []
    best_count: dict[str, int] = {}
    mem_ratio: dict[str, list[float]] = {"ddp1": [], "ddp2": [], "ddp3": []}
    sp_ratio: list[float] = []
    dp_vs_tp: list[float] = []
    for rows in groups.values():
        ok = [r for r in rows if r["status"] == "ok" and r["kind"] != "single"]
        if ok:
            best = max(ok, key=lambda r: r["tok_s"])
            best_count[best["label"]] = best_count.get(best["label"], 0) + 1
        by_kind = {r["kind"]: r for r in rows if r["status"] == "ok"}
        if "ddp0" in by_kind:
            for k in mem_ratio:
                if k in by_kind:
                    mem_ratio[k].append(by_kind[k]["peak_alloc_gb"] / by_kind["ddp0"]["peak_alloc_gb"])
        if "tp" in by_kind and "tp_sp" in by_kind:
            sp_ratio.append(by_kind["tp_sp"]["tok_s"] / by_kind["tp"]["tok_s"])
        ddp_ok = [r["tok_s"] for r in rows if r["status"] == "ok" and r["kind"].startswith("ddp")]
        tp_ok = [r["tok_s"] for r in rows if r["status"] == "ok" and r["kind"].startswith("tp")]
        if ddp_ok and tp_ok:
            dp_vs_tp.append(max(ddp_ok) / max(tp_ok))
    if best_count:
        top = sorted(best_count.items(), key=lambda kv: -kv[1])
        obs.append("Fastest multi-GPU mode (by global tok/s): " + ", ".join(f"**{k}** in {n}/{len(groups)} cells" for k, n in top) + ".")
    for k, vals in mem_ratio.items():
        if vals:
            m = sum(vals) / len(vals)
            obs.append(f"{ZERO_DESC[int(k[-1])]} peak memory is **{m:.2f}×** DDP-0 on average ({(m - 1):+.0%}).")
    if dp_vs_tp:
        obs.append(f"Best data-parallel mode is **{sum(dp_vs_tp) / len(dp_vs_tp):.2f}×** the best tensor-parallel mode in tok/s on average.")
    if sp_ratio:
        obs.append(f"Adding sequence parallelism changes TP throughput by **{sum(sp_ratio) / len(sp_ratio):.2f}×** on average.")
    return obs


def render_report(meta: dict, results: list[dict]) -> str:
    enrich(results)
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["shape"], r["mb"]), []).append(r)
    for rows in groups.values():
        rows.sort(key=mode_order)
    first_ok = next((r for r in results if r["status"] == "ok"), None)
    peak = meta.get("peak_tflops") or 0.0
    gpus = meta.get("gpus", [])

    L: list[str] = []
    L.append("# Mint parallelism benchmark\n")
    L.append(f"_Generated {meta['started']} on `{meta['host']}`_\n")

    # ---- environment
    L.append("## 1. Environment\n")
    env_rows = [
        ["GPUs", f"{len(gpus) or meta.get('n_gpus', '?')} × {gpus[0]['name']}" if gpus else str(meta.get("n_gpus", "?"))],
        ["GPU memory", f"{float(gpus[0]['memory.total']) / 1024:.1f} GB each" if gpus else "–"],
        ["Driver / compute cap.", f"{gpus[0].get('driver_version', '–')} / {gpus[0].get('compute_cap', first_ok['compute_capability'] if first_ok else '–')}" if gpus else "–"],
        ["PyTorch / CUDA", f"{first_ok['torch']} / {first_ok['cuda']}" if first_ok else "–"],
        ["Autocast dtype", first_ok["amp_dtype"] if first_ok else "–"],
        ["Peak used for MFU", f"{peak:g} TFLOPS per GPU (rated dense FP16/BF16 tensor-core)" if peak else "unknown (MFU omitted)"],
        ["Sequence length", str(meta["seq_len"])],
        ["Timed steps / warmup", f"{meta['steps']} / {meta['warmup']}"],
        ["Optimizer step", "excluded (`--no-optim`)" if meta["no_optim"] else "DistMuonAdamW + grad clip 1.0 + GradScaler"],
        ["Python", meta["python"]],
    ]
    L.append(md_table(["Item", "Value"], env_rows))
    if meta.get("topology"):
        L.append("\n<details><summary>GPU interconnect topology (<code>nvidia-smi topo -m</code>)</summary>\n\n```\n" + meta["topology"] + "\n```\n</details>")

    # ---- methodology
    L.append("\n## 2. Methodology\n")
    L.append(
        "- **Modes**: `single` (1 GPU) · `ddp0..3` (data parallel over all GPUs; ZeRO level 0/1/2/3) · "
        "`tp` / `tp_sp` (tensor parallel over all GPUs without / with sequence parallel). TP and DDP are never combined.\n"
        "- **Micro-batch (mb)** is *per data-parallel replica*. DDP/ZeRO therefore scale the global batch with the GPU count "
        "(weak scaling); TP runs process one batch jointly, so its global batch equals `single`.\n"
        "- **tok/s** is global (all GPUs). **TFLOPS** = model FLOPs/token × tok/s (6 FLOPs per matmul param + attention, real vocab; "
        "padding and recompute excluded). **MFU** = TFLOPS per GPU ÷ rated peak.\n"
        "- **Scaling efficiency** = (tok/s ÷ single-GPU tok/s) ÷ GPU count. For TP this is below 1 by design: TP buys memory headroom, not throughput, at this scale.\n"
        "- **Memory** is `torch.cuda.max_memory_allocated` per GPU over the timed steps (max across ranks).\n"
        "- **GPU util / mem-ctrl util / power / SM clock / temp** are `nvidia-smi` samples (100 ms) taken on every GPU during the timed steps. "
        "*GPU util* is the share of time any kernel is running, not SM occupancy.\n"
        "- **Kernel busy / matmul / attention / NCCL / other** come from `torch.profiler` over 3 extra steps; kernels are classified by name (heuristic). "
        "Busy % can saturate at 100 when kernels overlap on several streams. NCCL share is communication time, including overlapped time.\n"
        "- `[eager]` marks runs where the *model* was not `torch.compile`d (compile failed, or unsupported with ZeRO-3), so they are not directly comparable with compiled runs. "
        "The loss is compiled unless `--no-compile` is used; the optimizer's fused kernels are always compiled.\n"
    )

    # ---- summary
    L.append("## 3. Summary\n")
    rows = []
    for (shape, mb), grp in groups.items():
        ok = [r for r in grp if r["status"] == "ok"]
        if not ok:
            rows.append([f"`{shape}`", str(mb), "–", "–", "–", "–", "all runs failed"])
            continue
        best = max(ok, key=lambda r: r["tok_s"])
        rows.append([
            f"`{shape}`", str(mb), best["label"], fmt(best["tok_s"], "{:,.0f}"), fmt(best["tflops"], "{:.2f}"),
            fmt(best["mfu"] * 100 if best.get("mfu") is not None else None, "{:.1f}%"), fmt(best.get("speedup"), "{:.2f}×"),
        ])  # fmt: skip
    L.append(md_table(["Shape", "MB", "Fastest mode", "tok/s", "TFLOPS (total)", "MFU", "vs 1 GPU"], rows))
    obs = observations(groups)
    if obs:
        L.append("\n**Key observations**\n")
        L += [f"- {o}" for o in obs]

    # ---- model sizes
    L.append("\n## 4. Model sizes\n")
    seen: dict[str, dict] = {}
    for r in results:
        if r["status"] == "ok":
            seen.setdefault(r["shape"], r)
    rows = []
    for shape in dict.fromkeys(r["shape"] for r in results):
        r = seen.get(shape)
        d, hd, nl, nh, nkv = shape.split(":")
        if r:
            rows.append([
                f"`{shape}`", d, hd, nl, nh, nkv, f"{r['params_total'] / 1e6:.1f}", f"{r['params_non_embed'] / 1e6:.1f}",
                f"{r['params_total'] * 4 / 1e6:.0f}", f"{r['flops_per_token'] / 1e6:.1f}",
            ])  # fmt: skip
        else:
            rows.append([f"`{shape}`", d, hd, nl, nh, nkv, "–", "–", "–", "–"])
    L.append(md_table(["Shape", "d", "hidden", "layers", "heads", "kv", "Params M", "Non-emb M", "fp32 MB", "MFLOPs/token (train)"], rows))

    # ---- per-cell details
    L.append("\n## 5. Results by shape and micro-batch\n")
    for (shape, mb), grp in groups.items():
        r0 = seen.get(shape)
        size = f" · {r0['params_total'] / 1e6:.1f}M params" if r0 else ""
        L.append(f"### `{shape}`{size} · mb={mb}\n")

        rows = []
        for r in grp:
            if r["status"] != "ok":
                rows.append([r["label"], str(r["world"]), "–", "–", "–", "–", "–", "–", "–", "–", f"**{r['status']}** {r.get('note', '')}"])
                continue
            rows.append([
                r["label"] + ("" if r["compiled"] else " `[eager]`"), str(r["world"]), fmt(r["ms_step"], "{:.0f}"),
                fmt(r["tok_s"], "{:,.0f}"), fmt(r["tok_s_per_gpu"], "{:,.0f}"), fmt(r["tflops"], "{:.2f}"),
                fmt(r["tflops_per_gpu"], "{:.2f}"), fmt(r["mfu"] * 100 if r.get("mfu") is not None else None, "{:.1f}%"),
                fmt(r.get("speedup"), "{:.2f}×"), fmt(r["scaling_eff"] * 100 if r.get("scaling_eff") is not None else None, "{:.0f}%"),
                r.get("note", ""),
            ])  # fmt: skip
        L.append("**Throughput & efficiency**\n")
        L.append(md_table(["Mode", "GPUs", "Step ms", "tok/s", "tok/s/GPU", "TFLOPS", "TFLOPS/GPU", "MFU", "Speedup", "Scaling eff.", "Note"], rows))

        ok = [r for r in grp if r["status"] == "ok"]
        ddp0 = next((r for r in ok if r["kind"] == "ddp0"), None)
        rows = []
        for r in ok:
            delta = f"{r['peak_alloc_gb'] / ddp0['peak_alloc_gb'] - 1:+.0%}" if ddp0 and r["kind"] != "single" else "–"
            rows.append([
                r["label"], fmt(_mean([x["param_gb"] for x in r["ranks"]]), "{:.3f}"),
                fmt(_mean([x["optstate_gb"] for x in r["ranks"]]), "{:.3f}"), fmt(_mean([x["static_gb"] for x in r["ranks"]]), "{:.2f}"),
                fmt(r["peak_alloc_gb"], "{:.2f}"), fmt(r["peak_reserved_gb"], "{:.2f}"), delta,
            ])  # fmt: skip
        if rows:
            L.append("\n**Memory per GPU (GB)**\n")
            L.append(md_table(["Mode", "Params", "Optimizer state", "Static (after step)", "Peak allocated", "Peak reserved", "Peak vs DDP-0"], rows))

        rows = []
        for r in ok:
            rows.append([
                r["label"], fmt(rank_avg(r, "smi", "util_pct"), "{:.0f}%"), fmt(rank_avg(r, "smi", "mem_util_pct"), "{:.0f}%"),
                fmt(rank_avg(r, "smi", "power_w"), "{:.0f}"), fmt(rank_avg(r, "smi", "sm_clock_mhz"), "{:.0f}"),
                fmt(rank_avg(r, "smi", "temp_c"), "{:.0f}"), fmt(rank_avg(r, "prof", "kernel_busy_pct"), "{:.0f}%"),
                fmt(rank_avg(r, "prof", "matmul_pct"), "{:.0f}%"), fmt(rank_avg(r, "prof", "attention_pct"), "{:.0f}%"),
                fmt(rank_avg(r, "prof", "comm_pct"), "{:.0f}%"), fmt(rank_avg(r, "prof", "other_pct"), "{:.0f}%"),
            ])  # fmt: skip
        if rows:
            L.append("\n**GPU utilization & kernel breakdown (mean over GPUs)**\n")
            L.append(md_table(
                ["Mode", "GPU util", "Mem-ctrl util", "Power W", "SM clock MHz", "Temp °C", "Kernel busy", "Matmul", "Attention", "NCCL", "Other"], rows
            ))  # fmt: skip

        multi = [r for r in ok if r["world"] > 1]
        if multi:
            L.append("\n<details><summary>Per-GPU breakdown</summary>\n")
            rows = []
            for r in multi:
                for x in r["ranks"]:
                    smi, pf = x.get("smi") or {}, x.get("prof") or {}
                    rows.append([
                        r["label"], f"GPU {x['rank']}", fmt(x["ms_step"], "{:.0f}"), fmt(x["peak_alloc_gb"], "{:.2f}"),
                        fmt(x["peak_reserved_gb"], "{:.2f}"), fmt(smi.get("util_pct"), "{:.0f}%"), fmt(smi.get("mem_util_pct"), "{:.0f}%"),
                        fmt(smi.get("power_w"), "{:.0f}"), fmt(smi.get("sm_clock_mhz"), "{:.0f}"), fmt(pf.get("kernel_busy_pct"), "{:.0f}%"),
                        fmt(pf.get("comm_pct"), "{:.0f}%"),
                    ])  # fmt: skip
            L.append(md_table(
                ["Mode", "GPU", "Step ms", "Peak alloc GB", "Peak reserved GB", "GPU util", "Mem-ctrl util", "Power W", "SM MHz", "Kernel busy", "NCCL"], rows
            ))  # fmt: skip
            L.append("\n</details>")
        L.append("")

    failed = [r for r in results if r["status"] not in ("ok",)]
    if failed:
        L.append("## 6. Skipped / failed runs\n")
        L.append(md_table(
            ["Shape", "MB", "Mode", "Status", "Reason"],
            [[f"`{r['shape']}`", str(r["mb"]), r["label"], r["status"], r.get("note", "")] for r in failed],
        ))  # fmt: skip
    return "\n".join(L) + "\n"


def write_outputs(out_dir: Path, meta: dict, results: list[dict]) -> None:
    enrich(results)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    ok_results = [r for r in results if r["status"] == "ok"]
    flat_keys = [
        "shape", "mb", "mode", "label", "world", "tp", "ddp", "sp", "status", "compiled", "params_total", "params_non_embed",
        "flops_per_token", "tokens_per_step", "ms_step", "tok_s", "tok_s_per_gpu", "tflops", "tflops_per_gpu", "mfu",
        "speedup", "scaling_eff", "peak_alloc_gb", "peak_reserved_gb", "note",
    ]  # fmt: skip
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    gpu_keys = ["shape", "mb", "mode", "rank", "gpu", "ms_step", "peak_alloc_gb", "peak_reserved_gb", "static_gb", "param_gb", "optstate_gb"]
    smi_keys = ["util_pct", "mem_util_pct", "mem_used_gb", "power_w", "power_max_w", "sm_clock_mhz", "temp_c"]
    prof_keys = ["kernel_busy_pct", "matmul_pct", "attention_pct", "comm_pct", "memcpy_pct", "other_pct"]
    with open(out_dir / "per_gpu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(gpu_keys + [f"smi_{k}" for k in smi_keys] + [f"prof_{k}" for k in prof_keys])
        for r in ok_results:
            for x in r["ranks"]:
                smi, pf = x.get("smi") or {}, x.get("prof") or {}
                w.writerow(
                    [r["shape"], r["mb"], r["mode"]] + [x.get(k) for k in gpu_keys[3:]]
                    + [smi.get(k) for k in smi_keys] + [pf.get(k) for k in prof_keys]
                )  # fmt: skip
    (out_dir / "report.md").write_text(render_report(meta, results))


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="d:hidden:layers:heads:kv")
    p.add_argument("--micro-batches", nargs="+", type=int, default=[64, 128], help="per data-parallel replica")
    p.add_argument("--modes", nargs="+", choices=MODE_KINDS, default=MODE_KINDS)
    p.add_argument("--gpus", type=int, default=None, help="GPUs to use (default: all visible)")
    p.add_argument("--tp-sizes", nargs="+", type=int, default=None, help="TP degrees (default: all GPUs)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--pad-vocab", type=int, default=64)
    p.add_argument("--peak-tflops", type=float, default=None, help="per-GPU peak for MFU (default: look up by GPU name)")
    p.add_argument("--no-optim", action="store_true", help="skip optimizer (forward+backward only)")
    p.add_argument("--no-compile", action="store_true", help="never use torch.compile")
    p.add_argument("--no-profile", action="store_true", help="skip the torch.profiler pass")
    p.add_argument("--timeout", type=int, default=900, help="seconds per run")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--report-from", type=Path, default=None, help="re-render report/CSVs from an existing run dir")
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--shape", help=argparse.SUPPRESS)
    p.add_argument("--mb", type=int, help=argparse.SUPPRESS)
    p.add_argument("--tp", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--ddp", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--sp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        a.peak_tflops = a.peak_tflops or 0.0
    return a


def main() -> None:
    a = parse_args()
    if a.worker:
        worker(a)
        return

    if a.report_from:
        data = json.loads((a.report_from / "results.json").read_text())
        write_outputs(a.report_from, data["meta"], data["results"])
        print(f"report written to {a.report_from / 'report.md'}")
        return

    gpus, topo = query_gpus()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    n_avail = len(visible.split(",")) if visible else len(gpus)
    if not n_avail:
        try:
            import torch

            n_avail = torch.cuda.device_count()
        except ImportError:
            n_avail = 0
    if n_avail == 0:
        sys.exit("No CUDA GPUs found.")
    n_gpus = min(a.gpus or n_avail, n_avail)
    tp_sizes = a.tp_sizes or [n_gpus]
    peak = resolve_peak(gpus, a.peak_tflops)
    modes = build_modes(n_gpus, a.modes, tp_sizes)

    out_dir = a.out_dir or ROOT / "benchmarks" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    meta = {
        "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "python": platform.python_version(),
        "gpus": gpus[:n_gpus],
        "n_gpus": n_gpus,
        "topology": topo,
        "peak_tflops": peak,
        "seq_len": a.seq_len,
        "steps": a.steps,
        "warmup": a.warmup,
        "no_optim": a.no_optim,
    }
    total = len(modes) * len(a.shapes) * len(a.micro_batches)
    print(f"{n_gpus} GPU(s); {len(modes)} modes × {len(a.shapes)} shapes × {len(a.micro_batches)} micro-batches = {total} runs")
    print(f"output: {out_dir}\n")

    results: list[dict] = []
    done = 0
    try:
        for shape in a.shapes:
            for mb in a.micro_batches:
                for mode in modes:
                    done += 1
                    base = {**mode.info(), "shape": shape, "mb": mb, "seq_len": a.seq_len}
                    reason = incompatible(mode, shape, a.seq_len)
                    if reason:
                        r = {"status": "SKIPPED", "note": reason}
                    else:
                        r = run_config(a, mode, shape, mb, out_dir, peak, no_compile=a.no_compile)
                        if r["status"] == "FAILED" and not a.no_compile:
                            first_err = r["note"]
                            r = run_config(a, mode, shape, mb, out_dir, peak, no_compile=True)
                            if r["status"] == "ok":
                                r["note"] = f"eager fallback: {first_err[:80]}"
                    r = {**base, **r}
                    results.append(r)
                    print(f"[{done}/{total}] {shape:18s} mb={mb:<4d} {mode.label:22s} {summarize_line(r)}", flush=True)
                    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
                    time.sleep(2)  # let the GPUs release memory
    except KeyboardInterrupt:
        print("\ninterrupted; writing partial results")
    finally:
        if results:
            write_outputs(out_dir, meta, results)
            print(f"\nreport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
