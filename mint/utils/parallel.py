import fnmatch
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from mint.utils.logger import logger


def dp_tp_ranks(rank: int, tp_size: int) -> tuple[int, int]:
    return rank // tp_size, rank % tp_size


@dataclass(frozen=True)
class ParallelContext:
    rank: int = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    ddp_level: int = 0
    sp: bool = False
    dp_group: Any = None
    tp_group: Any = None
    mesh: DeviceMesh | None = None

    @classmethod
    def build(
        cls,
        *,
        tp_size: int = 1,
        ddp_level: int = 0,
        sp: bool = False,
        device_type: str = "cpu",
    ) -> "ParallelContext":
        if not (dist.is_available() and dist.is_initialized()):
            if tp_size != 1:
                raise ValueError(f"tp_size={tp_size} requires a distributed launch")
            return cls(
                rank=int(os.getenv("RANK", "0")),
                world_size=int(os.getenv("WORLD_SIZE", "1")),
                ddp_level=ddp_level,
                sp=sp,
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if tp_size < 1 or world_size % tp_size != 0:
            raise ValueError(f"world_size={world_size} is not divisible by tp_size={tp_size}")

        if tp_size == 1:
            return cls(
                rank=rank,
                world_size=world_size,
                dp_rank=rank,
                dp_size=world_size,
                ddp_level=ddp_level,
                sp=sp,
                dp_group=None,
            )

        dp_size = world_size // tp_size
        mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        dp_rank, tp_rank = dp_tp_ranks(rank, tp_size)
        return cls(
            rank=rank,
            world_size=world_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            ddp_level=ddp_level,
            sp=sp,
            dp_group=mesh.get_group("dp"),
            tp_group=mesh.get_group("tp"),
            mesh=mesh,
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class _ZeROParam(nn.Parameter):
    _sliced_param: torch.Tensor | None
    _sliced_grad: torch.Tensor | None
    _orig_shape: torch.Size
    _orig_numel: int
    _chunk_size: int
    _pad_size: int
    _is_sharded: bool
    _dp_group: Any
    _dp_rank: int
    _dp_size: int
    _ddp_level: int

    _fallback_logged: bool = False
    _communication_logged: bool = False
    _grad_fallback_logged: bool = False
    _grad_communication_logged: bool = False

    def __new__(
        cls,
        data: torch.Tensor | None = None,
        requires_grad: bool = True,
    ) -> "_ZeROParam":
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @classmethod
    def from_param(
        cls,
        param: nn.Parameter,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_group: Any = None,
        ddp_level: int = 0,
    ) -> "_ZeROParam":
        orig_shape = param.data.shape
        orig_numel = param.data.numel()
        chunk_size = (orig_numel + dp_size - 1) // dp_size if dp_size > 1 else orig_numel
        pad_size = chunk_size * dp_size - orig_numel if dp_size > 1 else 0

        if ddp_level == 3 and dp_size > 1:
            flat = param.data.detach().reshape(-1)
            if pad_size > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_size))
            sliced = flat[dp_rank * chunk_size : (dp_rank + 1) * chunk_size].clone()
            zp = cls(
                torch.empty(0, dtype=param.dtype, device=param.device),
                requires_grad=param.requires_grad,
            )
            zp._is_sharded = True
            zp._sliced_param = sliced
        else:
            zp = cls(param.data.clone(), requires_grad=param.requires_grad)
            zp._is_sharded = False
            zp._sliced_param = None

        zp._sliced_grad = None
        zp._orig_shape = orig_shape
        zp._orig_numel = orig_numel
        zp._chunk_size = chunk_size
        zp._pad_size = pad_size
        zp._dp_group = dp_group
        zp._dp_rank = dp_rank
        zp._dp_size = dp_size
        zp._ddp_level = ddp_level
        return zp

    def all_gather_param(self, async_op: bool = False) -> Callable[[], None] | None:
        if not self._is_sharded:
            return None
        if self._dp_size == 1 or not (dist.is_available() and dist.is_initialized()):
            if not _ZeROParam._fallback_logged:
                _ZeROParam._fallback_logged = True
                logger.warning(
                    f"[ZeROParam debug] Fallback to local parameter copy detected! "
                    f"dp_size={self._dp_size}, dist_initialized={dist.is_initialized()}. "
                    f"No real NCCL communication is occurring!"
                )
            if self._sliced_param is not None:
                if self._sliced_param.numel() == self._orig_numel:
                    self.data = self._sliced_param.view(self._orig_shape)
                else:
                    self.data = self._sliced_param.repeat(self._dp_size)[: self._orig_numel].view(self._orig_shape)
            self._is_sharded = False
            return None

        if self._sliced_param is None:
            return None

        if not _ZeROParam._communication_logged:
            _ZeROParam._communication_logged = True
            logger.info(
                f"[ZeROParam debug] Performing REAL NCCL all_gather_into_tensor communication! "
                f"dp_size={self._dp_size}, group_is_none={self._dp_group is None}, async_op={async_op}"
            )

        gathered = torch.empty(
            self._chunk_size * self._dp_size,
            dtype=self._sliced_param.dtype,
            device=self._sliced_param.device,
        )
        work = dist.all_gather_into_tensor(
            gathered, self._sliced_param, group=self._dp_group, async_op=async_op
        )

        def finish() -> None:
            if work is not None:
                work.wait()
            self.data = gathered[: self._orig_numel].view(self._orig_shape)
            self._is_sharded = False

        if async_op and work is not None:
            return finish
        finish()
        return None

    def free_param(self) -> None:
        if self._ddp_level == 3 and self._dp_size > 1 and not self._is_sharded:
            dtype = (
                self._sliced_param.dtype
                if self._sliced_param is not None
                else self.dtype
            )
            self.data = torch.empty(0, dtype=dtype, device=self.device)
            self._is_sharded = True

    def reduce_scatter_grad(self, grad: torch.Tensor, async_op: bool = False) -> Any:
        if self._dp_size == 1 or not (dist.is_available() and dist.is_initialized()):
            if not _ZeROParam._grad_fallback_logged:
                _ZeROParam._grad_fallback_logged = True
                logger.warning(
                    f"[ZeROParam debug] Fallback to local gradient copy detected! "
                    f"dp_size={self._dp_size}, dist_initialized={dist.is_initialized()}. "
                    f"No real NCCL communication is occurring!"
                )
            g = grad.detach().reshape(-1)[: self._orig_numel].clone()
            self._sliced_grad = g if self._sliced_grad is None else self._sliced_grad + g
            return None

        if not _ZeROParam._grad_communication_logged:
            _ZeROParam._grad_communication_logged = True
            logger.info(
                f"[ZeROParam debug] Performing REAL NCCL reduce_scatter_tensor communication! "
                f"dp_size={self._dp_size}, group_is_none={self._dp_group is None}, async_op={async_op}"
            )

        flat_grad = grad.detach().reshape(-1)
        if self._pad_size > 0:
            flat_grad = torch.nn.functional.pad(flat_grad, (0, self._pad_size))

        out_slice = torch.empty(self._chunk_size, dtype=grad.dtype, device=grad.device)
        work = dist.reduce_scatter_tensor(
            out_slice,
            flat_grad,
            op=dist.ReduceOp.AVG,
            group=self._dp_group,
            async_op=async_op,
        )

        def finish() -> None:
            if work is not None:
                work.wait()
            if self._sliced_grad is None:
                self._sliced_grad = out_slice
            else:
                self._sliced_grad.add_(out_slice)

        if async_op and work is not None:
            return finish
        finish()
        return None
        

class _AllGatherSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        if group is None or dist.get_world_size(group) == 1:
            return x
        tp_size = dist.get_world_size(group)
        bsz, seq_len, d = x.shape
        x_t = x.transpose(0, 1).contiguous()
        gathered_t = torch.empty(seq_len * tp_size, bsz, d, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered_t, x_t, group=group)
        return gathered_t.transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        group = ctx.group
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None
        tp_size = dist.get_world_size(group)
        bsz, seq_len, d = grad_output.shape
        grad_t = grad_output.transpose(0, 1).contiguous()
        out_t = torch.empty(
            seq_len // tp_size, bsz, d, dtype=grad_output.dtype, device=grad_output.device
        )
        dist.reduce_scatter_tensor(out_t, grad_t, op=dist.ReduceOp.SUM, group=group)
        return out_t.transpose(0, 1).contiguous(), None


class _ReduceScatterSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        if group is None or dist.get_world_size(group) == 1:
            return x
        tp_size = dist.get_world_size(group)
        bsz, seq_len, d = x.shape
        x_t = x.transpose(0, 1).contiguous()
        out_t = torch.empty(seq_len // tp_size, bsz, d, dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out_t, x_t, op=dist.ReduceOp.SUM, group=group)
        return out_t.transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        group = ctx.group
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None
        tp_size = dist.get_world_size(group)
        bsz, seq_len, d = grad_output.shape
        grad_t = grad_output.transpose(0, 1).contiguous()
        gathered_t = torch.empty(
            seq_len * tp_size, bsz, d, dtype=grad_output.dtype, device=grad_output.device
        )
        dist.all_gather_into_tensor(gathered_t, grad_t, group=group)
        return gathered_t.transpose(0, 1).contiguous(), None


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: Any) -> torch.Tensor:
        if group is None or dist.get_world_size(group) == 1:
            return x
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0], None


class _ScatterSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: Any, rank: int) -> torch.Tensor:
        ctx.group = group
        if group is None or dist.get_world_size(group) == 1:
            return x
        tp_size = dist.get_world_size(group)
        return x.chunk(tp_size, dim=1)[rank].contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        group = ctx.group
        if group is None or dist.get_world_size(group) == 1:
            return grad_output, None, None
        tp_size = dist.get_world_size(group)
        bsz, seq_len, d = grad_output.shape
        grad_t = grad_output.transpose(0, 1).contiguous()
        gathered_t = torch.empty(
            seq_len * tp_size, bsz, d, dtype=grad_output.dtype, device=grad_output.device
        )
        dist.all_gather_into_tensor(gathered_t, grad_t, group=group)
        return gathered_t.transpose(0, 1).contiguous(), None, None


def _match_plan(name: str, plan: dict[str, str]) -> str | None:
    for pattern, kind in plan.items():
        if fnmatch.fnmatch(name, pattern):
            return kind
    return None


def apply_tp(
    model: nn.Module,
    parallel: ParallelContext,
    tp_plan: dict[str, str] | None = None,
) -> nn.Module:
    if parallel.tp_size <= 1:
        return model

    plan = tp_plan or getattr(getattr(model, "config", None), "tp_plan", None)
    if not plan:
        return model

    tp_size = parallel.tp_size
    tp_rank = parallel.tp_rank
    tp_group = parallel.tp_group
    sp = parallel.sp

    for name, module in model.named_modules():
        kind = _match_plan(name, plan)
        if kind == "col" and isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.chunk(tp_size, dim=0)[tp_rank].contiguous()
            if module.bias is not None:
                module.bias.data = module.bias.data.chunk(tp_size, dim=0)[tp_rank].contiguous()
            if sp:
                module.register_forward_pre_hook(
                    lambda _m, args, grp=tp_group: (
                        _AllGatherSeq.apply(args[0], grp),
                        *args[1:],
                    )
                )
        elif kind == "row" and isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.chunk(tp_size, dim=1)[tp_rank].contiguous()
            if sp:
                module.register_forward_hook(
                    lambda _m, _args, out, grp=tp_group: _ReduceScatterSeq.apply(out, grp)
                )
            else:
                module.register_forward_hook(
                    lambda _m, _args, out, grp=tp_group: _AllReduce.apply(out, grp)
                )

        if hasattr(module, "n_heads") and getattr(module, "n_heads") % tp_size == 0:
            module.n_heads //= tp_size  # type: ignore[operator]
        if hasattr(module, "n_kv_heads") and getattr(module, "n_kv_heads") % tp_size == 0:
            module.n_kv_heads //= tp_size  # type: ignore[operator]

    if sp:
        tok_embed = getattr(model, "tok_embeddings", None)
        if isinstance(tok_embed, nn.Module):
            tok_embed.register_forward_hook(
                lambda _m, _args, out, grp=tp_group, rk=tp_rank: _ScatterSeq.apply(out, grp, rk)
            )
        output_mod = getattr(model, "output", None)
        if isinstance(output_mod, nn.Module):
            output_mod.register_forward_pre_hook(
                lambda _m, args, grp=tp_group: (_AllGatherSeq.apply(args[0], grp), *args[1:])
            )

    return model


def apply_ddp(model: nn.Module, parallel: ParallelContext) -> nn.Module:
    memo: dict[int, _ZeROParam] = {}
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            if id(param) not in memo:
                memo[id(param)] = _ZeROParam.from_param(
                    param,
                    dp_rank=parallel.dp_rank,
                    dp_size=parallel.dp_size,
                    dp_group=parallel.dp_group,
                    ddp_level=parallel.ddp_level,
                )
            module._parameters[name] = memo[id(param)]

    for p in set(memo.values()):
        if not p.requires_grad:
            continue

        def _make_grad_hook(param: _ZeROParam) -> Any:
            def _post_grad(_p: nn.Parameter) -> None:
                if param.grad is not None:
                    if param._ddp_level in (2, 3):
                        param.reduce_scatter_grad(param.grad)
                        param.grad = None
                    elif param._ddp_level in (0, 1) and param._dp_size > 1:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=param._dp_group)
                if param._ddp_level == 3 and param._dp_size > 1:
                    param.free_param()

            return _post_grad

        p.register_post_accumulate_grad_hook(_make_grad_hook(p))

    if parallel.ddp_level == 3 and parallel.dp_size > 1:
        param_modules = [
            m for m in model.modules() if any(isinstance(p, _ZeROParam) for p in m.parameters(recurse=False))
        ]

        fwd_prefetch: dict[int, list[Callable[[], None]]] = {}
        bwd_prefetch: dict[int, list[Callable[[], None]]] = {}

        def _unshard_module(mod: nn.Module, async_op: bool = False) -> list[Callable[[], None]]:
            handles: list[Callable[[], None]] = []
            for p in mod.parameters(recurse=False):
                if isinstance(p, _ZeROParam) and p._is_sharded:
                    h = p.all_gather_param(async_op=async_op)
                    if h is not None:
                        handles.append(h)
            return handles

        def _reshard_module(mod: nn.Module, only_no_grad: bool = False) -> None:
            for p in mod.parameters(recurse=False):
                if isinstance(p, _ZeROParam) and not p._is_sharded:
                    if not only_no_grad or not p.requires_grad:
                        p.free_param()

        for idx, mod in enumerate(param_modules):
            def _make_hooks(i: int, subm: nn.Module) -> None:
                def _pre_fwd(m: nn.Module, _args: Any) -> None:
                    if i in fwd_prefetch:
                        for fn in fwd_prefetch.pop(i):
                            fn()
                    _unshard_module(m)

                    if i + 1 < len(param_modules) and (i + 1) not in fwd_prefetch:
                        handles = _unshard_module(param_modules[i + 1], async_op=True)
                        if handles:
                            fwd_prefetch[i + 1] = handles

                def _post_fwd(m: nn.Module, _args: Any, _output: Any) -> None:
                    _reshard_module(m)

                def _pre_bwd(m: nn.Module, _grad_output: Any) -> None:
                    if i in bwd_prefetch:
                        for fn in bwd_prefetch.pop(i):
                            fn()
                    _unshard_module(m)

                    if i - 1 >= 0 and (i - 1) not in bwd_prefetch:
                        handles = _unshard_module(param_modules[i - 1], async_op=True)
                        if handles:
                            bwd_prefetch[i - 1] = handles

                def _post_bwd(m: nn.Module, _grad_input: Any, _grad_output: Any) -> None:
                    _reshard_module(m, only_no_grad=True)

                subm.register_forward_pre_hook(_pre_fwd)
                subm.register_forward_hook(_post_fwd)
                subm.register_full_backward_pre_hook(_pre_bwd)
                subm.register_full_backward_hook(_post_bwd)

            _make_hooks(idx, mod)

    return model

def setup_parallel(
    model: nn.Module,
    parallel: ParallelContext,
    tp_plan: dict[str, str] | None = None,
) -> nn.Module:
    model = apply_tp(model, parallel, tp_plan=tp_plan)
    model = apply_ddp(model, parallel)
    return model
