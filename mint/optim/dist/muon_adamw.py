from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

from mint.config.base import Config
from mint.optim.muon_adamw import MuonAdamWConfig
from mint.utils.parallel import _ZeROParam


@dataclass
class AdamConfig(Config):
    adamw_beta1: float = 0.8
    adamw_beta2_lm_head: float = 0.96
    adamw_beta2_embedding: float = 0.995
    adamw_beta2_scalar: float = 0.95
    adamw_eps: float = 1e-10


@dataclass
class MuonConfig(Config):
    muon_momentum: float = 0.95
    muon_beta2: float = 0.9
    muon_ns_steps: int = 5


@dataclass
class DistMuonAdamWConfig(MuonAdamWConfig):
    pass


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class DistMuonAdamW(Optimizer):
    def __init__(self, param_groups: list[dict], group: dist.ProcessGroup | None = None) -> None:
        super().__init__(param_groups, defaults={})
        self.process_group = group

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._fused_adam = torch.compile(
            self._adamw_step_fused,
            fullgraph=True,
            dynamic=True,
        )

        self._fused_muon = torch.compile(
            self._muon_step_fused,
            fullgraph=True,
            dynamic=True,
        )

    @staticmethod
    def _adamw_step_fused(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        step_t: Tensor,
        lr_t: Tensor,
        beta1_t: Tensor,
        beta2_t: Tensor,
        eps_t: Tensor,
        wd_t: Tensor,
    ) -> None:
        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, 1 - beta1_t)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)

        bias1 = 1 - beta1_t**step_t
        bias2 = 1 - beta2_t**step_t

        denom = (exp_avg_sq / bias2).sqrt() + eps_t
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size.item())

    @staticmethod
    def _muon_step_fused(
        grad: Tensor,
        param: Tensor,
        momentum_buffer: Tensor,
        second_momentum_buffer: Tensor,
        momentum_t: Tensor,
        lr_t: Tensor,
        wd_t: Tensor,
        beta2_t: Tensor,
        ns_steps: int,
        red_dim: int,
    ) -> None:
        momentum = momentum_t.to(grad.dtype)
        momentum_buffer.lerp_(grad, 1 - momentum)
        g = grad.lerp(momentum_buffer, momentum)

        if g.ndim >= 2:
            X = g.bfloat16() if g.dtype == torch.bfloat16 else g
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
            if g.size(-2) > g.size(-1):
                for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                    A = X.mT @ X
                    B = b * A + c * (A @ A)
                    X = a * X + X @ B
            else:
                for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                    A = X @ X.mT
                    B = b * A + c * (A @ A)
                    X = a * X + B @ X
            g = X
        else:
            g = g / (g.norm() * 1.01 + 1e-6)

        beta2 = beta2_t.to(g.dtype)
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm_sq = v_mean.sum(dim=-1, keepdim=True) * red_dim_size
        v_norm = v_norm_sq.sqrt()

        second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=-1, keepdim=True).sqrt()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        g = g * final_scale.to(g.dtype)

        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (g * param) >= 0
        param.sub_(lr * g + lr * wd * param * mask)

    def _step_adamw(self, group: dict, gather_list: list) -> None:
        for p in group["params"]:
            if isinstance(p, _ZeROParam):
                if p._ddp_level == 3:
                    grad_slice = p._sliced_grad
                    p_slice = p._sliced_param
                elif p._ddp_level == 2:
                    if p._sliced_grad is None:
                        continue
                    start = p._dp_rank * p._chunk_size
                    end = start + p._chunk_size
                    flat_p = p.data.reshape(-1)
                    p_slice = flat_p[start:end]
                    grad_slice = p._sliced_grad
                elif p._ddp_level == 1:
                    if p.grad is None:
                        continue
                    start = p._dp_rank * p._chunk_size
                    end = start + p._chunk_size
                    flat_p = p.data.reshape(-1)
                    p_slice = flat_p[start:end]
                    flat_g = p.grad.reshape(-1)
                    if p._pad_size > 0:
                        flat_g = torch.nn.functional.pad(flat_g, (0, p._pad_size))
                    grad_slice = flat_g[start:end]
                else:
                    grad_slice = p.grad
                    p_slice = p.data
            else:
                grad_slice = p.grad
                p_slice = p.data

            if grad_slice is None or p_slice is None:
                continue

            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)

            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            self._fused_adam(
                p_slice,
                grad_slice,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

            if isinstance(p, _ZeROParam):
                p._sliced_grad = None
                if p._ddp_level in (1, 2) and p._dp_size > 1:
                    full_p = p.data.reshape(-1)
                    if p._pad_size > 0:
                        full_p = torch.nn.functional.pad(full_p, (0, p._pad_size))
                    work = dist.all_gather_into_tensor(
                        full_p, p_slice, group=p._dp_group, async_op=True
                    )
                    gather_list.append({"work": work, "p": p, "full_p": full_p})

    def _step_muon(self, group: dict, gather_list: list) -> None:
        for p in group["params"]:
            if isinstance(p, _ZeROParam):
                if p._ddp_level == 3:
                    grad_slice = p._sliced_grad
                    p_slice = p._sliced_param
                elif p._ddp_level == 2:
                    if p._sliced_grad is None:
                        continue
                    start = p._dp_rank * p._chunk_size
                    end = start + p._chunk_size
                    flat_p = p.data.reshape(-1)
                    p_slice = flat_p[start:end]
                    grad_slice = p._sliced_grad
                elif p._ddp_level == 1:
                    if p.grad is None:
                        continue
                    start = p._dp_rank * p._chunk_size
                    end = start + p._chunk_size
                    flat_p = p.data.reshape(-1)
                    p_slice = flat_p[start:end]
                    flat_g = p.grad.reshape(-1)
                    if p._pad_size > 0:
                        flat_g = torch.nn.functional.pad(flat_g, (0, p._pad_size))
                    grad_slice = flat_g[start:end]
                else:
                    grad_slice = p.grad.reshape(-1) if p.grad is not None else None
                    p_slice = p.data.reshape(-1)
            else:
                grad_slice = p.grad.reshape(-1) if p.grad is not None else None
                p_slice = p.data.reshape(-1)

            if grad_slice is None or p_slice is None:
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p_slice)
            if "second_momentum_buffer" not in state:
                state["second_momentum_buffer"] = torch.zeros_like(p_slice)

            shape = getattr(p, "_orig_shape", p.shape)
            ratio = max(1.0, shape[-2] / shape[-1]) ** 0.5 if len(shape) >= 2 else 1.0
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_lr_t.fill_(group["lr"] * ratio)
            self._muon_wd_t.fill_(group["weight_decay"])
            self._muon_beta2_t.fill_(group["beta2"])

            self._fused_muon(
                grad_slice,
                p_slice,
                state["momentum_buffer"],
                state["second_momentum_buffer"],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group.get("ns_steps", 5),
                -1,
            )

            if isinstance(p, _ZeROParam):
                p._sliced_grad = None
                if p._ddp_level in (1, 2) and p._dp_size > 1:
                    full_p = p.data.reshape(-1)
                    if p._pad_size > 0:
                        full_p = torch.nn.functional.pad(full_p, (0, p._pad_size))
                    work = dist.all_gather_into_tensor(
                        full_p, p_slice, group=p._dp_group, async_op=True
                    )
                    gather_list.append({"work": work, "p": p, "full_p": full_p})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        gather_list: list[dict[str, Any]] = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group, gather_list)
            elif group["kind"] == "muon":
                self._step_muon(group, gather_list)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        for item in gather_list:
            if item.get("work") is not None:
                item["work"].wait()
            if "p" in item:
                item["p"].data.copy_(
                    item["full_p"][: item["p"]._orig_numel].view(item["p"]._orig_shape)
                )

        return loss
