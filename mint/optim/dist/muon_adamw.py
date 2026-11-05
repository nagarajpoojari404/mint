from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

from mint.config.base import Config


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
class MuonAdamWConfig(AdamConfig, MuonConfig):
    pass


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class DistMuonAdamW(Optimizer):
    def __init__(self, param_groups: list[dict]) -> None:
        super().__init__(param_groups, defaults={})

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
            dynamic=False,
        )

        self._fused_muon = torch.compile(
            self._muon_step_fused,
            fullgraph=True,
            dynamic=False,
        )

    @staticmethod
    def _adamw_step_fused(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,  # First moment (same shape as p)
        exp_avg_sq: Tensor,  # Second moment (same shape as p)
        step_t: Tensor,  # 0-D CPU tensor: step count
        lr_t: Tensor,  # 0-D CPU tensor: learning rate
        beta1_t: Tensor,  # 0-D CPU tensor: beta1
        beta2_t: Tensor,  # 0-D CPU tensor: beta2
        eps_t: Tensor,  # 0-D CPU tensor: epsilon
        wd_t: Tensor,  # 0-D CPU tensor: weight decay
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
        stacked_grads: Tensor,  # (N, H, W) - stacked gradients
        stacked_params: Tensor,  # (N, H, W) - stacked parameters
        momentum_buffer: Tensor,  # (N, H, W) - momentum buffer
        second_momentum_buffer: Tensor,  # (N, H, 1) or (N, 1, W) - variance buffer
        momentum_t: Tensor,  # 0-D CPU tensor: momentum coefficient
        lr_t: Tensor,  # 0-D CPU tensor: learning rate
        wd_t: Tensor,  # 0-D CPU tensor: weight decay
        beta2_t: Tensor,  # 0-D CPU tensor: beta2 for variance
        ns_steps: int,  # Number of Newton-Schulz iterations
        red_dim: int,  # Reduction dimension (-1 or -2)
    ) -> None:

        # Nesterov momentum
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        # Polar Express orthogonalization
        # Use bfloat16 for speed when available (fp16 is unstable here)
        X = g.bfloat16() if g.dtype == torch.bfloat16 else g
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

        if g.size(-2) > g.size(-1):  # Tall matrix
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:  # Wide matrix
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

        # Variance reduction (NorMuon)
        beta2 = beta2_t.to(g.dtype)
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
        v_norm = v_norm_sq.sqrt()

        second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        g = g * final_scale.to(g.dtype)

        # Cautious weight decay + parameter update
        # Only apply weight decay when gradient aligns with parameter
        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if p.numel() < 1024:
                work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
                future = work.get_future() if work is not None else None  # type: ignore
                param_infos[p] = {"future": future, "grad_slice": grad, "is_small": True}
            else:
                assert grad.shape[0] % world_size == 0, (
                    f"AdamW requires shape[0] ({grad.shape[0]}) "
                    f"divisible by world_size ({world_size})"
                )
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                work = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                )
                future = work.get_future() if work is not None else None  # type: ignore
                param_infos[p] = {"future": future, "grad_slice": grad_slice, "is_small": False}

        return {"param_infos": param_infos}

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        params = group["params"]
        if not params:
            return {"future": None}

        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack gradients and zero-pad
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[: len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params) :].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        work = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        )
        future = work.get_future() if work is not None else None  # type: ignore

        return {
            "future": future,
            "grad_chunk": grad_chunk,
            "stacked_grads": stacked_grads,
            "chunk_size": chunk_size,
        }

    def _compute_adamw(
        self, group: dict, info: dict, gather_list: list, rank: int, world_size: int
    ) -> None:
        param_infos = info["param_infos"]
        for p in group["params"]:
            if p not in param_infos:
                continue

            pinfo = param_infos[p]
            pinfo["future"].wait()
            grad_slice = pinfo["grad_slice"]
            state = self.state[p]

            # Determine slice to update
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]

            # Lazy state initialization
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)

            state["step"] += 1

            # Fill 0-D tensors and run fused kernel
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

            # Large params need all_gather
            if not pinfo["is_small"]:
                work = dist.all_gather_into_tensor(p, p_slice, async_op=True)
                future = work.get_future() if work is not None else None  # type: ignore
                gather_list.append({"future": future, "params": None})

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        if info["future"] is None:
            return

        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"]
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)

        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Prepare output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])

            self._fused_muon(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        work = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True)
        future = work.get_future() if work is not None else None  # type: ignore
        gather_list.append({"future": future, "stacked_params": stacked_params, "params": params})

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                for p, src in zip(
                    info["params"],
                    info["stacked_params"][: len(info["params"])].unbind(0),
                    strict=False,
                ):
                    p.copy_(src)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon":
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos, strict=False):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon":
                self._compute_muon(group, info, gather_list, rank)

        self._finish_gathers(gather_list)
        return loss
