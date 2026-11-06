import os
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from mint.config.base import Config
from mint.utils.logger import logger


@dataclass
class DeviceConfig(Config):
    mixed_precision: bool = True


class Device:
    def __init__(self, config: DeviceConfig) -> None:
        self._init_distributed()
        self.device = self._detect()
        self.type = self.device.type

        self.is_cuda = self.type == "cuda"
        self.is_mps = self.type == "mps"
        self.is_cpu = self.type == "cpu"

        self.use_amp = self.is_cuda and config.mixed_precision
        self.amp_dtype = self._get_amp_dtype()
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        self._setup(config)
        logger.warning(f"training with autocast: {self.use_amp} with {self.amp_dtype}")

    @staticmethod
    def _init_distributed() -> None:
        # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; plain `python` runs stay single-process.
        if int(os.getenv("WORLD_SIZE", "1")) <= 1 or dist.is_initialized():
            return
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    def _detect(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_amp_dtype(self) -> torch.dtype | None:
        if not self.use_amp:
            return None
        # bf16 is only native on Ampere+ (sm80); Turing/Volta emulate it slowly -> use fp16 + GradScaler
        if self.is_cuda and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16

    def _setup(self, _config: Config) -> None:
        if self.is_mps:
            logger.info("MPS device active (fp32)")
        elif self.is_cuda:
            amp_dtype = (
                str(self.amp_dtype).replace("torch.", "")
                if self.amp_dtype is not None
                else "disabled"
            )
            logger.info(f"CUDA device active | AMP={self.use_amp} | dtype={amp_dtype}")
        else:
            logger.info("CPU device active")

    def compile_model(self, model: nn.Module, *, enabled: bool = True) -> nn.Module:
        if not enabled or self.is_cpu:
            return model
        try:
            if self.is_cuda:
                logger.info("Compiling model with torch.compile...")
                return cast("nn.Module", torch.compile(model))
            if self.is_mps:
                logger.info("Compiling model with torch.compile (aot_eager backend)...")
                return cast("nn.Module", torch.compile(model, backend="aot_eager"))
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            return model
        return model

    def move_to_device(self, model: nn.Module, *, from_meta: bool = False) -> nn.Module:
        logger.info(f"Moving model to {self.device}...")

        if from_meta:
            logger.info("Materializing model from meta device...")
            model = model.to_empty(device=self.device)

            def _reset(m: nn.Module) -> None:
                reset_fn = getattr(m, "reset_parameters", None)
                if callable(reset_fn):
                    reset_fn()

            model.apply(_reset)
        else:
            model = model.to(self.device)

        return model

    def autocast(self) -> torch.autocast:
        return autocast(
            device_type="cuda" if self.is_cuda else "cpu",
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

    def to_device(self, *tensors: torch.Tensor, non_blocking: bool = True):  # noqa: ANN201
        if self.is_cuda:
            return [t.to(self.device, non_blocking=non_blocking) for t in tensors]
        return [t.to(self.device) for t in tensors]

    def backward(self, loss: torch.Tensor) -> None:
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 0.0,
        params=None,  # noqa: ANN001
    ) -> None:
        if self.use_amp:
            if grad_clip > 0:
                if params is None:
                    raise ValueError("params required for grad clipping")
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if grad_clip > 0:
                if params is None:
                    raise ValueError("params required for grad clipping")
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            optimizer.step()

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_mps:
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()

    def memory(self) -> str:
        if self.is_cuda:
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved() / 1024**3
            return f"{a:.2f}GB / {r:.2f}GB"
        if self.is_mps:
            try:
                a = torch.mps.current_allocated_memory() / 1024**3
            except Exception:
                return ""
            else:
                return f"{a:.2f}GB"
        return ""

    def process_info(self) -> dict:
        dist_on = dist.is_available() and dist.is_initialized()

        rank = dist.get_rank() if dist_on else int(os.getenv("RANK", "0"))
        world = dist.get_world_size() if dist_on else int(os.getenv("WORLD_SIZE", "1"))
        local = int(os.getenv("LOCAL_RANK", rank))

        name = f"{self.type}:{local}" if local != rank else str(self.device)

        return {
            "rank": rank,
            "local_rank": local,
            "world_size": world,
            "name": name,
            "is_main": rank == 0,
            "distributed": dist_on,
        }

    def __str__(self) -> str:
        return str(self.device)
