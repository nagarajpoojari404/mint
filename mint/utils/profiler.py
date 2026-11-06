from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.profiler as tp

from mint.config.base import Config
from mint.utils.logger import logger


@dataclass
class ProfilerConfig(Config):
    enabled: bool = False

    output_dir: str = "./profiles"

    schedule_wait: int = 1
    schedule_warmup: int = 1
    schedule_active: int = 3
    schedule_repeat: int = 0   # 0 = run once then stop

    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False   # stack traces (expensive - keep off by default)
    with_flops: bool = True

    activities: list[str] = field(default_factory=lambda: ["cpu", "cuda"])

    trace_prefix: str = "trace"
    run_name: str = ""   # traces go to <output_dir>/<run_name>; empty = timestamp


class TorchProfiler:
    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config or ProfilerConfig()
        self._profiler: tp.profile | None = None

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._profiler = self._build()
        self._profiler.start()
        logger.info(f"[profiler] started -> traces will be written to '{self.config.output_dir}'")

    def stop(self) -> None:
        if self._profiler is None:
            return
        self._profiler.stop()
        logger.info("[profiler] stopped")
        self._profiler = None

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def __enter__(self) -> TorchProfiler:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _build(self) -> tp.profile:
        activities = []
        for name in self.config.activities:
            act = name.strip().lower()
            if act == "cpu":
                activities.append(tp.ProfilerActivity.CPU)
            elif act == "cuda" and torch.cuda.is_available():
                activities.append(tp.ProfilerActivity.CUDA)

        run_name = self.config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(self.config.output_dir) / run_name
        out.mkdir(parents=True, exist_ok=True)

        return tp.profile(
            activities=activities,
            schedule=tp.schedule(
                wait=self.config.schedule_wait,
                warmup=self.config.schedule_warmup,
                active=self.config.schedule_active,
                repeat=self.config.schedule_repeat,
            ),
            on_trace_ready=tp.tensorboard_trace_handler(
                str(out), worker_name=f"{self.config.trace_prefix}_rank{os.getenv('RANK', '0')}"
            ),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
        )
