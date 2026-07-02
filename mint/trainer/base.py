import importlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from mint.config.base import Config
from mint.data.dataloader import DistributedDataloader
from mint.eval.base import EvalConfig
from mint.trainer.scheduler import SchedulerConfig
from mint.utils.checkpointer import Checkpointer, CheckpointerConfig
from mint.utils.device import Device
from mint.utils.logger import LoggerConfig, logger


@dataclass
class BasetrainConfig(Config):
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    use_meta_device: bool = True
    compile_model: bool = True

    train_num_steps: int = 1000
    grad_clip: float = 1.0
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    eval_num_steps: int = 10
    core_eval_every_n_step: int = 500
    gradient_accumulation_steps: int = 1

    ckpt: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    lg: LoggerConfig = field(default_factory=LoggerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: DistributedDataloader,
        device: Device,
        config: BasetrainConfig,
        tokenizer: Any = None,  # noqa: ANN401
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
        self.process_info = self.device.process_info()
        self.is_main_process = self.process_info["is_main"]
        self.wandb_run = self._init_wandb()

    def _init_wandb(self) -> Any | None:  # noqa: ANN401
        if not self.config.lg.wandb_enabled or not self.is_main_process:
            return None

        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is enabled in config but the package is not installed.")

        wandb = __import__("wandb")

        run = wandb.init(
            project=self.config.lg.wandb_project,
            name=self.config.lg.wandb_run_name,
            entity=self.config.lg.wandb_entity,
            config={
                key: str(value) if isinstance(value, torch.dtype) else value
                for key, value in self.config.__dict__.items()
            },
        )
        logger.info(f"W&B initialized | project={self.config.lg.wandb_project} | run={run.name}")
        return run

    def _log_wandb(self, metrics: dict, step: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _get_dataloader_state(self) -> dict:
        return {}

    def _validate_batch(self, inputs: torch.Tensor, targets: torch.Tensor, step: int) -> None:
        pass

    def _forward_pass(
        self, inputs: torch.Tensor, doc_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if doc_ids is not None:
            return self.model(inputs, doc_ids=doc_ids)
        return self.model(inputs)

    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        step: int,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_batch(inputs, targets, step)

        with self.device.autocast():
            logits = self._forward_pass(inputs, doc_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )

            mask_sum = loss_mask.sum()
            if mask_sum > 0:
                masked_loss = (loss * loss_mask.view(-1).float()).sum()
                loss = masked_loss / mask_sum.float()
            else:
                logger.warning(f"Zero mask sum at step {step}, skipping batch")
                loss = torch.tensor(0.0, device=self.device.device, dtype=torch.float32)

            return loss / self.config.gradient_accumulation_steps

    def _perform_optimization_step(
        self, _micro_step: int, step: int, scheduler_metrics: dict | None = None
    ) -> dict:
        if scheduler_metrics is None:
            if not hasattr(self, "scheduler"):
                raise AttributeError("Scheduler not initialized")
            scheduler_metrics = self.scheduler.step(self.optimizer, step)

        self.device.optimizer_step(
            self.optimizer,
            grad_clip=self.config.grad_clip,
            params=self.model.parameters(),
        )
        self.device.synchronize()

        return scheduler_metrics

    def _calculate_step_metrics(
        self,
        accumulated_loss: float,
        step_time: float,
        tokens_count: int,
        scheduler_metrics: dict,
        train_start_time: float,
    ) -> dict:
        tokens_per_second = tokens_count / step_time if step_time > 0 else 0.0
        elapsed_time = time.perf_counter() - train_start_time

        return {
            "train/loss": accumulated_loss,
            "train/step_time_sec": step_time,
            "train/tokens_per_step": tokens_count,
            "train/tokens_per_second": tokens_per_second,
            "train/elapsed_time_sec": elapsed_time,
            "scheduler/lr_multiplier": scheduler_metrics["lr_multiplier"],
            "scheduler/muon_momentum": scheduler_metrics["muon_momentum"],
            "scheduler/muon_weight_decay": scheduler_metrics["muon_weight_decay"],
        }

    def _handle_interrupt(self, step: int, checkpointer: Any) -> bool:  # noqa: ANN401
        if checkpointer.interrupt_requested:
            if self.is_main_process:
                logger.warning("Saving checkpoint due to keyboard interrupt...")
                checkpointer.save_checkpoint(
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=self._get_dataloader_state(),
                    force=True,
                )
                logger.info("Checkpoint saved. Exiting gracefully.")
            return True
        return False

    def _log_training_progress(self, step: int, metrics: dict) -> None:
        if step % self.config.log_every_n_steps == 0:
            logger.info(
                f"Step {step:4d} | "
                f"Loss: {metrics['train/loss']:.4f} | "
                f"Step Time: {metrics['train/step_time_sec']:.4f}s | "
                f"Tokens/s: {metrics['train/tokens_per_second']:.2f} | "
                f"Memory: {self.device.memory()}"
            )
            self._log_wandb({**metrics, "train/memory": self.device.memory()}, step=step)

    def _maybe_save_checkpoint(
        self,
        step: int,
        checkpointer: Checkpointer,
        *,
        force: bool = False,
    ) -> None:
        if not self.is_main_process:
            return

        if force or (
            self.config.ckpt.save_checkpoint_every_n_steps is not None
            and step % self.config.ckpt.save_checkpoint_every_n_steps == 0
            and step > 0
        ):
            checkpointer.save_checkpoint(
                step=step,
                model=self.model,
                optimizer=self.optimizer,
                dataloader_state=self._get_dataloader_state(),
                force=force,
            )

    def _finalize_training(self, step: int, checkpointer: Any) -> None:  # noqa: ANN401
        if self.is_main_process:
            logger.info("Saving final checkpoint")
            checkpointer.save_checkpoint(
                step=step,
                model=self.model,
                optimizer=self.optimizer,
                dataloader_state=self._get_dataloader_state(),
                force=True,
            )

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _log_sample_predictions(self, step: int, num_samples: int = 3) -> None:
        if not self.is_main_process or not self.tokenizer:
            return

        samples = self.dataloader.sample(num_samples=num_samples)
        for i, sample in enumerate(samples, 1):
            with torch.no_grad():
                input_tensor = sample["input_tokens"].unsqueeze(0).to(self.device.device)
                logits = self.model(input_tensor)
                pred_tokens = logits.argmax(dim=-1).squeeze(0)
                pred_str = self.tokenizer.decode(pred_tokens.unsqueeze(0))[0]

            logger.debug(f"Sample {i}:")
            logger.debug(f"Input:  ...{sample['input_str'][-100:]}")
            logger.debug(f"Target: ...{sample['target_str'][-100:]}")
            logger.debug(f"Pred:   ...{pred_str[-100:]}")
