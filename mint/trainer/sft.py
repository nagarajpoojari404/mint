import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from mint.data.dist.sft import DistributedSFTDataloader
from mint.eval.dist.chatcore import ChatCoreEvaluator
from mint.trainer.base import BasetrainConfig, BaseTrainer
from mint.trainer.scheduler import Scheduler
from mint.utils.checkpointer import Checkpointer
from mint.utils.device import Device
from mint.utils.logger import logger


@dataclass
class SFTConfig(BasetrainConfig): ...


class SFTTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: DistributedSFTDataloader,
        device: Device,
        config: SFTConfig,
        tokenizer: Any = None,  # noqa: ANN401
        eval_datasets: list | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            config=config,
            tokenizer=tokenizer,
        )

        self.checkpointer = Checkpointer(
            config=config.ckpt,
            is_main_process=self.is_main_process,
        )

        self.scheduler = Scheduler(
            num_iterations=config.train_num_steps,
            config=config.sched,
        )

        self.evaluator = None
        if eval_datasets:
            self.evaluator = ChatCoreEvaluator(
                model=model,
                config=config.eval,
                tokenizer=tokenizer,
                device=device,
                datasets=eval_datasets,
            )

            if config.ckpt.resume_from_checkpoint:
                self._load_pretrained_checkpoint()

    def _load_pretrained_checkpoint(self) -> None:
        checkpoint_info = self.checkpointer.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=self.config.ckpt.resume_from_checkpoint,
            load_best=self.config.ckpt.load_best_checkpoint,
        )

        logger.info(
            f"Loaded pretrained checkpoint from step {checkpoint_info.get('step', 'unknown')}"
        )

        # Check for NaN/Inf in model parameters
        nan_params = []
        inf_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

        if nan_params:
            logger.error(f"NaN detected in model parameters after loading checkpoint: {nan_params}")
        if inf_params:
            logger.error(f"Inf detected in model parameters after loading checkpoint: {inf_params}")

        logger.info("Starting SFT training from step 0")

    def _validate_batch(self, inputs: torch.Tensor, targets: torch.Tensor, step: int) -> None:
        if torch.isnan(inputs).any():
            logger.error(f"NaN detected in inputs at step {step}")
        if torch.isinf(inputs).any():
            logger.error(f"Inf detected in inputs at step {step}")

    def _forward_pass(
        self, inputs: torch.Tensor, doc_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.model(inputs)

        # Check for NaN in logits
        if torch.isnan(logits).any():
            logger.warning("NaN detected in logits")
            # Log some statistics about the inputs
            logger.warning(
                f"Input shape: {inputs.shape}, min: {inputs.min()}, "
                f"max: {inputs.max()}, mean: {inputs.float().mean()}"
            )

        return logits

    def _run_evaluation(self, step: int, num_examples: int | None = None) -> None:
        if self.evaluator is None:
            return

        self.model.eval()
        with torch.no_grad():
            results = self.evaluator.evaluate(num_examples=num_examples)
        self.model.train()

        metrics = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    metrics[f"eval/{key}/{sub_key}"] = sub_value
            else:
                metrics[f"eval/{key}"] = value

        if self.is_main_process:
            log_items = " | ".join(
                f"{key}={value:.4f}"
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            )
            logger.info(f"Eval complete | step={step} | {log_items}")

        self._log_wandb(metrics, step=step)

    def train(self) -> None:
        self.model.train()
        train_start_time = time.perf_counter()
        accumulated_loss = 0.0
        micro_step = 0

        batch_iterator = self.dataloader.batch_loader(split="train")
        step_start_time = time.perf_counter()

        for inputs, targets, loss_mask, _ in batch_iterator:
            step = micro_step // self.config.gradient_accumulation_steps

            # Handle interrupt
            if (
                self._handle_interrupt(step, self.checkpointer)
                or step >= self.config.train_num_steps
            ):
                break

            if micro_step % self.config.gradient_accumulation_steps == 0:
                step_start_time = time.perf_counter()
            if micro_step % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            is_accumulating = (micro_step + 1) % self.config.gradient_accumulation_steps != 0
            loss = self._compute_loss(inputs, targets, loss_mask, step)

            self.device.backward(loss)
            accumulated_loss += loss.item()
            micro_step += 1

            if not is_accumulating:
                step = micro_step // self.config.gradient_accumulation_steps

                scheduler_metrics = self.scheduler.step(self.optimizer, step)
                self._perform_optimization_step(micro_step, step, scheduler_metrics)

                step_time = time.perf_counter() - step_start_time
                tokens_per_step = (
                    loss_mask.sum().item()
                    * self.process_info["world_size"]
                    * self.config.gradient_accumulation_steps
                )

                metrics = self._calculate_step_metrics(
                    accumulated_loss,
                    step_time,
                    tokens_per_step,
                    scheduler_metrics,
                    train_start_time,
                )

                if step % self.config.eval_every_n_steps == 0 and step > 0:
                    self._run_evaluation(step=step, num_examples=100)
                    self._log_sample_predictions(step)

                self._maybe_save_checkpoint(step, self.checkpointer)
                self._log_training_progress(step, metrics)

                accumulated_loss = 0.0

        final_step = micro_step // self.config.gradient_accumulation_steps
        self._finalize_training(final_step, self.checkpointer)
