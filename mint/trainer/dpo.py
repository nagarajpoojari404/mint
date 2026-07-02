import textwrap
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from mint.data.dist.dpo import DistributedDPODataloader
from mint.eval.dist.chatcore import ChatCoreEvaluator
from mint.loss.dpo import DPOLoss, compute_logps
from mint.trainer.base import BasetrainConfig, BaseTrainer
from mint.trainer.scheduler import Scheduler
from mint.utils.checkpointer import Checkpointer
from mint.utils.device import Device
from mint.utils.logger import logger


@dataclass
class DPOConfig(BasetrainConfig): ...


class DPOTrainer(BaseTrainer):
    def __init__(
        self,
        policy_model: nn.Module,
        optimizer: Optimizer,
        dataloader: DistributedDPODataloader,
        device: Device,
        config: DPOConfig,
        reference_model: nn.Module,
        tokenizer: Any = None,  # noqa: ANN401
        eval_datasets: list | None = None,
    ) -> None:
        super().__init__(
            model=policy_model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            config=config,
            tokenizer=tokenizer,
        )

        self.dataloader: DistributedDPODataloader = dataloader
        # model or policy model both mean same, the one which is
        # being trained
        self.policy_model = policy_model
        self.reference_model = reference_model

        self.checkpointer = Checkpointer(
            config=config.ckpt,
            is_main_process=self.is_main_process,
        )

        # Initialize scheduler only if config has sched attribute
        self.scheduler = None
        if hasattr(config, "sched") and config.sched is not None:
            self.scheduler = Scheduler(
                num_iterations=config.train_num_steps,
                config=config.sched,
            )
            # Set initial_lr in optimizer param groups for scheduler
            for group in optimizer.param_groups:
                group["initial_lr"] = group["lr"]

        self.criterion = DPOLoss()

        self.evaluator = None
        if eval_datasets:
            self.evaluator = ChatCoreEvaluator(
                model=policy_model,
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

    def _validate_batch(self, inputs: torch.Tensor, targets: torch.Tensor, step: int) -> None: ...

    def _forward_pass(
        self, inputs: torch.Tensor, doc_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(inputs)

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

    def _log_sample_predictions(self, step: int, num_samples: int = 3) -> None:
        if not self.is_main_process or not self.tokenizer:
            return

        samples = self.dataloader.sample(num_samples=num_samples)
        for i, sample in enumerate(samples, 1):
            with torch.no_grad():
                chosen_tensor = sample["chosen_tokens"].unsqueeze(0).to(self.device.device)

                policy_output = self.policy_model(chosen_tensor)
                policy_logits = (
                    policy_output.logits if hasattr(policy_output, "logits") else policy_output
                )
                policy_pred_tokens = policy_logits.argmax(dim=-1).squeeze(0)
                policy_pred_str = self.tokenizer.decode(
                    policy_pred_tokens.unsqueeze(0), skip_special_tokens=True
                )[0]

                ref_output = self.reference_model(chosen_tensor)
                ref_logits = ref_output.logits if hasattr(ref_output, "logits") else ref_output
                ref_pred_tokens = ref_logits.argmax(dim=-1).squeeze(0)
                ref_pred_str = self.tokenizer.decode(
                    ref_pred_tokens.unsqueeze(0), skip_special_tokens=True
                )[0]

            logger.debug(f"\n{'=' * 80}")
            logger.debug(f"DPO Sample {i} (Step {step}):")
            logger.debug(f"{'-' * 80}")

            logger.debug("CHOSEN:")
            for line in textwrap.wrap(sample["chosen_str"], width=76):
                logger.debug(f"  {line}")

            logger.debug(f"\n{'-' * 80}")
            logger.debug("REJECTED:")
            for line in textwrap.wrap(sample["rejected_str"], width=76):
                logger.debug(f"  {line}")

            logger.debug(f"\n{'-' * 80}")
            logger.debug("POLICY MODEL PREDICTION:")
            for line in textwrap.wrap(policy_pred_str, width=76):
                logger.debug(f"  {line}")

            logger.debug(f"\n{'-' * 80}")
            logger.debug("REFERENCE MODEL PREDICTION:")
            for line in textwrap.wrap(ref_pred_str, width=76):
                logger.debug(f"  {line}")

            logger.debug(f"{'=' * 80}\n")

    def _compute_loss(
        self,
        concat_ids: torch.Tensor,
        concat_mask: torch.Tensor,
        c_labels: torch.Tensor,
        r_labels: torch.Tensor,
    ) -> torch.Tensor:
        policy_outputs = self.policy_model(concat_ids)
        # Extract logits from the model output (HuggingFace models return CausalLMOutputWithPast)
        policy_logits = (
            policy_outputs.logits if hasattr(policy_outputs, "logits") else policy_outputs
        )

        # split back into chosen and rejected chunks
        p_chosen_logits, p_rejected_logits = policy_logits.chunk(2, dim=0)

        p_chosen_logps = compute_logps(p_chosen_logits, c_labels)
        p_rejected_logps = compute_logps(p_rejected_logits, r_labels)

        # forward pass through reference model (No grads needed)
        with torch.no_grad():
            ref_outputs = self.reference_model(concat_ids)
            # Extract logits from the model output
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs

            r_chosen_logits, r_rejected_logits = ref_logits.chunk(2, dim=0)

            ref_chosen_logps = compute_logps(r_chosen_logits, c_labels)
            ref_rejected_logps = compute_logps(r_rejected_logits, r_labels)

        # evaluate DPO Engine loss metrics
        loss, metrics = self.criterion(
            p_chosen_logps, p_rejected_logps, ref_chosen_logps, ref_rejected_logps
        )
        return loss / self.config.gradient_accumulation_steps, metrics

    def train(self) -> None:
        self.policy_model.train()
        self.reference_model.eval()

        train_start_time = time.perf_counter()
        accumulated_loss = 0.0
        micro_step = 0

        batch_iterator = self.dataloader.batch_loader(split="train")
        step_start_time = time.perf_counter()

        for batch in batch_iterator:
            # Calculate current step for interrupt check (before increment)
            step = micro_step // self.config.gradient_accumulation_steps

            if (
                self._handle_interrupt(step, self.checkpointer)
                or step >= self.config.train_num_steps
            ):
                break

            if micro_step % self.config.gradient_accumulation_steps == 0:
                step_start_time = time.perf_counter()
            if micro_step % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # prepare
            c_ids = batch.chosen_input_ids
            c_mask = batch.chosen_attn_mask
            c_labels = batch.chosen_labels

            r_ids = batch.rejected_input_ids
            r_mask = batch.rejected_attn_mask
            r_labels = batch.rejected_labels

            # Concatenate inputs batchwise to optimize forward pass operations
            concat_ids = torch.cat([c_ids, r_ids], dim=0)
            concat_mask = torch.cat([c_mask, r_mask], dim=0)

            is_accumulating = (micro_step + 1) % self.config.gradient_accumulation_steps != 0

            loss, metrics = self._compute_loss(concat_ids, concat_mask, c_labels, r_labels)

            self.device.backward(loss)
            accumulated_loss += loss.item()
            micro_step += 1

            if not is_accumulating:
                # Calculate step after completing gradient accumulation
                step = micro_step // self.config.gradient_accumulation_steps

                # Use scheduler if available, otherwise use empty metrics
                scheduler_metrics = {}
                if self.scheduler is not None:
                    scheduler_metrics = self.scheduler.step(self.optimizer, step)

                self._perform_optimization_step(micro_step, step, scheduler_metrics)

                step_time = time.perf_counter() - step_start_time

                # this is over estimate because we are also counting PAD tokens
                tokens_per_step = (
                    concat_ids.numel()
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

        # Calculate final step for finalization
        final_step = micro_step // self.config.gradient_accumulation_steps
        self._finalize_training(final_step, self.checkpointer)
