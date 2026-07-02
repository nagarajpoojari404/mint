import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from mint.data.dataloader import DistributedDataloader
from mint.eval.base import Evaluator
from mint.eval.bpb import BPBEvaluator
from mint.eval.dist.core import CoreEvaluator
from mint.trainer.base import BasetrainConfig, BaseTrainer
from mint.trainer.scheduler import Scheduler
from mint.utils.checkpointer import Checkpointer
from mint.utils.device import Device
from mint.utils.logger import logger


@dataclass
class PretrainConfig(BasetrainConfig): ...


class PreTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: DistributedDataloader,
        device: Device,
        config: PretrainConfig,
        tokenizer: Any = None,  # noqa: ANN401
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

        self.core_evaluator = CoreEvaluator(
            model=model,
            config=config.eval,
            tokenizer=tokenizer,
            device=device,
        )

        self.bpb_evaluator = BPBEvaluator(
            model=model,
            config=config.eval,
            device=device,
            dataloader=dataloader,
        )

        self.start_step = 0
        if config.ckpt.resume_from_checkpoint:
            self._load_pretrained_checkpoint()

    def _load_pretrained_checkpoint(self) -> None:
        checkpoint_info = self.checkpointer.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=self.config.ckpt.resume_from_checkpoint,
            load_best=self.config.ckpt.load_best_checkpoint,
        )

        self.start_step = checkpoint_info["step"]
        self.dataloader_state = self._extract_rank_dataloader_state(
            checkpoint_info.get("dataloader_state", {})
        )

        if self.dataloader_state and hasattr(self.dataloader, "set_state"):
            self.dataloader.set_state(self.dataloader_state)
            logger.info(
                f"Restored dataloader state: "
                f"shard={self.dataloader_state.get('shard_idx', 0)}, "
                f"rg={self.dataloader_state.get('rg_idx', 0)}, "
                f"batches={self.dataloader_state.get('batches_consumed', 0)}"
            )

        logger.info(f"Resuming training from step {self.start_step}")

    def _local_dataloader_state(self) -> dict:
        if hasattr(self.dataloader, "get_state"):
            return self.dataloader.get_state()
        return {}

    # executed by main process only
    # collect all local dataloader state & save full checkpoint
    def _get_dataloader_state(self) -> dict:
        local_state = self._local_dataloader_state()
        if not self.process_info["distributed"]:
            return local_state

        if self.is_main_process:
            gathered_states: list[dict] = [{} for _ in range(self.process_info["world_size"])]
            dist.gather_object(local_state, gathered_states, dst=0)
            return {"per_rank": dict(enumerate(gathered_states))}

        dist.gather_object(local_state, None, dst=0)
        return {}

    def _extract_rank_dataloader_state(self, checkpoint_dataloader_state: dict) -> dict:
        if not checkpoint_dataloader_state:
            return {}

        if not self.process_info["distributed"]:
            return checkpoint_dataloader_state

        per_rank_state = checkpoint_dataloader_state.get("per_rank", {})
        return per_rank_state.get(self.process_info["rank"], {})

    def _run_evaluation(
        self,
        evaluator: Evaluator,
        step: int,
        metric_prefix: str,
        *,
        save_checkpoint: bool = False,
        **eval_kwargs,  # noqa: ANN003
    ) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            results = evaluator.evaluate(**eval_kwargs)
        self.model.train()

        metrics = {}

        def flatten(prefix: str, value: Any) -> None:  # noqa: ANN401
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flatten(f"{prefix}/{sub_key}", sub_value)
            else:
                metrics[prefix] = value

        for key, value in results.items():
            flatten(f"{metric_prefix}/{key}", value)

        if self.is_main_process:
            log_items = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            logger.info(f"Eval complete | step={step} | {log_items}")

        self._log_wandb(metrics, step=step)

        if self.is_main_process and results and save_checkpoint:
            val_loss = results.get("loss")
            if val_loss is not None:
                is_best = self.checkpointer.should_checkpoint_on_eval(val_loss)
                self.checkpointer.save_checkpoint(
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=self._get_dataloader_state(),
                    val_loss=val_loss,
                    is_best=is_best,
                )

        return results

    def train(self) -> None:
        self.model.train()
        train_start_time = time.perf_counter()
        accumulated_loss = 0.0

        # Calculate starting micro_step from start_step
        micro_step = self.start_step * self.config.gradient_accumulation_steps

        resume_state = getattr(self, "dataloader_state", None)

        batch_iterator = self.dataloader.batch_loader(split="train", resume_state=resume_state)

        step_start_time = time.perf_counter()

        for inputs, targets, loss_mask, doc_ids in batch_iterator:
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

            is_accumulating = (micro_step + 1) % self.config.gradient_accumulation_steps != 0
            loss = self._compute_loss(inputs, targets, loss_mask, step, doc_ids)

            self.device.backward(loss)
            accumulated_loss += loss.item()
            micro_step += 1

            if not is_accumulating:
                # Calculate step after completing gradient accumulation
                step = micro_step // self.config.gradient_accumulation_steps
                scheduler_metrics = self.scheduler.step(self.optimizer, step)
                self._perform_optimization_step(micro_step, step, scheduler_metrics)

                # Clear cache periodically to reduce fragmentation
                if step % 10 == 0:
                    self.device.empty_cache()

                step_time = time.perf_counter() - step_start_time
                tokens_per_step = (
                    targets.numel()
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
                    self._run_evaluation(
                        evaluator=self.bpb_evaluator,
                        step=step,
                        metric_prefix="val",
                        save_checkpoint=True,
                        num_steps=self.config.eval_num_steps,
                    )
                    self._log_sample_predictions(step)

                if step % self.config.core_eval_every_n_step == 0 and step > 0:
                    self._run_evaluation(
                        evaluator=self.core_evaluator,
                        step=step,
                        metric_prefix="core",
                        task_labels=["hellaswag_zeroshot"],
                        limit=1,
                    )

                self._maybe_save_checkpoint(step, self.checkpointer)
                self._log_training_progress(step, metrics)

                accumulated_loss = 0.0

        final_step = micro_step // self.config.gradient_accumulation_steps
        self._finalize_training(final_step, self.checkpointer)
