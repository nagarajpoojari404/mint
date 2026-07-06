import importlib
import textwrap
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

    def _generate_greedy(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: set[int],
    ) -> list[int]:
        """Autoregressively generate tokens from prompt.

        Stops when any token in ``stop_token_ids`` is produced or
        ``max_new_tokens`` is reached. The stop token is included in the
        returned list so the caller can detect which condition triggered.
        """
        generated: list[int] = []
        context = prompt.unsqueeze(0).to(self.device.device)  # [1, T]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(context)          # [1, T, vocab]
                next_token = int(logits[0, -1].argmax())
                generated.append(next_token)
                if next_token in stop_token_ids:
                    break
                next_tensor = torch.tensor(
                    [[next_token]], dtype=torch.long, device=self.device.device
                )
                context = torch.cat([context, next_tensor], dim=1)

        return generated

    def _log_sample_predictions(self, step: int, num_samples: int = 3) -> None:
        if not self.is_main_process or not self.tokenizer:
            return

        tok = self.tokenizer

        # Fetch the *full* multi-token boundary sequences so we can locate assistant
        # turns correctly even when the boundary's first token is shared with other
        # roles (e.g. ChatML <|im_start|> prefixes both user and assistant turns).
        if hasattr(tok, "encode_special_ids"):
            asst_start_ids = tok.encode_special_ids("<|assistant_start|>")
            asst_end_ids   = tok.encode_special_ids("<|assistant_end|>")
        else:
            asst_start_ids = [tok.encode_special("<|assistant_start|>")]
            asst_end_ids   = [tok.encode_special("<|assistant_end|>")]

        n_start = len(asst_start_ids)

        # Stop generation at the first token of the end sequence OR eos.
        stop_ids = {asst_end_ids[0] if asst_end_ids else tok.eos_token, tok.eos_token}
        max_new = 150

        def _matches(seq: list[int], pos: int, pattern: list[int]) -> bool:
            return seq[pos : pos + len(pattern)] == pattern

        def _wrap(text: str, width: int = 96) -> str:
            lines = text.splitlines()
            return "\n".join(textwrap.fill(ln, width=width) if ln.strip() else ln for ln in lines)

        sep = "─" * 64
        sep2 = "·" * 64

        samples = self.dataloader.sample(num_samples=num_samples)
        for i, sample in enumerate(samples, 1):
            tokens: list[int] = sample["tokens"].tolist()
            conv = sample["conversation"]

            # ── locate every (assistant_start, assistant_end) span in token stream ──
            # Match the full boundary sequence at each position to avoid false positives
            # from shared leading tokens (e.g. <|im_start|> in ChatML).
            turns: list[tuple[int, int]] = []  # (start_of_asst_start_seq, start_of_asst_end_seq)
            j = 0
            while j < len(tokens):
                if _matches(tokens, j, asst_start_ids):
                    start = j
                    j += n_start
                    while j < len(tokens) and not _matches(tokens, j, asst_end_ids):
                        j += 1
                    end = j  # points at start of asst_end sequence (or past-end)
                    turns.append((start, end))
                else:
                    j += 1

            # ── build conversation lines from the raw dict ──
            messages = conv.get("messages", [])
            # filter out system message (it was prepended into first user turn)
            display_msgs = [m for m in messages if m["role"] != "system"]

            lines: list[str] = [f"\n{sep}", f" Sample {i} / step {step}", sep]

            pred_turn_idx = 0  # which assistant span we are filling
            for msg in display_msgs:
                role = msg["role"]
                content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])

                if role == "user":
                    lines.append("\n  \033[1mUser\033[0m")
                    lines.append(_wrap("  " + content))

                elif role == "assistant":
                    # Ground-truth
                    lines.append("\n  \033[1mAssistant (target)\033[0m")
                    lines.append(_wrap("  " + content))

                    # Predicted: run autoregressive generation from the token prefix
                    # up to and including the full <|assistant_start|> sequence.
                    if pred_turn_idx < len(turns):
                        turn_start, _turn_end = turns[pred_turn_idx]
                        # context = everything up to and including all of asst_start_ids
                        prompt_tokens = torch.tensor(
                            tokens[: turn_start + n_start], dtype=torch.long
                        )
                        generated = self._generate_greedy(prompt_tokens, max_new, stop_ids)
                        # strip trailing stop token if present
                        if generated and generated[-1] in stop_ids:
                            generated = generated[:-1]
                        pred_str = tok.decode(
                            torch.tensor(generated, dtype=torch.long).unsqueeze(0)
                        )[0]
                        lines.append("\n  \033[1mAssistant (pred)\033[0m")
                        lines.append(_wrap("  " + pred_str.strip()))
                        pred_turn_idx += 1

                    lines.append("  " + sep2)

            lines.append(sep)
            logger.debug("\n".join(lines))
