import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from mint.config.base import Config
from mint.utils.logger import logger


@dataclass
class CheckpointerConfig(Config):
    checkpoint_dir: str = "checkpoints"
    save_checkpoint_every_n_steps: int | None = 200
    keep_last_n_checkpoints: int = 1
    resume_from_checkpoint: str | None = ""
    load_best_checkpoint: bool = True


class Checkpointer:
    def __init__(self, config: CheckpointerConfig, *, is_main_process: bool) -> None:

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.save_every_n_steps = config.save_checkpoint_every_n_steps
        self.keep_last_n = config.keep_last_n_checkpoints
        self.is_main_process = is_main_process

        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.interrupt_requested = False

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:  # noqa: ANN001, ARG002
        logger.warning("\n⚠️  Keyboard interrupt received! Saving checkpoint before exit...")
        self.interrupt_requested = True

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader_state: dict[str, Any],
        val_loss: float | None = None,
        *,
        is_best: bool = False,
        force: bool = False,
    ) -> Path | None:

        if not self.is_main_process:
            return None

        should_save = force or is_best
        if not should_save and self.save_every_n_steps is not None:
            should_save = step % self.save_every_n_steps == 0

        if not should_save:
            return None

        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dataloader_state": dataloader_state,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }

        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f" Saved latest checkpoint at step {step} to {latest_path}")

        step_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, step_path)
        logger.info(f" Saved checkpoint at step {step} to {step_path}")

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f" Saved best checkpoint (val_loss={val_loss:.4f}) to {best_path}")

        self._cleanup_old_checkpoints()
        self._save_metadata(step, val_loss)

        return step_path

    def _cleanup_old_checkpoints(self) -> None:
        step_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if len(step_checkpoints) > self.keep_last_n:
            for old_checkpoint in step_checkpoints[: -self.keep_last_n]:
                old_checkpoint.unlink()
                logger.debug(f"🗑️  Removed old checkpoint: {old_checkpoint.name}")

    def _save_metadata(self, step: int, val_loss: float | None) -> None:
        metadata = {
            "last_step": step,
            "last_val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }

        metadata_path = self.checkpoint_dir / "metadata.json"
        with Path(metadata_path).open("w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_model(  # noqa: ANN206
        cls,
        model: nn.Module,
        checkpoint_path: str | None = None,
    ):
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        else:
            raise ValueError("invalid checkpoint path")
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return {"step": 0, "dataloader_state": {}}

        logger.info(f"oading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu")

        state_dict = checkpoint["model_state_dict"]
        # Strip the LogitsWrapper prefix only when loading into a bare model
        # (e.g. chat/inference).  During training the model IS a LogitsWrapper,
        # so its state_dict already expects "_model.*" keys — leave them intact.
        if not hasattr(model, "_model") and any(key.startswith("_model.") for key in state_dict):
            state_dict = {
                key.removeprefix("_model."): value for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)

        logger.info(
            f"Loaded checkpoint from step {checkpoint['step']} "
            f"(val_loss={checkpoint.get('val_loss', 'N/A')})"
        )

        return {
            "step": checkpoint["step"],
            "dataloader_state": checkpoint.get("dataloader_state", {}),
            "val_loss": checkpoint.get("val_loss"),
        }

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        checkpoint_path: str | None = None,
        *,
        load_best: bool = False,
    ) -> dict[str, Any]:

        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.checkpoint_dir / "checkpoint_best.pt"
        else:
            path = self.checkpoint_dir / "checkpoint_latest.pt"

        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return {"step": 0, "dataloader_state": {}}

        logger.info(f"oading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu")

        state_dict = checkpoint["model_state_dict"]
        # Strip the LogitsWrapper prefix only when loading into a bare model
        # (e.g. chat/inference).  During training the model IS a LogitsWrapper,
        # so its state_dict already expects "_model.*" keys — leave them intact.
        if not hasattr(model, "_model") and any(key.startswith("_model.") for key in state_dict):
            state_dict = {
                key.removeprefix("_model."): value for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(
            f"Loaded checkpoint from step {checkpoint['step']} "
            f"(val_loss={checkpoint.get('val_loss', 'N/A')})"
        )

        return {
            "step": checkpoint["step"],
            "dataloader_state": checkpoint.get("dataloader_state", {}),
            "val_loss": checkpoint.get("val_loss"),
        }

    def should_checkpoint_on_eval(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False

    def get_resume_info(self) -> dict[str, Any]:
        metadata_path = self.checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with Path(metadata_path).open() as f:
                return json.load(f)
        return {}
