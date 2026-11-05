import sys
from dataclasses import dataclass

from loguru import logger

from mint.config.base import Config


__all__ = ["LoggerConfig", "logger"]


logger.remove()

log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stderr,
    format=log_format,
    level="INFO",
    colorize=True,
)


@dataclass
class LoggerConfig(Config):
    wandb_enabled: bool = True
    wandb_project: str = "goda"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
