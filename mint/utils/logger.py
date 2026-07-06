import sys
from dataclasses import dataclass

from loguru import logger

from mint.config.base import Config


logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)


@dataclass
class LoggerConfig(Config):
    wandb_enabled: bool = True
    wandb_project: str = "goda"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
