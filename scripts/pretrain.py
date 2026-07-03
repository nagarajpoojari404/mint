import argparse
from dataclasses import dataclass

import torch
from torch import nn

from mint.config.base import Config
from mint.data.dataloader import DataloaderConfig, DistributedDataloader
from mint.data.dist.pretrain import DistributedBOSBestfitPretrainDataloader
from mint.nn.models import Gemma, GemmaConfig, configure_optimizer
from mint.optim.muon_adamw import MuonAdamWConfig
from mint.tokenizer import TikTokenizer
from mint.trainer.pretrain import PretrainConfig, PreTrainer
from mint.utils.device import Device, DeviceConfig
from mint.utils.logger import logger


@dataclass
class MetaConfig(Config):
    train: PretrainConfig
    model: GemmaConfig
    optim: MuonAdamWConfig
    device: DeviceConfig
    dl: DataloaderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gemma model with distributed dataloader")
    parser.add_argument(
        "--config", type=str, default="config_d12.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--min-shards",
        type=int,
        default=2,
        help="Minimum shards required before starting training",
    )
    parser.add_argument(
        "--max-shards-to-wait",
        type=int,
        default=-1,
        help="Maximum shards to wait for (-1 = wait indefinitely)",
    )
    args = parser.parse_args()

    tokenizer = TikTokenizer()
    config: MetaConfig = MetaConfig.from_toml(toml_path=args.config)

    if config.model.vocab_size != tokenizer.vocab_size:
        logger.error(
            f"Config vocab_size ({config.model.vocab_size}) does not match tokenizer vocab_size ({tokenizer.vocab_size})"
        )
        logger.error(f"Please update the config file to set vocab_size: {tokenizer.vocab_size}")  # noqa: S608
        raise ValueError(
            f"Vocab size mismatch: config={config.model.vocab_size}, tokenizer={tokenizer.vocab_size}"
        )

    device = Device(config.device)

    model: nn.Module
    if config.train.use_meta_device:
        logger.info("Initializing model on meta device...")
        with torch.device("meta"):
            model = Gemma(config.model, gradient_checkpointing=config.train.gradient_checkpointing)
        logger.info(f"Model parameters: {model.get_num_params():,}")
        model = device.move_to_device(model, from_meta=True)
    else:
        model = Gemma(config.model)
        logger.info(f"Model parameters: {model.get_num_params():,}")
        model = device.move_to_device(model, from_meta=False)

    if config.train.compile_model:
        model = device.compile_model(model, enabled=True)  # type: ignore[assignment]

    optimizer = configure_optimizer(
        model,
        model_cfg=config.model,
        optim_cfg=config.optim,
        sched_cfg=config.train.sched,
    )
    dataloader: DistributedDataloader = DistributedBOSBestfitPretrainDataloader(
        device=device,
        config=config.dl,
        tokenizer=tokenizer,
        min_shards_required=args.min_shards,
        max_shards_to_wait=args.max_shards_to_wait,
    )

    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")
    logger.info(f"Dataloader min_shards: {args.min_shards}")

    trainer = PreTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        config=config.train,
        tokenizer=tokenizer,
    )

    # LoRA().apply(model=model, target_modules=[], r=8, alpha=16)  # noqa: ERA001
    trainer.train()


if __name__ == "__main__":
    main()
