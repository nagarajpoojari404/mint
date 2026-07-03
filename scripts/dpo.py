import argparse
from dataclasses import dataclass

import torch
from torch import nn

from mint.config.base import Config
from mint.data.dataloader import DataloaderConfig
from mint.data.dist.dpo import DistributedDPODataloader
from mint.nn.models import Gemma, GemmaConfig, configure_optimizer
from mint.optim.muon_adamw import MuonAdamWConfig
from mint.tokenizer import TikTokenizer
from mint.trainer.dpo import DPOConfig, DPOTrainer
from mint.utils.checkpointer import Checkpointer
from mint.utils.device import Device, DeviceConfig
from mint.utils.logger import logger


@dataclass
class MetaConfig(Config):
    train: DPOConfig
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
        "--ref-model",
        type=str,
        default=-1,
        help="path to ref model checkpoint",
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

    policy_model: nn.Module
    if config.train.use_meta_device:
        logger.info("Initializing model on meta device...")
        with torch.device("meta"):
            policy_model = Gemma(
                config.model, gradient_checkpointing=config.train.gradient_checkpointing
            )
            reference_model = Gemma(
                config.model, gradient_checkpointing=config.train.gradient_checkpointing
            )
        logger.info(f"Policy Model parameters: {policy_model.get_num_params():,}")
        logger.info(f"Reference Model parameters: {policy_model.get_num_params():,}")
        policy_model = device.move_to_device(policy_model, from_meta=True)
        reference_model = device.move_to_device(reference_model, from_meta=True)
        reference_model.requires_grad_(requires_grad=False)
    else:
        raise ValueError("use meta device")

    if config.train.compile_model:
        policy_model = device.compile_model(policy_model, enabled=True)  # type: ignore[assignment]

    Checkpointer.load_model(reference_model, checkpoint_path=args.ref_model)

    optimizer = configure_optimizer(
        policy_model,
        model_cfg=config.model,
        optim_cfg=config.optim,
        sched_cfg=config.train.sched,
    )
    dataloader: DistributedDPODataloader = DistributedDPODataloader(
        device=device,
        config=config.dl,
        tokenizer=tokenizer,
        filename="hh_rlhf_formatted.jsonl",
    )

    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")
    logger.info(f"Dataloader min_shards: {args.min_shards}")

    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
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
